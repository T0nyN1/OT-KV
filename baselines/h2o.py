# baselines/h2o.py
import math
from typing import Optional

import torch

from evaluation.models.base_cache import BaseCompressCache


class H2OCache(BaseCompressCache):
    """
    Heavy-Hitter Oracle (H2O) KV cache.

    The attention hook in ``EvaluatorHFLM`` observes attention weights after the
    model layer has already called ``Cache.update``. Because of that, this cache
    consumes the attention scores from the previous forward pass and uses them
    to prune the existing cache before appending the current token.
    """

    def __init__(self, compression_ratio: float = 0.5, recent_window: int = 256,
                 sink_size: int = 4, **kwargs):
        if not 0 < compression_ratio <= 1:
            raise ValueError("compression_ratio must be in (0, 1].")
        if recent_window < 0:
            raise ValueError("recent_window must be non-negative.")
        if sink_size < 0:
            raise ValueError("sink_size must be non-negative.")

        super().__init__(compression_ratio=compression_ratio, **kwargs)
        self.recent_window = int(recent_window)
        self.sink_size = int(sink_size)

        self.hh_scores = {}
        self.token_positions = {}
        self.total_seen_tokens = {}

    def on_prefill(self, key_states: torch.Tensor, value_states: torch.Tensor,
                   layer_idx: int, cache_kwargs: dict):
        # Initial prefill attention is only available after this update returns.
        # The wrapper calls compress_after_prefill once hooks have populated
        # scores, so this method only records the prompt state.
        return self._process_new_tokens(
            key_states=key_states,
            value_states=value_states,
            layer_idx=layer_idx,
            compress=False,
            force_new_tokens=False,
        )

    def on_decode_step(self, key_states: torch.Tensor, value_states: torch.Tensor,
                       layer_idx: int, cache_kwargs: dict):
        return self._process_new_tokens(
            key_states=key_states,
            value_states=value_states,
            layer_idx=layer_idx,
            compress=True,
            force_new_tokens=True,
        )

    def on_prefill_end(self):
        """
        Compress prompt KV after the prefill forward pass has produced attention
        scores through the wrapper hook.
        """
        for layer_idx in list(self.token_positions.keys()):
            self._consume_attention_scores(layer_idx)

            positions = self.token_positions[layer_idx]
            scores = self.hh_scores[layer_idx]
            total_tokens = self.total_seen_tokens.get(layer_idx, positions.numel())
            budget = self._target_budget(total_tokens)

            keep_indices = self._select_keep_indices(
                positions=positions,
                scores=scores,
                total_tokens=total_tokens,
                budget=budget,
            )

            self._prune_existing_cache(layer_idx, keep_indices)
            self.token_positions[layer_idx] = positions.index_select(0, keep_indices)
            self.hh_scores[layer_idx] = scores.index_select(0, keep_indices)

    def _process_new_tokens(self, key_states: torch.Tensor, value_states: torch.Tensor,
                            layer_idx: int, compress: bool,
                            force_new_tokens: bool):
        device = key_states.device
        q_len = key_states.shape[-2]

        self._ensure_layer_state(layer_idx, device)
        self._consume_attention_scores(layer_idx)

        prev_total = self.total_seen_tokens.get(layer_idx, 0)
        total_after = prev_total + q_len
        existing_len = self.token_positions[layer_idx].numel()

        new_positions = torch.arange(prev_total, total_after, device=device, dtype=torch.long)
        new_scores = torch.zeros(q_len, device=device, dtype=torch.float32)

        positions = torch.cat([self.token_positions[layer_idx].to(device), new_positions])
        scores = torch.cat([self.hh_scores[layer_idx].to(device), new_scores])

        if compress:
            budget = self._target_budget(total_after)
            force_keep = None
            if force_new_tokens:
                force_keep = torch.arange(existing_len, existing_len + q_len,
                                          device=device, dtype=torch.long)
            keep_indices = self._select_keep_indices(
                positions=positions,
                scores=scores,
                total_tokens=total_after,
                budget=budget,
                force_keep=force_keep,
            )
        else:
            keep_indices = torch.arange(positions.numel(), device=device, dtype=torch.long)

        existing_keep = keep_indices[keep_indices < existing_len]
        new_keep = keep_indices[keep_indices >= existing_len] - existing_len

        self._prune_existing_cache(layer_idx, existing_keep)

        self.token_positions[layer_idx] = positions.index_select(0, keep_indices)
        self.hh_scores[layer_idx] = scores.index_select(0, keep_indices)
        self.total_seen_tokens[layer_idx] = total_after

        if new_keep.numel() == q_len:
            return key_states, value_states

        new_keep = new_keep.to(device=key_states.device)
        return (
            key_states.index_select(-2, new_keep),
            value_states.index_select(-2, new_keep),
        )

    def _ensure_layer_state(self, layer_idx: int, device):
        if layer_idx not in self.hh_scores:
            self.hh_scores[layer_idx] = torch.empty(0, device=device, dtype=torch.float32)
        if layer_idx not in self.token_positions:
            self.token_positions[layer_idx] = torch.empty(0, device=device, dtype=torch.long)
        if layer_idx not in self.total_seen_tokens:
            self.total_seen_tokens[layer_idx] = 0

    def _consume_attention_scores(self, layer_idx: int):
        attn_weights = self.current_attention_scores.pop(layer_idx, None)
        if attn_weights is None or layer_idx not in self.hh_scores:
            return

        current_scores = self.hh_scores[layer_idx]
        if current_scores.numel() == 0:
            return

        score_update = self._attention_to_token_scores(attn_weights)
        score_update = score_update.to(device=current_scores.device, dtype=current_scores.dtype)

        usable = min(current_scores.numel(), score_update.numel())
        if usable == 0:
            return

        updated_scores = current_scores.clone()
        updated_scores[:usable] = updated_scores[:usable] + score_update[:usable]
        self.hh_scores[layer_idx] = updated_scores

    @staticmethod
    def _attention_to_token_scores(attn_weights: torch.Tensor) -> torch.Tensor:
        scores = attn_weights.detach().float()
        if scores.dim() == 0:
            return scores.reshape(1)
        reduce_dims = tuple(range(scores.dim() - 1))
        if reduce_dims:
            scores = scores.sum(dim=reduce_dims)
        return scores.reshape(-1)

    def _target_budget(self, total_tokens: int) -> int:
        if total_tokens <= 0:
            return 0
        ratio_budget = math.ceil(total_tokens * self.compression_ratio)
        min_budget = min(total_tokens, self.sink_size + 1)
        return min(total_tokens, max(1, min_budget, ratio_budget))

    def _select_keep_indices(self, positions: torch.Tensor, scores: torch.Tensor,
                             total_tokens: int, budget: int,
                             force_keep: Optional[torch.Tensor] = None) -> torch.Tensor:
        num_tokens = positions.numel()
        budget = min(max(int(budget), 0), num_tokens)
        if budget >= num_tokens:
            return torch.arange(num_tokens, device=positions.device, dtype=torch.long)
        if budget == 0:
            return torch.empty(0, device=positions.device, dtype=torch.long)

        keep_mask = torch.zeros(num_tokens, device=positions.device, dtype=torch.bool)
        remaining = budget

        def add_ordered(candidate_indices: torch.Tensor, take_from_end: bool = False):
            nonlocal remaining
            if remaining <= 0 or candidate_indices.numel() == 0:
                return
            candidate_indices = candidate_indices[~keep_mask[candidate_indices]]
            if candidate_indices.numel() == 0:
                return
            if candidate_indices.numel() > remaining:
                if take_from_end:
                    candidate_indices = candidate_indices[-remaining:]
                else:
                    candidate_indices = candidate_indices[:remaining]
            keep_mask[candidate_indices] = True
            remaining -= candidate_indices.numel()

        if force_keep is not None:
            add_ordered(force_keep.to(device=positions.device, dtype=torch.long),
                        take_from_end=True)

        if self.sink_size > 0:
            sink_indices = torch.nonzero(positions < self.sink_size, as_tuple=False).flatten()
            add_ordered(sink_indices)

        if self.recent_window > 0:
            recent_start = max(0, total_tokens - self.recent_window)
            recent_indices = torch.nonzero(positions >= recent_start, as_tuple=False).flatten()
            add_ordered(recent_indices, take_from_end=True)

        if remaining > 0:
            hh_candidates = torch.nonzero(~keep_mask, as_tuple=False).flatten()
            if hh_candidates.numel() <= remaining:
                add_ordered(hh_candidates)
            else:
                _, top_offsets = torch.topk(scores[hh_candidates], k=remaining)
                keep_mask[hh_candidates[top_offsets]] = True

        return torch.nonzero(keep_mask, as_tuple=False).flatten().sort().values

    def _prune_existing_cache(self, layer_idx: int, keep_indices: torch.Tensor):
        if layer_idx >= len(self.key_cache):
            return

        key_cache = self.key_cache[layer_idx]
        value_cache = self.value_cache[layer_idx]
        if key_cache is None or value_cache is None:
            return

        keep_indices = keep_indices.to(device=key_cache.device, dtype=torch.long)
        self.key_cache[layer_idx] = key_cache.index_select(-2, keep_indices)
        self.value_cache[layer_idx] = value_cache.index_select(-2, keep_indices)
