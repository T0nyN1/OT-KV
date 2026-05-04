# baselines/pyramidkv.py
import math
from typing import Optional

import torch

from evaluation.models.base_cache import BaseCompressCache


class PyramidKVCache(BaseCompressCache):
    """
    PyramidKV-style cache policy.

    It uses attention top-k like SnapKV, but assigns larger older-token budgets
    to lower layers and smaller budgets to upper layers.
    """

    def __init__(self, compression_ratio: float = 0.5, recent_window: int = 256,
                 sink_size: int = 0, observation_window: Optional[int] = None,
                 pyramid_low_scale: float = 1.5, pyramid_high_scale: float = 0.5,
                 **kwargs):
        if not 0 < compression_ratio <= 1:
            raise ValueError("compression_ratio must be in (0, 1].")
        if recent_window < 0:
            raise ValueError("recent_window must be non-negative.")
        if sink_size < 0:
            raise ValueError("sink_size must be non-negative.")
        if observation_window is not None and observation_window <= 0:
            raise ValueError("observation_window must be positive when provided.")
        if pyramid_low_scale <= 0 or pyramid_high_scale <= 0:
            raise ValueError("pyramid scales must be positive.")

        super().__init__(compression_ratio=compression_ratio, **kwargs)
        self.recent_window = int(recent_window)
        self.sink_size = int(sink_size)
        self.observation_window = int(observation_window or max(1, recent_window))
        self.pyramid_low_scale = float(pyramid_low_scale)
        self.pyramid_high_scale = float(pyramid_high_scale)

        self.token_positions = {}
        self.attention_scores = {}
        self.total_seen_tokens = {}
        self.num_layers = None

    def on_prefill(self, key_states: torch.Tensor, value_states: torch.Tensor,
                   layer_idx: int, cache_kwargs: dict):
        self.current_attention_scores.pop(layer_idx, None)
        self.num_layers = max(self.num_layers or 0, layer_idx + 1)
        return self._process_new_tokens(
            key_states=key_states,
            value_states=value_states,
            layer_idx=layer_idx,
            compress=False,
            force_new_tokens=False,
        )

    def on_prefill_end(self):
        if self.token_positions:
            self.num_layers = max(self.token_positions.keys()) + 1

        for layer_idx in list(self.token_positions.keys()):
            self._consume_attention_scores(layer_idx)

            positions = self.token_positions[layer_idx]
            scores = self.attention_scores[layer_idx]
            total_tokens = self.total_seen_tokens.get(layer_idx, positions.numel())
            keep_indices = self._select_keep_indices(
                positions=positions,
                scores=scores,
                total_tokens=total_tokens,
                layer_idx=layer_idx,
            )

            self._prune_existing_cache(layer_idx, keep_indices)
            self.token_positions[layer_idx] = positions.index_select(0, keep_indices)
            self.attention_scores[layer_idx] = scores.index_select(0, keep_indices)

    def on_decode_step(self, key_states: torch.Tensor, value_states: torch.Tensor,
                       layer_idx: int, cache_kwargs: dict):
        self.num_layers = max(self.num_layers or 0, layer_idx + 1)
        return self._process_new_tokens(
            key_states=key_states,
            value_states=value_states,
            layer_idx=layer_idx,
            compress=True,
            force_new_tokens=True,
        )

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
        scores = torch.cat([self.attention_scores[layer_idx].to(device), new_scores])

        if compress:
            force_keep = None
            if force_new_tokens:
                force_keep = torch.arange(existing_len, existing_len + q_len,
                                          device=device, dtype=torch.long)
            keep_indices = self._select_keep_indices(
                positions=positions,
                scores=scores,
                total_tokens=total_after,
                layer_idx=layer_idx,
                force_keep=force_keep,
            )
        else:
            keep_indices = torch.arange(positions.numel(), device=device, dtype=torch.long)

        existing_keep = keep_indices[keep_indices < existing_len]
        new_keep = keep_indices[keep_indices >= existing_len] - existing_len

        self._prune_existing_cache(layer_idx, existing_keep)
        self.token_positions[layer_idx] = positions.index_select(0, keep_indices)
        self.attention_scores[layer_idx] = scores.index_select(0, keep_indices)
        self.total_seen_tokens[layer_idx] = total_after

        if new_keep.numel() == q_len:
            return key_states, value_states

        new_keep = new_keep.to(device=key_states.device, dtype=torch.long)
        return (
            key_states.index_select(-2, new_keep),
            value_states.index_select(-2, new_keep),
        )

    def _ensure_layer_state(self, layer_idx: int, device):
        if layer_idx not in self.token_positions:
            self.token_positions[layer_idx] = torch.empty(0, device=device, dtype=torch.long)
        if layer_idx not in self.attention_scores:
            self.attention_scores[layer_idx] = torch.empty(0, device=device, dtype=torch.float32)
        if layer_idx not in self.total_seen_tokens:
            self.total_seen_tokens[layer_idx] = 0

    def _consume_attention_scores(self, layer_idx: int):
        attn_weights = self.current_attention_scores.pop(layer_idx, None)
        if attn_weights is None or layer_idx not in self.attention_scores:
            return

        current_scores = self.attention_scores[layer_idx]
        if current_scores.numel() == 0:
            return

        score_update = self._attention_to_token_scores(attn_weights)
        score_update = score_update.to(device=current_scores.device, dtype=current_scores.dtype)

        usable = min(current_scores.numel(), score_update.numel())
        if usable == 0:
            return

        updated_scores = current_scores.clone()
        updated_scores[:usable] = updated_scores[:usable] + score_update[:usable]
        self.attention_scores[layer_idx] = updated_scores

    def _attention_to_token_scores(self, attn_weights: torch.Tensor) -> torch.Tensor:
        scores = attn_weights.detach().float()
        if scores.dim() >= 4 and self.observation_window > 0:
            scores = scores[..., -self.observation_window:, :]
        if scores.dim() == 0:
            return scores.reshape(1)
        reduce_dims = tuple(range(scores.dim() - 1))
        if reduce_dims:
            scores = scores.sum(dim=reduce_dims)
        return scores.reshape(-1)

    def _select_keep_indices(self, positions: torch.Tensor, scores: torch.Tensor,
                             total_tokens: int, layer_idx: int,
                             force_keep: Optional[torch.Tensor] = None) -> torch.Tensor:
        num_tokens = positions.numel()
        if num_tokens == 0:
            return torch.empty(0, device=positions.device, dtype=torch.long)

        keep_mask = torch.zeros(num_tokens, device=positions.device, dtype=torch.bool)

        if force_keep is not None and force_keep.numel() > 0:
            force_keep = force_keep.to(device=positions.device, dtype=torch.long)
            keep_mask[force_keep] = True

        if self.sink_size > 0:
            keep_mask |= positions < self.sink_size

        if self.recent_window > 0:
            recent_start = max(0, total_tokens - self.recent_window)
            keep_mask |= positions >= recent_start

        older_candidates = torch.nonzero(~keep_mask, as_tuple=False).flatten()
        older_budget = self._older_budget(total_tokens, older_candidates.numel(), layer_idx)

        if older_budget >= older_candidates.numel():
            keep_mask[older_candidates] = True
        elif older_budget > 0:
            _, top_offsets = torch.topk(scores[older_candidates], k=older_budget)
            keep_mask[older_candidates[top_offsets]] = True

        if not keep_mask.any():
            keep_mask[-1] = True

        return torch.nonzero(keep_mask, as_tuple=False).flatten().sort().values

    def _older_budget(self, total_tokens: int, available_older_tokens: int,
                      layer_idx: int) -> int:
        protected = self.sink_size + self.recent_window
        theoretical_older = max(0, total_tokens - protected)
        effective_ratio = self._layer_compression_ratio(layer_idx)
        budget = math.floor(theoretical_older * effective_ratio)
        return min(max(0, budget), int(available_older_tokens))

    def _layer_compression_ratio(self, layer_idx: int) -> float:
        num_layers = max(1, int(self.num_layers or (layer_idx + 1)))
        if num_layers == 1:
            return self.compression_ratio

        depth = layer_idx / float(num_layers - 1)
        scale = self.pyramid_low_scale + depth * (self.pyramid_high_scale - self.pyramid_low_scale)
        mean_scale = 0.5 * (self.pyramid_low_scale + self.pyramid_high_scale)
        if mean_scale <= 0:
            return self.compression_ratio
        return min(1.0, max(0.0, self.compression_ratio * scale / mean_scale))
