# baselines/echokv.py
import math
from typing import Optional

import torch
import torch.nn.functional as F

from evaluation.models.base_cache import BaseCompressCache


class EchoKVCache(BaseCompressCache):
    """
    Training-free EchoKV-style cache policy.

    Keep a fixed recent window, optional sink tokens, and representative older
    tokens selected by key-space similarity. This is an eviction-only baseline:
    it does not add learned reconstruction modules.
    """

    def __init__(self, compression_ratio: float = 0.5, recent_window: int = 256,
                 sink_size: int = 0, max_representative_scan: Optional[int] = None,
                 **kwargs):
        if not 0 < compression_ratio <= 1:
            raise ValueError("compression_ratio must be in (0, 1].")
        if recent_window < 0:
            raise ValueError("recent_window must be non-negative.")
        if sink_size < 0:
            raise ValueError("sink_size must be non-negative.")
        if max_representative_scan is not None and max_representative_scan <= 0:
            raise ValueError("max_representative_scan must be positive when provided.")

        super().__init__(compression_ratio=compression_ratio, **kwargs)
        self.recent_window = int(recent_window)
        self.sink_size = int(sink_size)
        self.max_representative_scan = max_representative_scan

        self.token_positions = {}
        self.total_seen_tokens = {}

    def on_prefill(self, key_states: torch.Tensor, value_states: torch.Tensor,
                   layer_idx: int, cache_kwargs: dict):
        self.current_attention_scores.pop(layer_idx, None)
        return self._process_new_tokens(
            key_states=key_states,
            value_states=value_states,
            layer_idx=layer_idx,
            compress=False,
            force_new_tokens=False,
        )

    def on_prefill_end(self):
        for layer_idx in list(self.token_positions.keys()):
            positions = self.token_positions[layer_idx]
            total_tokens = self.total_seen_tokens.get(layer_idx, positions.numel())
            key_cache = self.key_cache[layer_idx] if layer_idx < len(self.key_cache) else None
            keep_indices = self._select_keep_indices(
                positions=positions,
                total_tokens=total_tokens,
                key_states=key_cache,
            )

            self._prune_existing_cache(layer_idx, keep_indices)
            self.token_positions[layer_idx] = positions.index_select(0, keep_indices)

    def on_decode_step(self, key_states: torch.Tensor, value_states: torch.Tensor,
                       layer_idx: int, cache_kwargs: dict):
        self.current_attention_scores.pop(layer_idx, None)
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

        prev_total = self.total_seen_tokens.get(layer_idx, 0)
        total_after = prev_total + q_len
        existing_len = self.token_positions[layer_idx].numel()

        new_positions = torch.arange(prev_total, total_after, device=device, dtype=torch.long)
        positions = torch.cat([self.token_positions[layer_idx].to(device), new_positions])

        if compress:
            force_keep = None
            if force_new_tokens:
                force_keep = torch.arange(existing_len, existing_len + q_len,
                                          device=device, dtype=torch.long)
            combined_keys = self._combined_key_states(layer_idx, key_states)
            keep_indices = self._select_keep_indices(
                positions=positions,
                total_tokens=total_after,
                key_states=combined_keys,
                force_keep=force_keep,
            )
        else:
            keep_indices = torch.arange(positions.numel(), device=device, dtype=torch.long)

        existing_keep = keep_indices[keep_indices < existing_len]
        new_keep = keep_indices[keep_indices >= existing_len] - existing_len

        self._prune_existing_cache(layer_idx, existing_keep)
        self.token_positions[layer_idx] = positions.index_select(0, keep_indices)
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
        if layer_idx not in self.total_seen_tokens:
            self.total_seen_tokens[layer_idx] = 0

    def _select_keep_indices(self, positions: torch.Tensor, total_tokens: int,
                             key_states: Optional[torch.Tensor],
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
        older_budget = self._older_budget(total_tokens, older_candidates.numel())

        if older_budget >= older_candidates.numel():
            keep_mask[older_candidates] = True
        elif older_budget > 0:
            representatives = self._select_representatives(
                key_states=key_states,
                candidates=older_candidates,
                budget=older_budget,
            )
            keep_mask[representatives] = True

        if not keep_mask.any():
            keep_mask[-1] = True

        return torch.nonzero(keep_mask, as_tuple=False).flatten().sort().values

    def _older_budget(self, total_tokens: int, available_older_tokens: int) -> int:
        protected = self.sink_size + self.recent_window
        theoretical_older = max(0, total_tokens - protected)
        budget = math.floor(theoretical_older * self.compression_ratio)
        return min(max(0, budget), int(available_older_tokens))

    def _select_representatives(self, key_states: Optional[torch.Tensor],
                                candidates: torch.Tensor, budget: int) -> torch.Tensor:
        if candidates.numel() <= budget:
            return candidates
        if budget <= 0:
            return torch.empty(0, device=candidates.device, dtype=torch.long)
        if key_states is None:
            return self._evenly_spaced_candidates(candidates, budget)

        features = self._key_features(key_states)
        if features is None or features.shape[0] <= int(candidates.max().item()):
            return self._evenly_spaced_candidates(candidates, budget)

        if self.max_representative_scan is not None and candidates.numel() > self.max_representative_scan:
            candidates = self._evenly_spaced_candidates(candidates, self.max_representative_scan)
            budget = min(budget, candidates.numel())

        features = features.to(device=candidates.device)
        candidate_features = features.index_select(0, candidates)

        picked = []
        chunk_offsets = torch.tensor_split(
            torch.arange(candidates.numel(), device=candidates.device, dtype=torch.long),
            budget,
        )
        for offsets in chunk_offsets:
            if offsets.numel() == 0:
                continue
            chunk_features = candidate_features.index_select(0, offsets)
            centroid = F.normalize(chunk_features.mean(dim=0, keepdim=True), dim=-1)
            similarities = torch.matmul(chunk_features, centroid.squeeze(0))
            best_offset = offsets[torch.argmax(similarities)]
            picked.append(best_offset)

        if not picked:
            return self._evenly_spaced_candidates(candidates, budget)

        picked_offsets = torch.stack(picked)
        return candidates.index_select(0, picked_offsets).sort().values

    @staticmethod
    def _key_features(key_states: torch.Tensor) -> Optional[torch.Tensor]:
        if key_states is None or key_states.numel() == 0:
            return None

        keys = key_states.detach().float()
        seq_dim = keys.dim() - 2
        keys = keys.transpose(seq_dim, -2)
        seq_len = keys.shape[-2]
        keys = keys.reshape(-1, seq_len, keys.shape[-1]).mean(dim=0)
        return F.normalize(keys, dim=-1)

    @staticmethod
    def _evenly_spaced_candidates(candidates: torch.Tensor, budget: int) -> torch.Tensor:
        if candidates.numel() <= budget:
            return candidates
        offsets = torch.linspace(
            0,
            candidates.numel() - 1,
            steps=budget,
            device=candidates.device,
        ).round().to(dtype=torch.long)
        return candidates.index_select(0, offsets).sort().values

    def _combined_key_states(self, layer_idx: int, key_states: torch.Tensor) -> torch.Tensor:
        if layer_idx >= len(self.key_cache):
            return key_states

        existing_keys = self.key_cache[layer_idx]
        if existing_keys is None or existing_keys.numel() == 0:
            return key_states

        return torch.cat([existing_keys, key_states], dim=-2)