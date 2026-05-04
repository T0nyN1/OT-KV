# baselines/streamingllm.py
import torch

from evaluation.models.base_cache import BaseCompressCache


class StreamingLLMCache(BaseCompressCache):
    """
    StreamingLLM cache policy.

    Keep attention sinks from the beginning and a fixed-size recent window.
    This policy does not need attention scores.
    """

    def __init__(self, compression_ratio: float = 1.0, recent_window: int = 256,
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
            keep_indices = self._select_keep_indices(
                positions=positions,
                total_tokens=total_tokens,
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
            keep_indices = self._select_keep_indices(
                positions=positions,
                total_tokens=total_after,
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
                             force_keep: torch.Tensor = None) -> torch.Tensor:
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

        if not keep_mask.any():
            keep_mask[-1] = True

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
