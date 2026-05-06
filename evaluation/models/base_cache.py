from typing import Union

import torch
from transformers.cache_utils import DynamicCache


class BaseCompressCache(DynamicCache):
    requires_attention: bool = True

    def __init__(self,
                 compression_size: Union[int, float],
                 mode: str = "prefill",
                 sink_size: Union[int, float] = 4,
                 recent_size: Union[int, float] = 256,
                 **kwargs):
        super().__init__()

        self.compression_size = compression_size
        self.mode = mode.lower()

        self._init_sink_size = sink_size
        self._init_recent_size = recent_size
        self.kwargs = kwargs

        self.current_attention_scores = {}

        self.budget = 0
        self.sink_size = 0
        self.recent_size = 0
        self.middle_budget = 0
        self.prefill_length = 0
        self._prefill_finalized = False

        self._decode_step = 0

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, cache_kwargs=None) -> tuple[
        torch.Tensor, torch.Tensor]:
        q_len = key_states.shape[-2]
        is_prefill = q_len > 1

        if is_prefill:
            key_states, value_states = self.on_prefill(key_states, value_states, layer_idx, cache_kwargs)
            if layer_idx == 0:
                self._decode_step = 0
                self._print_monitor("Initialization", q_len, self._get_phys_length(0))
        else:
            if layer_idx == 0:
                flag = self.finalize_prefill()
                if flag:
                    self._print_monitor("Prefill", None, self._get_phys_length(0))
            key_states, value_states = self.on_decode_step(key_states, value_states, layer_idx, cache_kwargs)
            if layer_idx == 0:
                self._print_monitor("Decode", self._decode_step, self._get_phys_length(0))
                self._decode_step += 1

        result = super().update(key_states, value_states, layer_idx, cache_kwargs)
        return result

    def _print_monitor(self, stage: str, step_or_len: int, cache_len: int):
        budget = getattr(self, "budget", "N/A")
        sink = getattr(self, "sink_size", "N/A")
        recent = getattr(self, "recent_size", "N/A")
        middle = getattr(self, "middle_budget", "N/A")

        if stage == "Initialization":
            print(
                f"[KV Monitor] Stage: {stage} | "
                f"Input Tokens: {step_or_len:<4} | "
            )
        elif stage == "Prefill":
            print(
                f"[KV Monitor] Stage: {stage} | "
                f"Cache: {cache_len:<5} | "
                f"Budget: {budget:<4} (Sink:{sink} Middle:{middle} Recent:{recent})"
            )
        elif stage == "Decode":
            log_str = (f"[KV Monitor] Stage: {stage}  | Step: {step_or_len:<4} | "
                       f"Cache: {cache_len:<4} | "
                       f"Budget: {budget:<4} (Sink:{sink} Middle:{middle} Recent:{recent})")
            print(f"\r{log_str}\033[K", end="", flush=True)

    def finalize_prefill(self):
        if self._prefill_finalized:
            return False

        if self.prefill_length > 0:
            self._prefill_finalized = True
            return False

        if self.get_seq_length() == 0:
            return False

        self.on_prefill_end()
        self._prefill_finalized = True
        return True

    def on_prefill(self, key_states, value_states, layer_idx, cache_kwargs):
        raise NotImplementedError

    def on_decode_step(self, key_states, value_states, layer_idx, cache_kwargs):
        raise NotImplementedError

    def on_prefill_end(self):
        raise NotImplementedError

    def reduce_attention(self, attn_weights: torch.Tensor) -> torch.Tensor:
        assert attn_weights is not None
        return attn_weights.sum(dim=-2).detach()

    def _update_budget(self, total_tokens: int, is_prefill_end: bool = False):
        if is_prefill_end:
            self.prefill_length = total_tokens

            if isinstance(self.compression_size, int):
                self.budget = self.compression_size
            elif isinstance(self.compression_size, float):
                self.budget = int(total_tokens * self.compression_size)

            if isinstance(self._init_sink_size, int):
                self.sink_size = self._init_sink_size
            elif isinstance(self._init_sink_size, float):
                self.sink_size = int(self.budget * self._init_sink_size)

            if isinstance(self._init_recent_size, int):
                self.recent_size = self._init_recent_size
            elif isinstance(self._init_recent_size, float):
                self.recent_size = int(self.budget * self._init_recent_size)

            self.middle_budget = max(0, self.budget - self.sink_size - self.recent_size)


        else:
            if self.mode == "entire" and isinstance(self.compression_size, float):
                self.budget = int(total_tokens * self.compression_size)
                self.middle_budget = max(0, self.budget - self.sink_size - self.recent_size)

    def get_middle_budget(self, layer_idx: int, total_tokens: int) -> int:
        if self.mode == "entire" and isinstance(self.compression_size, float):
            budget = int(total_tokens * self.compression_size)
            return max(0, budget - self.sink_size - self.recent_size)

        return self.middle_budget

    def get_middle_cache(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        k, v = self._get_existing_cache(layer_idx)
        if k is None or v is None:
            return None, None

        seq_len = k.shape[-2]
        middle_start = self.sink_size
        middle_end = max(self.sink_size, seq_len - self.recent_size)

        if middle_start >= middle_end:
            return None, None

        return k[..., middle_start:middle_end, :], v[..., middle_start:middle_end, :]

    def prune_middle_cache(self, layer_idx: int, keep_middle_indices: torch.Tensor):
        k, v = self._get_existing_cache(layer_idx)
        if k is None or v is None:
            return

        seq_len = k.shape[-2]
        middle_start = self.sink_size
        middle_end = max(self.sink_size, seq_len - self.recent_size)

        sink_k, sink_v = k[..., :middle_start, :], v[..., :middle_start, :]
        recent_k, recent_v = k[..., middle_end:, :], v[..., middle_end:, :]

        if middle_start < middle_end:
            middle_k, middle_v = k[..., middle_start:middle_end, :], v[..., middle_start:middle_end, :]
            keep_middle_indices = keep_middle_indices.to(device=k.device, dtype=torch.long)
            pruned_middle_k = middle_k.index_select(-2, keep_middle_indices)
            pruned_middle_v = middle_v.index_select(-2, keep_middle_indices)
        else:
            pruned_middle_k = k.new_empty((*k.shape[:-2], 0, k.shape[-1]))
            pruned_middle_v = v.new_empty((*v.shape[:-2], 0, v.shape[-1]))

        new_k = torch.cat([sink_k, pruned_middle_k, recent_k], dim=-2)
        new_v = torch.cat([sink_v, pruned_middle_v, recent_v], dim=-2)

        self._replace_existing_cache(layer_idx, new_k, new_v)

    def replace_middle_cache(self, layer_idx: int, new_middle_k: torch.Tensor, new_middle_v: torch.Tensor):
        k, v = self._get_existing_cache(layer_idx)
        if k is None or v is None:
            return

        seq_len = k.shape[-2]
        middle_start = self.sink_size
        middle_end = max(self.sink_size, seq_len - self.recent_size)

        sink_k, sink_v = k[..., :middle_start, :], v[..., :middle_start, :]
        recent_k, recent_v = k[..., middle_end:, :], v[..., middle_end:, :]

        new_k = torch.cat([sink_k, new_middle_k, recent_k], dim=-2)
        new_v = torch.cat([sink_v, new_middle_v, recent_v], dim=-2)

        self._replace_existing_cache(layer_idx, new_k, new_v)

    def _get_existing_cache(self, layer_idx: int):
        if hasattr(self, "layers"):
            if layer_idx >= len(self.layers):
                return None, None
            layer = self.layers[layer_idx]
            return layer.keys, layer.values

        if hasattr(self, "key_cache"):
            if layer_idx >= len(self.key_cache):
                return None, None
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        return None, None

    def _replace_existing_cache(self, layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor):
        if key_states is None or value_states is None:
            return

        if hasattr(self, "layers"):
            if layer_idx >= len(self.layers):
                return
            layer = self.layers[layer_idx]
            layer.keys = key_states
            layer.values = value_states
            self._set_layer_length(layer, key_states.shape[-2])
            return

        if hasattr(self, "key_cache"):
            if layer_idx >= len(self.key_cache):
                return
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states

    @staticmethod
    def _set_layer_length(layer, seq_len: int):
        if hasattr(layer, "cumulative_length"):
            if isinstance(layer.cumulative_length, torch.Tensor):
                layer.cumulative_length.fill_(seq_len)
            else:
                layer.cumulative_length = seq_len
        if hasattr(layer, "cumulative_length_int"):
            layer.cumulative_length_int = seq_len

    def _get_phys_length(self, layer_idx=0):
        if hasattr(self, "layers"):
            if layer_idx < len(self.layers):
                k_tensor = self.layers[layer_idx].keys
                if k_tensor is not None and k_tensor.numel() > 0:
                    return k_tensor.shape[-2]
        elif hasattr(self, "key_cache"):
            k_cache = self.key_cache
            if layer_idx < len(k_cache) and k_cache[layer_idx] is not None:
                if k_cache[layer_idx].numel() > 0:
                    return k_cache[layer_idx].shape[-2]
        return self.get_seq_length(layer_idx)
