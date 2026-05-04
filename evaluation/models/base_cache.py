# evaluation/models/base_cache.py
import math

import torch
from transformers.cache_utils import DynamicCache


class BaseCompressCache(DynamicCache):
    """
    所有 KV Cache 压缩策略的基类，继承自 HuggingFace 原生 DynamicCache。
    包含实时的 Cache 长度监控功能。
    """

    def __init__(self, compression_ratio: float, **kwargs):
        super().__init__()
        self.compression_ratio = compression_ratio
        self.kwargs = kwargs

        # 暂存当前 step 各层的 attention scores
        self.current_attention_scores = {}
        self.budget = None

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, cache_kwargs=None) -> tuple[
        torch.Tensor, torch.Tensor]:
        q_len = key_states.shape[-2]

        # HF 判定 Prefill 的标准：输入长度 > 1 且缓存为空
        is_prefill = q_len > 1 and layer_idx == 0 and self.get_seq_length() == 0

        # 1. 阶段分发与自定义处理
        if is_prefill or q_len > 1:
            stage_name = "Prefill"
            key_states, value_states = self.on_prefill(key_states, value_states, layer_idx, cache_kwargs)
        else:
            stage_name = "Decode"
            key_states, value_states = self.on_decode_step(key_states, value_states, layer_idx, cache_kwargs)

        # 2. 调用父类方法真正写入 Cache 状态 (self.key_cache 等)
        result = super().update(key_states, value_states, layer_idx, cache_kwargs)

        return result

    def on_prefill(self, key_states, value_states, layer_idx, cache_kwargs):
        raise NotImplementedError

    def on_decode_step(self, key_states, value_states, layer_idx, cache_kwargs):
        raise NotImplementedError

    def on_prefill_end(self):
        raise NotImplementedError

    def _target_budget(self, total_tokens: int, mode="fixed") -> int:
        if mode == "expandable":
            if total_tokens <= 0:
                return 0
            ratio_budget = math.ceil(total_tokens * self.compression_ratio)
            min_budget = min(total_tokens, self.sink_size + 1)
            self.budget = min(total_tokens, max(1, min_budget, ratio_budget))

        elif mode == "fixed":
            if self.budget is None:
                self.budget = math.ceil(total_tokens * self.compression_ratio)
            else:
                pass
        elif mode == "custom":
            fixed_budget = int(self.kwargs.get("target_budget", 256))
            self.budget = fixed_budget
        else:
            raise ValueError(f"Unsupported budget_mode: '{mode}'. Expected 'fixed' or 'expandable'.")

    def _prune_existing_cache(self, layer_idx: int, keep_indices: torch.Tensor):
        # === 适配 Transformers 5.x ===
        if hasattr(self, "layers"):
            if layer_idx >= len(self.layers):
                return

            layer = self.layers[layer_idx]
            k_cache = layer.keys
            v_cache = layer.values

            if k_cache is None or v_cache is None or k_cache.numel() == 0:
                return

            keep_indices = keep_indices.to(device=k_cache.device, dtype=torch.long)
            # 在新架构下，直接替换 DynamicLayer 的内部属性
            self.layers[layer_idx].keys = k_cache.index_select(-2, keep_indices)
            self.layers[layer_idx].values = v_cache.index_select(-2, keep_indices)

        # === 适配 Transformers 4.x (兼容旧版逻辑) ===
        elif hasattr(self, "key_cache"):
            if layer_idx >= len(self.key_cache):
                return

            k_cache = self.key_cache[layer_idx]
            v_cache = self.value_cache[layer_idx]

            if k_cache is None or v_cache is None or k_cache.numel() == 0:
                return

            keep_indices = keep_indices.to(device=k_cache.device, dtype=torch.long)
            self.key_cache[layer_idx] = k_cache.index_select(-2, keep_indices)
            self.value_cache[layer_idx] = v_cache.index_select(-2, keep_indices)
