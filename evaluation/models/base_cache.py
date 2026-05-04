# evaluation/models/base_cache.py
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
