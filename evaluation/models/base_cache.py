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

        # [监控] 记录压缩前的历史 Cache 长度
        pre_cache_len = self.get_seq_length(layer_idx)

        # 1. 阶段分发与自定义处理
        if is_prefill or q_len > 1:
            stage_name = "Prefill"
            key_states, value_states = self.process_prefill(key_states, value_states, layer_idx, cache_kwargs)
        else:
            stage_name = "Decode"
            key_states, value_states = self.process_decode_step(key_states, value_states, layer_idx, cache_kwargs)

        # 2. 调用父类方法真正写入 Cache 状态 (self.key_cache 等)
        result = super().update(key_states, value_states, layer_idx, cache_kwargs)

        # [监控 2] 记录写入新 Token / 驱逐老 Token 后的最终 Cache 长度
        post_cache_len = self.get_seq_length(layer_idx)
        if layer_idx == 0:
            log_str = (f"[KV Monitor] Stage: {stage_name:<7} | "
                       f"New Tokens (q_len): {q_len:<4} | "
                       f"Cache Before: {pre_cache_len:<5} | "
                       f"Cache After: {post_cache_len:<5}")
            clear_to_end = "\033[K"
            if stage_name == "Prefill":
                print(f"\r{log_str}{clear_to_end}", flush=True)
            else:
                print(f"\r{log_str}{clear_to_end}", end="", flush=True)

        return result

    def process_prefill(self, key_states, value_states, layer_idx, cache_kwargs):
        return key_states, value_states

    def process_decode_step(self, key_states, value_states, layer_idx, cache_kwargs):
        raise NotImplementedError