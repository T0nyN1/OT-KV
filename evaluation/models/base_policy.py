# evaluation/models/base_policy.py
from abc import ABC, abstractmethod


class BaseKVPolicy(ABC):
    """
    所有 KV Cache 压缩算法的基础统一接口。
    """

    def __init__(self, compression_ratio: float, **kwargs):
        self.compression_ratio = compression_ratio
        self.kwargs = kwargs

    @abstractmethod
    def process_prefill(self, past_key_values, attention_scores=None, **kwargs):
        """
        在 Prefill (Context 编码) 阶段结束后调用。

        Args:
            past_key_values: 当前层的 KV Cache (通常是 tuple of tensors)
            attention_scores: 当前层的注意力分数矩阵 (如果模型配置输出了的话)

        Returns:
            处理后的 past_key_values
        """
        raise NotImplementedError

    @abstractmethod
    def process_decode_step(self, past_key_values, layer_idx: int, attention_scores=None, **kwargs):
        """
        在 Decode (自回归生成) 阶段，每生成一个新 Token 后调用。

        Args:
            past_key_values: 当前层的 KV Cache
            layer_idx: 当前执行的 Transformer 层级
            attention_scores: 这一步生成的单 Token 的注意力分数

        Returns:
            处理/驱逐后的 past_key_values
        """
        raise NotImplementedError
