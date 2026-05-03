# evaluation/models/cache_manager.py
from typing import Optional

from evaluation.models.base_policy import BaseKVPolicy


class KVCacheManager:
    """
    KV Cache 状态管理器。
    负责拦截模型的 forward 输出，并分发给具体的压缩 Policy 进行处理。
    """

    def __init__(self, policy: Optional[BaseKVPolicy] = None):
        self.policy = policy

    def on_layer_forward_end(self, past_key_values, layer_idx: int, is_prefill: bool, attention_scores=None):
        """
        此方法将被注入到 HF 模型的 Attention forward 结尾。
        """
        # 如果没有启用任何压缩策略（Baseline），直接返回原 Cache
        if self.policy is None:
            return past_key_values

        if is_prefill:
            # 必须传入 layer_idx，否则 kwargs.get('layer_idx', 0) 会让所有层覆盖第 0 层！
            return self.policy.process_prefill(
                past_key_values,
                layer_idx=layer_idx,  # <--- ADD THIS
                attention_scores=attention_scores
            )
        else:
            # 执行 Decode 阶段的策略 (例如 H2O / OT-KV 的逐层淘汰)
            return self.policy.process_decode_step(
                past_key_values,
                layer_idx=layer_idx,
                attention_scores=attention_scores
            )
