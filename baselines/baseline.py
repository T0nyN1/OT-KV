# baselines/baseline.py
from evaluation.models.base_policy import BaseKVPolicy


class BaselinePolicy(BaseKVPolicy):
    """
    全量 KV Cache (Baseline) 策略。
    继承自 BaseKVPolicy，但不执行任何驱逐操作，用于作为对比基准。
    """

    def __init__(self):
        # 对于 Baseline，压缩率固定为 1.0 (不压缩)
        super().__init__(compression_ratio=1.0)

    def process_prefill(self, past_key_values, attention_scores=None, **kwargs):
        # Prefill 阶段：原样返回
        return past_key_values

    def process_decode_step(self, past_key_values, layer_idx, attention_scores=None, **kwargs):
        # Decode 阶段：原样返回，不进行任何驱逐
        return past_key_values
