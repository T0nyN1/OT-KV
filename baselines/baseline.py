# baselines/baseline.py
import torch
from evaluation.models.base_cache import BaseCompressCache


class BaselineCache(BaseCompressCache):
    """
    Baseline 缓存策略 (无压缩/全量保留)。
    保留所有的 KV Cache，不对其进行任何驱逐。
    这是模型完整保留上下文时的理论上限性能（PPL 最低）。
    """

    def __init__(self, **kwargs):
        # Baseline 不需要压缩，为了兼容基类的初始化签名，我们将压缩率设为 1.0 (保留 100%)
        super().__init__(compression_ratio=1.0, **kwargs)

    def process_prefill(self, key_states: torch.Tensor, value_states: torch.Tensor,
                        layer_idx: int, cache_kwargs: dict):
        """
        Prefill 阶段 (Context 编码)：
        直接返回传入的 KV 张量，不进行任何截断。
        """
        # (可选) Baseline 不依赖 Attention 分数，如果 Hook 抓取了，可以直接清空以节省显存
        self.current_attention_scores.pop(layer_idx, None)

        return key_states, value_states

    def process_decode_step(self, key_states: torch.Tensor, value_states: torch.Tensor,
                            layer_idx: int, cache_kwargs: dict):
        """
        Decode 阶段 (自回归生成单 Token)：
        在这里，key_states 和 value_states 是当前步生成的【最新 1 个 Token】的 KV。
        因为是 Baseline，我们不需要驱逐历史 Token。
        直接返回它们，父类 (DynamicCache) 的 update 方法会自动把它们 append 到整体缓存中。
        """
        # Baseline 不依赖 Attention 分数，及时清空以释放显存 (避免 OOM)
        self.current_attention_scores.pop(layer_idx, None)

        return key_states, value_states