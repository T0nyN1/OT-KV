# baselines/baseline.py
import torch
from evaluation.models.base_cache import BaseCompressCache


class BaselineCache(BaseCompressCache):
    """
    基于新框架重写的 Baseline 缓存策略 (无压缩/全量保留)。
    保留所有的 KV Cache，不对其进行任何驱逐。
    这是模型完整保留上下文时的理论上限性能（PPL 最低）。
    """

    requires_attention = False

    def __init__(self, **kwargs):
        # Baseline 不需要压缩，强行覆盖关键参数：
        # 1. 压缩比例设为 1.0 (100% 保留)
        # 2. 模式设为 entire (允许无限动态扩容)
        # 3. Sink 和 Recent 清零 (因为不需要切片保护，全部都是安全的)
        kwargs["compression_size"] = 1.0
        kwargs["mode"] = "entire"
        kwargs["sink_size"] = 0
        kwargs["recent_size"] = 0

        super().__init__(**kwargs)

    def on_prefill(self, key_states: torch.Tensor, value_states: torch.Tensor,
                   layer_idx: int, cache_kwargs: dict):
        """
        Prefill 阶段 (Context 编码)：
        直接返回传入的 KV 张量，不进行任何截断。
        """
        # 及时清空 Hook 抓取到的 Attention 分数，避免显存泄漏 (OOM)
        self.current_attention_scores.pop(layer_idx, None)
        return key_states, value_states

    def on_prefill_end(self):
        """
        Prefill 结束时，只是单纯触发一下基类的预算更新逻辑，
        这样如果开启了 show_monitor=True，就能在控制台正确打印初始容量。
        """
        total_tokens = self.get_seq_length()
        self._update_budget(total_tokens, is_prefill_end=True)

    def on_decode_step(self, key_states: torch.Tensor, value_states: torch.Tensor,
                       layer_idx: int, cache_kwargs: dict):
        """
        Decode 阶段 (自回归生成单 Token)：
        因为是 Baseline，我们不需要调用 prune_middle_cache 驱逐任何历史 Token。
        直接返回最新生成的 KV 即可，基类会自动将其追加到缓存末尾。
        """
        self.current_attention_scores.pop(layer_idx, None)

        # 仅仅是为了更新监控器中的 Budget 数字，保持与物理长度同步
        total_after = self.get_seq_length() + key_states.shape[-2]
        self._update_budget(total_after, is_prefill_end=False)

        return key_states, value_states