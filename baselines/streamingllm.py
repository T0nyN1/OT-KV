# baselines/streamingllm.py
import torch
from evaluation.models.base_cache import BaseCompressCache


class StreamingLLMCache(BaseCompressCache):
    """
    基于新框架的 StreamingLLM 策略。
    无视 Attention 分数，直接清空 Middle 区域，仅保留基类保护的 Sink 和 Recent。
    """

    requires_attention = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.active_layers = set()  # 用于 on_prefill_end 遍历

    def on_prefill(self, key_states: torch.Tensor, value_states: torch.Tensor,
                   layer_idx: int, cache_kwargs: dict):
        self.active_layers.add(layer_idx)
        self.current_attention_scores.pop(layer_idx, None)  # 及时清空节省显存
        return key_states, value_states

    def on_prefill_end(self):
        total_tokens = self.get_seq_length()
        self._update_budget(total_tokens, is_prefill_end=True)
        for layer_idx in self.active_layers:
            self._prune_layer(layer_idx)

    def on_decode_step(self, key_states: torch.Tensor, value_states: torch.Tensor,
                       layer_idx: int, cache_kwargs: dict):
        self.current_attention_scores.pop(layer_idx, None)

        total_after = self.get_seq_length() + key_states.shape[-2]
        self._update_budget(total_after, is_prefill_end=False)
        self._prune_layer(layer_idx)

        return key_states, value_states

    def _prune_layer(self, layer_idx: int):
        middle_k, _ = self.get_middle_cache(layer_idx)
        if middle_k is None or middle_k.numel() == 0:
            return

        # StreamingLLM 的 Middle budget 为 0，传一个空的 keep_indices 交给基类即可
        keep_indices = torch.empty(0, dtype=torch.long, device=middle_k.device)
        self.prune_middle_cache(layer_idx, keep_indices)