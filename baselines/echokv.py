# baselines/echokv.py
from typing import Optional
import torch
import torch.nn.functional as F
from evaluation.models.base_cache import BaseCompressCache


class EchoKVCache(BaseCompressCache):
    """
    基于新框架的 EchoKV 策略。
    不使用注意力分数，仅在需要压缩时实时提取 Middle 区域的 Key 特征进行相似度聚类。
    """

    def __init__(self, max_representative_scan: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.max_representative_scan = max_representative_scan
        self.active_layers = set()

    def on_prefill(self, key_states: torch.Tensor, value_states: torch.Tensor,
                   layer_idx: int, cache_kwargs: dict):
        self.active_layers.add(layer_idx)
        self.current_attention_scores.pop(layer_idx, None)
        return key_states, value_states

    def on_prefill_end(self):
        total_tokens = self.get_seq_length()
        self._update_budget(total_tokens, is_prefill_end=True)
        for layer_idx in self.active_layers:
            self._prune_layer(layer_idx, total_tokens)

    def on_decode_step(self, key_states: torch.Tensor, value_states: torch.Tensor,
                       layer_idx: int, cache_kwargs: dict):
        self.current_attention_scores.pop(layer_idx, None)

        total_after = self.get_seq_length() + key_states.shape[-2]
        self._update_budget(total_after, is_prefill_end=False)
        self._prune_layer(layer_idx, total_after)

        return key_states, value_states

    def _prune_layer(self, layer_idx: int, total_tokens: int):
        middle_k, _ = self.get_middle_cache(layer_idx)
        if middle_k is None or middle_k.numel() == 0:
            return

        layer_middle_budget = self.get_middle_budget(layer_idx, total_tokens)

        if layer_middle_budget >= middle_k.shape[-2]:
            return

        if layer_middle_budget == 0:
            keep_indices = torch.empty(0, dtype=torch.long, device=middle_k.device)
            self.prune_middle_cache(layer_idx, keep_indices)
            return

        # 在 Middle 区域执行 EchoKV 的特征提取和代表性 Token 选择
        candidates = torch.arange(middle_k.shape[-2], device=middle_k.device, dtype=torch.long)
        keep_indices = self._select_representatives(middle_k, candidates, layer_middle_budget)

        self.prune_middle_cache(layer_idx, keep_indices)

    def _select_representatives(self, middle_k: torch.Tensor, candidates: torch.Tensor, budget: int) -> torch.Tensor:
        if candidates.numel() <= budget:
            return candidates

        features = self._key_features(middle_k)
        if features is None:
            return self._evenly_spaced_candidates(candidates, budget)

        if self.max_representative_scan is not None and candidates.numel() > self.max_representative_scan:
            candidates = self._evenly_spaced_candidates(candidates, self.max_representative_scan)
            budget = min(budget, candidates.numel())

        candidate_features = features.index_select(0, candidates)
        picked = []
        chunk_offsets = torch.tensor_split(torch.arange(candidates.numel(), device=candidates.device, dtype=torch.long),
                                           budget)

        for offsets in chunk_offsets:
            if offsets.numel() == 0: continue
            chunk_features = candidate_features.index_select(0, offsets)
            centroid = F.normalize(chunk_features.mean(dim=0, keepdim=True), dim=-1)
            similarities = torch.matmul(chunk_features, centroid.squeeze(0))
            best_offset = offsets[torch.argmax(similarities)]
            picked.append(best_offset)

        if not picked:
            return self._evenly_spaced_candidates(candidates, budget)

        picked_offsets = torch.stack(picked)
        return candidates.index_select(0, picked_offsets).sort().values

    @staticmethod
    def _key_features(key_states: torch.Tensor):
        keys = key_states.detach().float()
        seq_dim = keys.dim() - 2
        keys = keys.transpose(seq_dim, -2)
        seq_len = keys.shape[-2]
        keys = keys.reshape(-1, seq_len, keys.shape[-1]).mean(dim=0)
        return F.normalize(keys, dim=-1)

    @staticmethod
    def _evenly_spaced_candidates(candidates: torch.Tensor, budget: int):
        offsets = torch.linspace(0, candidates.numel() - 1, steps=budget, device=candidates.device).round().to(
            dtype=torch.long)
        return candidates.index_select(0, offsets).sort().values