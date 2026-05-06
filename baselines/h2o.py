import torch

from evaluation.models.base_cache import BaseCompressCache


class H2OCache(BaseCompressCache):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.hh_scores = {}

    def on_prefill(self, key_states: torch.Tensor, value_states: torch.Tensor,
                   layer_idx: int, cache_kwargs: dict):

        device = key_states.device
        q_len = key_states.shape[-2]

        if layer_idx not in self.hh_scores:
            self.hh_scores[layer_idx] = torch.zeros(q_len, device=device, dtype=torch.float32)
        else:

            new_scores = torch.zeros(q_len, device=device, dtype=torch.float32)
            self.hh_scores[layer_idx] = torch.cat([self.hh_scores[layer_idx], new_scores])

        return key_states, value_states

    def on_prefill_end(self):

        total_tokens = self.get_seq_length()
        self._update_budget(total_tokens, is_prefill_end=True)

        for layer_idx in list(self.hh_scores.keys()):
            self._consume_attention_scores(layer_idx)
            self._prune_layer(layer_idx, total_tokens)

    def on_decode_step(self, key_states: torch.Tensor, value_states: torch.Tensor,
                       layer_idx: int, cache_kwargs: dict):

        q_len = key_states.shape[-2]
        device = key_states.device

        self._consume_attention_scores(layer_idx)

        total_tokens = self.get_seq_length()
        total_after = total_tokens + q_len
        self._update_budget(total_after, is_prefill_end=False)

        self._prune_layer(layer_idx, total_after)

        new_scores = torch.zeros(q_len, device=device, dtype=torch.float32)
        self.hh_scores[layer_idx] = torch.cat([self.hh_scores[layer_idx], new_scores])

        return key_states, value_states

    def _prune_layer(self, layer_idx: int, total_tokens: int):

        middle_k, _ = self.get_middle_cache(layer_idx)
        if middle_k is None or middle_k.numel() == 0:
            return

        layer_middle_budget = self.get_middle_budget(layer_idx, total_tokens)

        seq_len = self._get_existing_cache(layer_idx)[0].shape[-2]
        middle_start = self.sink_size
        middle_end = max(self.sink_size, seq_len - self.recent_size)

        if middle_start >= middle_end:
            return

        middle_scores = self.hh_scores[layer_idx][middle_start:middle_end]

        if layer_middle_budget >= middle_scores.shape[0]:
            return

        if layer_middle_budget == 0:
            keep_indices = torch.empty(0, dtype=torch.long, device=middle_scores.device)
        else:
            _, keep_indices = torch.topk(middle_scores, k=layer_middle_budget)
            keep_indices = keep_indices.sort().values

        self.prune_middle_cache(layer_idx, keep_indices)

        sink_scores = self.hh_scores[layer_idx][:middle_start]
        recent_scores = self.hh_scores[layer_idx][middle_end:]
        pruned_middle_scores = middle_scores[keep_indices]

        self.hh_scores[layer_idx] = torch.cat([sink_scores, pruned_middle_scores, recent_scores])

    def _consume_attention_scores(self, layer_idx: int):

        attn_weights = self.current_attention_scores.pop(layer_idx, None)
        if attn_weights is None or layer_idx not in self.hh_scores:
            return

        current_scores = self.hh_scores[layer_idx]
        if current_scores.numel() == 0:
            return

        score_update = self._attention_to_token_scores(attn_weights)
        score_update = score_update.to(device=current_scores.device, dtype=current_scores.dtype)

        usable = min(current_scores.numel(), score_update.numel())
        if usable == 0:
            return

        updated_scores = current_scores.clone()
        updated_scores[:usable] = updated_scores[:usable] + score_update[:usable]
        self.hh_scores[layer_idx] = updated_scores

    @staticmethod
    def _attention_to_token_scores(attn_weights: torch.Tensor) -> torch.Tensor:

        scores = attn_weights.detach().float()
        if scores.dim() == 0:
            return scores.reshape(1)
        reduce_dims = tuple(range(scores.dim() - 1))
        if reduce_dims:
            scores = scores.sum(dim=reduce_dims)
        return scores.reshape(-1)
