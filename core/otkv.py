from typing import Dict, Tuple, Union

import torch
import torch.nn.functional as F

from evaluation.models.base_cache import BaseCompressCache


DEFAULT_COMPRESSION_RATIO = 0.5


def _validate_transport_mode(transport_mode: str):
    if transport_mode not in {"soft", "hard"}:
        raise ValueError("transport_mode must be 'soft' or 'hard'.")


def sinkhorn_log_space(cost_matrix: torch.Tensor, epsilon: float = 0.01,
                       max_iter: int = 50) -> torch.Tensor:
    """
    Run Sinkhorn in log space for numerical stability under fp16/bf16 inputs.
    """
    n_evict = cost_matrix.shape[-2]
    m_anchor = cost_matrix.shape[-1]

    if n_evict <= 0 or m_anchor <= 0:
        raise ValueError("cost_matrix must have positive evict and anchor dimensions.")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")

    cost_matrix = cost_matrix.float()
    f = torch.zeros_like(cost_matrix[:, :, :, 0])
    g = torch.zeros_like(cost_matrix[:, :, 0, :])

    mu = torch.full((n_evict,), 1.0 / n_evict, device=cost_matrix.device,
                    dtype=cost_matrix.dtype).log()
    nu = torch.full((m_anchor,), 1.0 / m_anchor, device=cost_matrix.device,
                    dtype=cost_matrix.dtype).log()

    for _ in range(max_iter):
        f = epsilon * (mu - torch.logsumexp((g.unsqueeze(-2) - cost_matrix) / epsilon, dim=-1))
        g = epsilon * (nu - torch.logsumexp((f.unsqueeze(-1) - cost_matrix) / epsilon, dim=-2))

    log_t = (f.unsqueeze(-1) + g.unsqueeze(-2) - cost_matrix) / epsilon
    return torch.exp(log_t)


def otkv_compress(key_states: torch.Tensor, value_states: torch.Tensor, budget: int,
                  gamma: float = 1.0, epsilon: float = 0.01,
                  sink_size: int = 4, transport_mode: str = "soft") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select anchor keys and merge evicted values into those anchors with OT.
    Inputs are expected to be [batch, num_heads, seq_len, head_dim].
    """
    _validate_transport_mode(transport_mode)

    if key_states.shape != value_states.shape:
        raise ValueError("key_states and value_states must have the same shape.")
    if key_states.dim() != 4:
        raise ValueError("key_states and value_states must be rank-4 tensors.")

    batch, num_heads, seq_len, head_dim = key_states.shape
    if seq_len == 0:
        return key_states, value_states

    budget = min(max(int(budget), 1), seq_len)
    if seq_len <= budget:
        return key_states, value_states

    sink_size = min(max(int(sink_size), 0), budget, seq_len)
    remaining_budget = budget - sink_size

    key_scores = key_states.float()
    weights = torch.norm(key_scores, dim=-1)
    position_weights = 1.0 / (
        seq_len - torch.arange(seq_len, device=key_states.device, dtype=weights.dtype) + 1.0
    )
    corrected_weights = weights * position_weights.view(1, 1, -1)

    anchor_pieces = []
    if sink_size > 0:
        sink_indices = torch.arange(sink_size, device=key_states.device)
        anchor_pieces.append(sink_indices.view(1, 1, -1).expand(batch, num_heads, -1))

    if remaining_budget > 0:
        _, remaining_indices = torch.topk(
            corrected_weights[:, :, sink_size:],
            remaining_budget,
            dim=-1,
        )
        anchor_pieces.append(remaining_indices + sink_size)

    anchor_indices = torch.cat(anchor_pieces, dim=-1).sort(dim=-1).values
    gather_index = anchor_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    k_anchor = torch.gather(key_states, 2, gather_index)
    v_anchor = torch.gather(value_states, 2, gather_index)
    w_anchor = torch.gather(corrected_weights, 2, anchor_indices).float()

    evict_mask = torch.ones(batch, num_heads, seq_len, device=key_states.device, dtype=torch.bool)
    evict_mask.scatter_(-1, anchor_indices, False)

    k_evict = key_states[evict_mask].view(batch, num_heads, -1, head_dim)
    v_evict = value_states[evict_mask].view(batch, num_heads, -1, head_dim)
    if k_evict.shape[-2] == 0:
        return k_anchor, v_anchor

    dists = 1.0 - torch.matmul(
        F.normalize(k_evict.float(), dim=-1),
        F.normalize(k_anchor.float(), dim=-1).transpose(-1, -2),
    )
    cost_matrix = dists / (w_anchor.unsqueeze(-2).pow(gamma) + 1e-6)

    if transport_mode == "soft":
        transport = sinkhorn_log_space(cost_matrix, epsilon=epsilon)
        v_merged = v_anchor.float() + torch.matmul(transport.transpose(-1, -2), v_evict.float())
    else:
        best_anchor = torch.argmin(cost_matrix, dim=-1)
        v_merged = v_anchor.float().clone()
        v_merged.scatter_add_(
            2,
            best_anchor.unsqueeze(-1).expand(-1, -1, -1, head_dim),
            v_evict.float(),
        )

    return k_anchor, v_merged.to(value_states.dtype)


def _protected_counts(seq_len: int, sink_size: int, recent_size: int) -> Dict[str, int]:
    seq_len = max(int(seq_len), 0)
    sink_keep = min(max(int(sink_size), 0), seq_len)
    remaining_after_sink = max(seq_len - sink_keep, 0)
    recent_keep = min(max(int(recent_size), 0), remaining_after_sink)
    middle_start = sink_keep
    middle_end = seq_len - recent_keep
    middle_len = max(middle_end - middle_start, 0)

    return {
        "sink_keep": sink_keep,
        "recent_keep": recent_keep,
        "middle_start": middle_start,
        "middle_end": middle_end,
        "middle_len": middle_len,
    }


def middle_budget_from_ratio(total_tokens: int, compression_ratio: float,
                             sink_size: int = 4, recent_size: int = 256) -> int:
    counts = _protected_counts(total_tokens, sink_size=sink_size, recent_size=recent_size)
    middle_len = counts["middle_len"]
    ratio = min(max(float(compression_ratio), 0.0), 1.0)

    if middle_len == 0:
        return 0
    if ratio >= 1.0:
        return middle_len
    return min(middle_len, max(int(middle_len * ratio), 1))


def otkv_segmented_compress_to_budget(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    middle_budget: int,
    sink_size: int = 4,
    recent_size: int = 256,
    gamma: float = 1.0,
    epsilon: float = 0.01,
    transport_mode: str = "soft",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Keep sink/recent windows and compress only the materialized middle region.
    ``middle_budget`` is absolute, so decode-time calls do not repeatedly shrink
    the already-compressed middle by the ratio again.
    """
    _validate_transport_mode(transport_mode)

    seq_len = key_states.shape[-2]
    counts = _protected_counts(seq_len, sink_size=sink_size, recent_size=recent_size)
    middle_keep = min(max(int(middle_budget), 0), counts["middle_len"])
    target_len = counts["sink_keep"] + middle_keep + counts["recent_keep"]

    if seq_len <= target_len or counts["middle_len"] <= middle_keep:
        return key_states, value_states

    key_pieces = []
    value_pieces = []

    if counts["sink_keep"] > 0:
        key_pieces.append(key_states[:, :, : counts["sink_keep"], :])
        value_pieces.append(value_states[:, :, : counts["sink_keep"], :])

    if middle_keep > 0:
        middle_k = key_states[:, :, counts["middle_start"] : counts["middle_end"], :]
        middle_v = value_states[:, :, counts["middle_start"] : counts["middle_end"], :]
        if counts["middle_len"] > middle_keep:
            middle_k, middle_v = otkv_compress(
                middle_k,
                middle_v,
                budget=middle_keep,
                gamma=gamma,
                epsilon=epsilon,
                sink_size=0,
                transport_mode=transport_mode,
            )
        key_pieces.append(middle_k)
        value_pieces.append(middle_v)

    if counts["recent_keep"] > 0:
        key_pieces.append(key_states[:, :, seq_len - counts["recent_keep"] :, :])
        value_pieces.append(value_states[:, :, seq_len - counts["recent_keep"] :, :])

    if not key_pieces:
        return key_states[:, :, :0, :], value_states[:, :, :0, :]

    return torch.cat(key_pieces, dim=2), torch.cat(value_pieces, dim=2)


def otkv_segmented_compress(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    compression_ratio: float = DEFAULT_COMPRESSION_RATIO,
    sink_size: int = 4,
    recent_size: int = 256,
    gamma: float = 1.0,
    epsilon: float = 0.01,
    transport_mode: str = "soft",
) -> Tuple[torch.Tensor, torch.Tensor]:
    middle_budget = middle_budget_from_ratio(
        key_states.shape[-2],
        compression_ratio=compression_ratio,
        sink_size=sink_size,
        recent_size=recent_size,
    )
    return otkv_segmented_compress_to_budget(
        key_states,
        value_states,
        middle_budget=middle_budget,
        sink_size=sink_size,
        recent_size=recent_size,
        gamma=gamma,
        epsilon=epsilon,
        transport_mode=transport_mode,
    )


class OTKVCache(BaseCompressCache):
    """
    Cache adapter for OTKV.

    The algorithm is cache-level: prefill writes full KV first, then compresses
    existing layer caches; decode compresses the existing cache before the new
    token is appended by DynamicCache.update.
    """

    def __init__(self, compression_size: Union[int, float] = DEFAULT_COMPRESSION_RATIO,
                 mode: str = "prefill",
                 sink_size: Union[int, float] = 4,
                 recent_size: Union[int, float] = 0.1,
                 gamma: float = 1.0, epsilon: float = 0.01,
                 transport_mode: str = "soft",
                 compress_interval: int = 1,
                 compression_ratio: Union[int, float, None] = None,
                 recent_window: Union[int, float, None] = None,
                 **kwargs):
        if compression_ratio is not None:
            compression_size = compression_ratio
        if recent_window is not None:
            recent_size = recent_window

        if isinstance(compression_size, float):
            if not 0 < compression_size <= 1:
                raise ValueError("compression_size ratio must be in (0, 1].")
        elif isinstance(compression_size, int):
            if compression_size <= 0:
                raise ValueError("compression_size token count must be positive.")
        else:
            raise TypeError("compression_size must be an int token count or float ratio.")
        if isinstance(sink_size, float):
            if not 0 <= sink_size <= 1:
                raise ValueError("sink_size ratio must be in [0, 1].")
        elif isinstance(sink_size, int):
            if sink_size < 0:
                raise ValueError("sink_size token count must be non-negative.")
        else:
            raise TypeError("sink_size must be an int token count or float ratio.")
        if isinstance(recent_size, float):
            if not 0 <= recent_size <= 1:
                raise ValueError("recent_size ratio must be in [0, 1].")
        elif isinstance(recent_size, int):
            if recent_size < 0:
                raise ValueError("recent_size token count must be non-negative.")
        else:
            raise TypeError("recent_size must be an int token count or float ratio.")
        if gamma < 0:
            raise ValueError("gamma must be non-negative.")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")
        if int(compress_interval) <= 0:
            raise ValueError("compress_interval must be a positive integer.")
        _validate_transport_mode(transport_mode)

        super().__init__(
            compression_size=compression_size,
            mode=mode,
            sink_size=sink_size,
            recent_size=recent_size,
            **kwargs,
        )
        self.compression_ratio = compression_size
        self.recent_window = recent_size
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.transport_mode = transport_mode
        self.compress_interval = int(compress_interval)
        self.decode_steps: Dict[int, int] = {}
        self.total_seen_tokens: Dict[int, int] = {}

    def on_prefill(self, key_states: torch.Tensor, value_states: torch.Tensor,
                   layer_idx: int, cache_kwargs: dict):
        self.current_attention_scores.pop(layer_idx, None)
        self.total_seen_tokens[layer_idx] = self.total_seen_tokens.get(layer_idx, 0) + key_states.shape[-2]
        return key_states, value_states

    def on_prefill_end(self):
        if getattr(self, "_prefill_finalized", False):
            return

        layer_indices = self._known_layer_indices()
        if not layer_indices:
            return

        prefill_total = max(self._total_seen_for_layer(layer_idx) for layer_idx in layer_indices)
        self._update_budget(prefill_total, is_prefill_end=True)
        self._clamp_budget_layout()

        for layer_idx in layer_indices:
            self.current_attention_scores.pop(layer_idx, None)
            self._compress_existing_layer(layer_idx, prefill_total)
        self.current_attention_scores.clear()
        self._prefill_finalized = True

    def on_decode_step(self, key_states: torch.Tensor, value_states: torch.Tensor,
                       layer_idx: int, cache_kwargs: dict):
        self.current_attention_scores.pop(layer_idx, None)
        prev_total = self._total_seen_for_layer(layer_idx)
        self._update_budget(prev_total, is_prefill_end=False)

        decode_step = self.decode_steps.get(layer_idx, 0)
        new_tokens = key_states.shape[-2]
        if decode_step % self.compress_interval == 0:
            self._compress_existing_layer(
                layer_idx,
                prev_total,
                reserve_tokens=max(new_tokens, self.compress_interval),
            )

        self.decode_steps[layer_idx] = decode_step + new_tokens
        self.total_seen_tokens[layer_idx] = prev_total + key_states.shape[-2]
        return key_states, value_states

    def _compress_existing_layer(self, layer_idx: int, total_tokens: int,
                                 reserve_tokens: int = 0):
        key_cache, value_cache = self._get_existing_cache(layer_idx)
        if key_cache is None or value_cache is None:
            return
        if key_cache.numel() == 0 or value_cache.numel() == 0:
            return

        reserve_tokens = max(0, int(reserve_tokens))
        middle_budget = min(
            key_cache.shape[-2],
            max(0, int(self.get_middle_budget(layer_idx, total_tokens)) - reserve_tokens),
        )
        new_k, new_v = otkv_segmented_compress_to_budget(
            key_cache,
            value_cache,
            middle_budget=middle_budget,
            sink_size=self.sink_size,
            recent_size=self.recent_size,
            gamma=self.gamma,
            epsilon=self.epsilon,
            transport_mode=self.transport_mode,
        )
        if new_k is not key_cache or new_v is not value_cache:
            self._replace_existing_cache(layer_idx, new_k, new_v)

    def _clamp_budget_layout(self):
        self.budget = max(0, int(self.budget))
        self.sink_size = min(max(0, int(self.sink_size)), self.budget)
        remaining = max(0, self.budget - self.sink_size)
        self.recent_size = min(max(0, int(self.recent_size)), remaining)
        self.middle_budget = max(0, self.budget - self.sink_size - self.recent_size)

    def _known_layer_indices(self):
        layer_indices = set(self.total_seen_tokens.keys())
        if hasattr(self, "layers"):
            layer_indices.update(range(len(self.layers)))
        elif hasattr(self, "key_cache"):
            layer_indices.update(range(len(self.key_cache)))
        return sorted(layer_indices)

    def _total_seen_for_layer(self, layer_idx: int) -> int:
        if layer_idx in self.total_seen_tokens:
            return self.total_seen_tokens[layer_idx]

        key_cache, _ = self._get_existing_cache(layer_idx)
        total_seen = 0 if key_cache is None else int(key_cache.shape[-2])
        self.total_seen_tokens[layer_idx] = total_seen
        return total_seen
