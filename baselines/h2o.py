# baselines/h2o.py
import torch
from evaluation.models.base_policy import BaseKVPolicy


class H2OPolicy(BaseKVPolicy):
    """
    H2O (Heavy Hitter Oracle) 缓存压缩策略。
    算法原理：保留注意力分数最高的 Token (Heavy Hitters) + 最近的 Token (Recent) + 起始 Token (Sink)。
    """

    def __init__(self, compression_ratio, recent_window=256, sink_size=4, **kwargs):
        super().__init__(compression_ratio, **kwargs)
        self.recent_window = recent_window
        self.sink_size = sink_size

        # 核心状态：存储每一层每个 Token 的累积注意力分数
        # key: layer_idx, value: torch.Tensor [batch, num_heads, seq_len]
        self.layer_scores = {}

    def _update_scores(self, layer_idx, attention_scores):
        """
        更新注意力分数。attention_scores 维度通常为 [batch, num_heads, q_len, kv_len]。
        在 Decode 阶段 q_len = 1。
        """
        if attention_scores is None:
            return

        # 1. 提取当前步的注意力权重 (取最后一行 query 对所有 key 的关注度)
        # [batch, num_heads, 1, kv_len] -> [batch, num_heads, kv_len]
        current_step_scores = attention_scores[:, :, -1, :].float()

        if layer_idx not in self.layer_scores:
            self.layer_scores[layer_idx] = current_step_scores
        else:
            # 2. 对齐长度：旧分数长度为 L，新分数长度为 L+1 (因为加入了当前 token)
            old_scores = self.layer_scores[layer_idx]
            batch, heads, old_len = old_scores.shape

            # 将旧分数补 0 后与新分数相加
            padding = torch.zeros((batch, heads, 1), device=old_scores.device, dtype=old_scores.dtype)
            updated_scores = torch.cat([old_scores, padding], dim=-1) + current_step_scores
            self.layer_scores[layer_idx] = updated_scores

    def process_prefill(self, past_key_values, attention_scores=None, **kwargs):
        """
        Prefill 阶段结束后，初始化分数。
        """
        # 如果模型输出了 Prefill 阶段的完整 Attention (O(N^2))，我们取均值作为初始分
        if attention_scores is not None:
            layer_idx = kwargs.get('layer_idx', 0)
            # 对 query 维度求和 [batch, heads, q_len, kv_len] -> [batch, heads, kv_len]
            self.layer_scores[layer_idx] = attention_scores.float().sum(dim=-2)

        return past_key_values

    def process_decode_step(self, past_key_values, layer_idx, attention_scores=None, **kwargs):
        """
        Decode 步进：更新分数 -> 判定预算 -> 执行驱逐
        """
        # 1. 更新计分板
        self._update_scores(layer_idx, attention_scores)

        # 获取当前 Cache 状态
        # 兼容 DynamicCache 或 Tuple 格式
        if hasattr(past_key_values, "key_cache"):
            k_cache = past_key_values.key_cache[layer_idx]
            v_cache = past_key_values.value_cache[layer_idx]
        else:
            k_cache, v_cache = past_key_values[0], past_key_values[1]

        current_len = k_cache.shape[-2]

        # 2. 判定是否需要驱逐
        # 计算总预算：基于压缩率，且不能小于 sink + recent
        budget = max(int(current_len * self.compression_ratio), self.sink_size + self.recent_window + 1)

        if current_len <= budget:
            return past_key_values

        # 3. 执行 H2O 驱逐逻辑
        # 划分区间：[0, sink_size] | [sink_size, recent_start] | [recent_start, current_len]
        recent_start = current_len - self.recent_window
        heavy_budget = budget - self.recent_window - self.sink_size

        scores = self.layer_scores[layer_idx]

        # 获取中间区域 (剔除 sink 和 recent) 的分数进行排序
        middle_scores = scores[:, :, self.sink_size:recent_start]

        # 挑选 Heavy Hitters
        _, topk_indices = torch.topk(middle_scores, heavy_budget, dim=-1)
        # 映射回全局索引
        topk_indices = topk_indices + self.sink_size

        # 组装最终保留的索引
        batch, heads = scores.shape[0], scores.shape[1]

        # Sink 索引
        sink_indices = torch.arange(self.sink_size, device=scores.device).view(1, 1, -1).expand(batch, heads, -1)
        # Recent 索引
        recent_indices = torch.arange(recent_start, current_len, device=scores.device).view(1, 1, -1).expand(batch,
                                                                                                             heads, -1)

        # 合并所有保留索引并排序
        keep_indices = torch.cat([sink_indices, topk_indices, recent_indices], dim=-1)
        keep_indices = torch.sort(keep_indices, dim=-1).values  # [batch, heads, budget]

        # 4. 更新 KV Cache 和 分数缓存
        def gather_kv(tensor, indices):
            # tensor: [B, H, L, D], indices: [B, H, Budget]
            head_dim = tensor.shape[-1]
            idx = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            return torch.gather(tensor, dim=2, index=idx)

        new_k = gather_kv(k_cache, keep_indices)
        new_v = gather_kv(v_cache, keep_indices)

        # 同步更新分数缓存
        self.layer_scores[layer_idx] = torch.gather(scores, dim=2, index=keep_indices)

        # 5. 写回并返回 (根据类型)
        if hasattr(past_key_values, "key_cache"):
            past_key_values.key_cache[layer_idx] = new_k
            past_key_values.value_cache[layer_idx] = new_v
            # 关键：手动同步 DynamicCache 的 seen_tokens 状态，防止偏移错误
            if layer_idx == 0:
                past_key_values._seen_tokens = new_k.shape[-2]
            return past_key_values
        else:
            return (new_k, new_v)