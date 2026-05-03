# baselines/h2o_cache.py
import torch
from evaluation.models.base_cache import BaseCompressCache


class H2OCache(BaseCompressCache):
    """
    H2O (Heavy Hitter Oracle) 缓存压缩策略的 Cache 原生实现。

    保留策略：
    1. Sink Tokens: 最早的几个起始 Token (维持注意力分布的稳定性)
    2. Local/Recent Window: 最近生成的若干个 Token
    3. Heavy Hitters: 累积注意力分数最高的核心 Token
    """

    def __init__(self, compression_ratio, recent_window=256, sink_size=4, **kwargs):
        super().__init__(compression_ratio, **kwargs)
        self.recent_window = recent_window
        self.sink_size = sink_size

        # 核心状态：存储每一层每个 Token 累积被关注的分数
        # 数据结构: Dict[layer_idx -> Tensor of shape (batch_size, num_heads, seq_len)]
        self.layer_scores = {}

    def process_prefill(self, key_states: torch.Tensor, value_states: torch.Tensor,
                        layer_idx: int, cache_kwargs: dict):
        """
        Prefill 阶段 (Context 编码)：
        初始化分数，不对 KV Cache 进行截断（通常 Prefill 阶段全量保留或由外部统一截断）。
        """
        # 从暂存区获取 Hook 抓取到的 Prefill Attention Scores
        # 形状通常为: [batch, heads, q_len, kv_len]
        attn_scores = self.current_attention_scores.pop(layer_idx, None)

        if attn_scores is not None:
            # 将每个 Token 接收到的所有注意力求和，作为初始的 Heavy Hitter 分数
            # [batch, heads, q_len, kv_len] -> [batch, heads, kv_len]
            self.layer_scores[layer_idx] = attn_scores.float().sum(dim=-2)
        else:
            # 如果没抓到 (比如底层没开 output_attentions)，兜底初始化为 0
            batch, heads, _, head_dim = key_states.shape
            kv_len = key_states.shape[-2]
            self.layer_scores[layer_idx] = torch.zeros(
                (batch, heads, kv_len),
                device=key_states.device,
                dtype=torch.float32
            )

        return key_states, value_states

    def process_decode_step(self, key_states: torch.Tensor, value_states: torch.Tensor,
                            layer_idx: int, cache_kwargs: dict):
        """
        Decode 阶段 (单步生成)：
        更新计分板 -> 判定是否超预算 -> 从历史缓存中驱逐低分 Token -> 返回当前 Token。
        """
        # 当前层历史缓存的长度
        hist_len = self.get_seq_length(layer_idx)
        if hist_len == 0:
            return key_states, value_states

        # 1. 更新计分板
        attn_scores = self.current_attention_scores.pop(layer_idx, None)
        if attn_scores is not None:
            # Decode 阶段 q_len = 1，attn_scores 形状: [batch, heads, 1, hist_len + 1]
            current_step_scores = attn_scores[:, :, 0, :].float()

            # 分离出对历史 Token 的关注度和对当前新 Token (Self) 的关注度
            score_to_past = current_step_scores[:, :, :hist_len]
            score_to_self = current_step_scores[:, :, hist_len:]

            # 累加历史分数，并拼上新 Token 的分数
            self.layer_scores[layer_idx] += score_to_past
            self.layer_scores[layer_idx] = torch.cat([self.layer_scores[layer_idx], score_to_self], dim=-1)

        # 2. 计算预算
        total_len = hist_len + 1  # 历史长度 + 当前新生成的 1 个 Token
        budget = max(
            int(total_len * self.compression_ratio),
            self.sink_size + self.recent_window + 1
        )

        # 如果没有超预算，直接返回当前 token，底层会自动拼接到 Cache 中
        if total_len <= budget:
            return key_states, value_states

        # 3. 触发驱逐 (Eviction)
        # 因为即将拼接 1 个新 Token，所以历史缓存 (hist_len) 只能保留 (budget - 1) 个
        historical_budget = budget - 1

        # 近期窗口需要给当前新 Token 留 1 个位置
        recent_keep = self.recent_window - 1
        heavy_budget = historical_budget - self.sink_size - recent_keep

        # 获取历史 Token 的分数 [batch, heads, hist_len]
        # 注意：这里我们只截取历史长度的分数进行排序，新 token 还没进缓存
        scores = self.layer_scores[layer_idx][:, :, :hist_len]
        batch, heads = scores.shape[0], scores.shape[1]
        device = scores.device

        # 生成保留的索引
        sink_indices = torch.arange(self.sink_size, device=device).view(1, 1, -1).expand(batch, heads, -1)
        recent_indices = torch.arange(hist_len - recent_keep, hist_len, device=device).view(1, 1, -1).expand(batch,
                                                                                                             heads, -1)

        if heavy_budget > 0:
            # 截取中间部分计算 Top-K
            middle_scores = scores[:, :, self.sink_size: hist_len - recent_keep]
            _, topk_indices = torch.topk(middle_scores, heavy_budget, dim=-1)
            # 将 Top-K 的相对索引映射回全局索引
            topk_indices = topk_indices + self.sink_size

            keep_indices = torch.cat([sink_indices, topk_indices, recent_indices], dim=-1)
        else:
            keep_indices = torch.cat([sink_indices, recent_indices], dim=-1)

        # 对保留索引进行排序，保证序列时间位置不乱
        keep_indices = torch.sort(keep_indices, dim=-1).values

        # 4. 执行历史缓存驱逐
        def gather_kv(tensor, indices):
            # tensor: [B, H, L, D], indices: [B, H, Keep_L]
            head_dim = tensor.shape[-1]
            idx = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            return torch.gather(tensor, dim=2, index=idx)

        # 更新底层 Cache
        self.key_cache[layer_idx] = gather_kv(self.key_cache[layer_idx], keep_indices)
        self.value_cache[layer_idx] = gather_kv(self.value_cache[layer_idx], keep_indices)

        # 同步更新计分板 (只保留未被驱逐的分数 + 最新的那个 Token 的分数)
        survived_scores = torch.gather(scores, dim=2, index=keep_indices)
        latest_token_score = self.layer_scores[layer_idx][:, :, -1:]  # 刚刚拼上去的当前 token 分数
        self.layer_scores[layer_idx] = torch.cat([survived_scores, latest_token_score], dim=-1)

        # 5. 返回当前新 Token 的 KV
        # BaseCompressCache 的 update 逻辑拿到返回值后，会把它追加到刚被你压缩好的 self.key_cache 末尾
        return key_states, value_states