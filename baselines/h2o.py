# baselines/h2o.py
import torch
from evaluation.models.base_cache import BaseCompressCache


class H2OCache(BaseCompressCache):
    """
    基于新 BaseCompressCache 框架重写的 Heavy-Hitter Oracle (H2O) KV cache。

    该策略通过累加历史 Attention 分数来评估 token 的重要性。
    依赖新框架统一管理的 Sink 和 Recent 边界，只对 Middle 区域进行驱逐。
    """

    def __init__(self, **kwargs):
        # 参数解析统一由基类处理 (compression_size, mode, sink_size, recent_size 等)
        super().__init__(**kwargs)

        # 记录每层 token 累计的 attention scores
        # 长度与当前保留在 Cache 中的 token 数量严格对齐
        self.hh_scores = {}

    def on_prefill(self, key_states: torch.Tensor, value_states: torch.Tensor,
                   layer_idx: int, cache_kwargs: dict):
        """
        Prefill 阶段：
        只为传入的 prompt tokens 初始化初始的 0 分数占位。
        暂不压缩，等待 on_prefill_end 获取首轮 attention 分数后再压。
        """
        device = key_states.device
        q_len = key_states.shape[-2]

        if layer_idx not in self.hh_scores:
            self.hh_scores[layer_idx] = torch.zeros(q_len, device=device, dtype=torch.float32)
        else:
            # 如果已有分数（例如多轮对话连续 prefill），则追加
            new_scores = torch.zeros(q_len, device=device, dtype=torch.float32)
            self.hh_scores[layer_idx] = torch.cat([self.hh_scores[layer_idx], new_scores])

        return key_states, value_states

    def on_prefill_end(self):
        """
        Prefill 阶段结束：
        1. 固化 Sink 和 Recent 边界大小
        2. 消耗 Prefill 产生的 Attention 分数
        3. 对所有层执行初始化的容量修剪
        """
        total_tokens = self.get_seq_length()
        self._update_budget(total_tokens, is_prefill_end=True)

        for layer_idx in list(self.hh_scores.keys()):
            self._consume_attention_scores(layer_idx)
            self._prune_layer(layer_idx, total_tokens)

    def on_decode_step(self, key_states: torch.Tensor, value_states: torch.Tensor,
                       layer_idx: int, cache_kwargs: dict):
        """
        Decode 阶段 (单步生成)：
        1. 消耗上一步自回归产生的 Attention 分数
        2. 根据需要对现有 Cache 进行驱逐 (挤出空间)
        3. 为当前正在生成的新 Token 创建分数占位
        """
        q_len = key_states.shape[-2]
        device = key_states.device

        # 1. 消耗上一步的 Attention 分数 (此时 hh_scores 与 KV Cache 长度还是对齐的)
        self._consume_attention_scores(layer_idx)

        # 2. 更新 Budget 容量 (如果 mode="entire" 会动态扩容)
        total_tokens = self.get_seq_length()
        total_after = total_tokens + q_len
        self._update_budget(total_after, is_prefill_end=False)

        # 3. 对现有缓存的 Middle 区域执行淘汰机制
        self._prune_layer(layer_idx, total_after)

        # 4. 给即将由父类 (DynamicCache) 追加到缓存的新 Token 在 hh_scores 占位
        new_scores = torch.zeros(q_len, device=device, dtype=torch.float32)
        self.hh_scores[layer_idx] = torch.cat([self.hh_scores[layer_idx], new_scores])

        # 5. 直接返回未经处理的新 Token，交给父类原样追加到 Cache 尾部
        return key_states, value_states

    def _prune_layer(self, layer_idx: int, total_tokens: int):
        """
        核心修剪逻辑：获取 Middle 分数 -> TopK -> 提交给基类修剪 -> 同步修剪自己的分数
        """
        middle_k, _ = self.get_middle_cache(layer_idx)
        if middle_k is None or middle_k.numel() == 0:
            return

        # 获取当前层 Middle 区域允许保留的 token 数
        layer_middle_budget = self.get_middle_budget(layer_idx, total_tokens)

        # 因为在 on_decode_step 时还未 append 新 token，
        # 此时 cache 的长度 seq_len 与 get_middle_cache 的划分规则保持绝对一致。
        seq_len = self._get_existing_cache(layer_idx)[0].shape[-2]
        middle_start = self.sink_size
        middle_end = max(self.sink_size, seq_len - self.recent_size)

        if middle_start >= middle_end:
            return

        # 提取对应 Middle 区域的 scores
        middle_scores = self.hh_scores[layer_idx][middle_start:middle_end]

        if layer_middle_budget >= middle_scores.shape[0]:
            return  # 空间充裕，无需修剪

        # 根据 H2O 的累计分数选出保留的 Top-K Indices
        if layer_middle_budget == 0:
            keep_indices = torch.empty(0, dtype=torch.long, device=middle_scores.device)
        else:
            _, keep_indices = torch.topk(middle_scores, k=layer_middle_budget)
            keep_indices = keep_indices.sort().values

        # 1. 提交给基类，由基类自动拼接 [Sink + Pruned_Middle + Recent] 更新 KV Cache
        self.prune_middle_cache(layer_idx, keep_indices)

        # 2. 严格同步修剪自己的 hh_scores 数组，保持与基类缓存的对齐
        sink_scores = self.hh_scores[layer_idx][:middle_start]
        recent_scores = self.hh_scores[layer_idx][middle_end:]
        pruned_middle_scores = middle_scores[keep_indices]

        self.hh_scores[layer_idx] = torch.cat([sink_scores, pruned_middle_scores, recent_scores])

    def _consume_attention_scores(self, layer_idx: int):
        """
        把基类 Hook 抓取到的当前步 attention_scores 累加到 self.hh_scores 中。
        """
        attn_weights = self.current_attention_scores.pop(layer_idx, None)
        if attn_weights is None or layer_idx not in self.hh_scores:
            return

        current_scores = self.hh_scores[layer_idx]
        if current_scores.numel() == 0:
            return

        score_update = self._attention_to_token_scores(attn_weights)
        score_update = score_update.to(device=current_scores.device, dtype=current_scores.dtype)

        # Usable 处理了可能存在的多余维度或越界情况
        usable = min(current_scores.numel(), score_update.numel())
        if usable == 0:
            return

        updated_scores = current_scores.clone()
        updated_scores[:usable] = updated_scores[:usable] + score_update[:usable]
        self.hh_scores[layer_idx] = updated_scores

    @staticmethod
    def _attention_to_token_scores(attn_weights: torch.Tensor) -> torch.Tensor:
        """
        将 Attention 矩阵降维为 1D 的 Token 级别分数。
        """
        scores = attn_weights.detach().float()
        if scores.dim() == 0:
            return scores.reshape(1)
        reduce_dims = tuple(range(scores.dim() - 1))
        if reduce_dims:
            scores = scores.sum(dim=reduce_dims)
        return scores.reshape(-1)