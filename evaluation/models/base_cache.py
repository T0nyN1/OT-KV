# evaluation/models/base_cache.py
from typing import Union

import torch
from transformers.cache_utils import DynamicCache


class BaseCompressCache(DynamicCache):
    """
    重设计后的 KV Cache 压缩基类。
    统一管理 Sink、Middle、Recent 区域的划分，并维护 Budget 动态计算逻辑。
    支持整数（绝对数量）或小数（比例）来配置压缩率、Sink 大小和 Recent 大小。
    """

    def __init__(self,
                 compression_size: Union[int, float],
                 mode: str = "prefill",
                 sink_size: Union[int, float] = 4,  # 支持输入固定 token 数量或比例
                 recent_size: Union[int, float] = 256,  # 支持输入固定 token 数量或比例
                 **kwargs):
        super().__init__()

        self.compression_size = compression_size
        self.mode = mode.lower()  # "prefill" 或 "entire"

        # 记录初始设定的值 (在 prefill 结束后会解析为固定整数)
        self._init_sink_size = sink_size
        self._init_recent_size = recent_size
        self.kwargs = kwargs

        # 暂存当前 step 各层的 attention scores
        self.current_attention_scores = {}

        # 核心缓存容量控制属性 (运行时固定为绝对长度整数)
        self.budget = 0
        self.sink_size = 0
        self.recent_size = 0
        self.middle_budget = 0
        self.prefill_length = 0

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, cache_kwargs=None) -> tuple[
        torch.Tensor, torch.Tensor]:
        q_len = key_states.shape[-2]
        is_prefill = q_len > 1 and layer_idx == 0 and self.get_seq_length() == 0

        # 1. 阶段分发与自定义处理
        if is_prefill or q_len > 1:
            key_states, value_states = self.on_prefill(key_states, value_states, layer_idx, cache_kwargs)
        else:
            key_states, value_states = self.on_decode_step(key_states, value_states, layer_idx, cache_kwargs)

        # 2. 调用父类方法真正写入 Cache 状态
        result = super().update(key_states, value_states, layer_idx, cache_kwargs)
        return result

    def on_prefill(self, key_states, value_states, layer_idx, cache_kwargs):
        raise NotImplementedError

    def on_decode_step(self, key_states, value_states, layer_idx, cache_kwargs):
        raise NotImplementedError

    def on_prefill_end(self):
        raise NotImplementedError

    # ==========================================
    # Budget 管理模块
    # ==========================================

    def _update_budget(self, total_tokens: int, is_prefill_end: bool = False):
        """
        更新容量上限与区域划分。
        - is_prefill_end=True: 设定初始 Budget，解析并固化 sink_size 和 recent_size。
        - is_prefill_end=False: 依据 mode 和 compression_size 决定 middle 是否扩容。
        """
        if is_prefill_end:
            self.prefill_length = total_tokens

            # 1. 计算总 Budget
            if isinstance(self.compression_size, int):
                self.budget = self.compression_size - 1
            elif isinstance(self.compression_size, float):
                self.budget = int(total_tokens * self.compression_size) - 1

            # 2. 解析并固化 Sink 大小
            if isinstance(self._init_sink_size, int):
                self.sink_size = self._init_sink_size
            elif isinstance(self._init_sink_size, float):
                self.sink_size = int(self.budget * self._init_sink_size)

            # 3. 解析并固化 Recent 大小
            if isinstance(self._init_recent_size, int):
                self.recent_size = self._init_recent_size
            elif isinstance(self._init_recent_size, float):
                self.recent_size = int(self.budget * self._init_recent_size)

            # 4. 计算 Middle 剩余可用预算
            self.middle_budget = max(0, self.budget - self.sink_size - self.recent_size)

            print(f"[Cache Monitor] Total Tokens  : {total_tokens}")
            print(f"[Cache Monitor] Budget        : {self.budget}")
            print(f"[Cache Monitor] Sink Size     : {self.sink_size}")
            print(f"[Cache Monitor] Recent Size   : {self.recent_size}")
            print(f"[Cache Monitor] Middle Size   : {self.middle_budget}")

        else:
            # 仅在 entire 模式且使用比例压缩时，才随 token 增加动态扩容 Budget
            if self.mode == "entire" and isinstance(self.compression_size, float):
                self.budget = int(total_tokens * self.compression_size) - 1
                # Sink 和 Recent 长度不变，只允许 Middle 动态扩容
                self.middle_budget = max(0, self.budget - self.sink_size - self.recent_size)

    def get_middle_budget(self, layer_idx: int, total_tokens: int) -> int:
        """
        获取当前层 Middle 区域的允许保留大小。
        默认实现：返回计算好的全局 self.middle_budget。
        子类 (如 PyramidKV) 可以重写此方法以实现跨层不同预算。
        """
        if self.mode == "entire" and isinstance(self.compression_size, float):
            # 动态扩容模式下实时计算当前总 budget
            budget = int(total_tokens * self.compression_size)
            return max(0, budget - self.sink_size - self.recent_size)

        return self.middle_budget

    # ==========================================
    # Middle 区域操作接口 (供子类使用)
    # ==========================================

    def get_middle_cache(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        获取当前层的 Middle 区域 KV Cache 副本，供子类进行特征分析或提取。
        返回 (middle_k, middle_v)，若无 Middle 区域则返回 (None, None)。
        """
        k, v = self._get_existing_cache(layer_idx)
        if k is None or v is None:
            return None, None

        seq_len = k.shape[-2]
        middle_start = self.sink_size
        middle_end = max(self.sink_size, seq_len - self.recent_size)

        if middle_start >= middle_end:
            return None, None

        return k[..., middle_start:middle_end, :], v[..., middle_start:middle_end, :]

    def prune_middle_cache(self, layer_idx: int, keep_middle_indices: torch.Tensor):
        """
        对 Middle 区域进行修剪。
        子类传入针对 Middle 区域的局部保留索引 (范围 0 到 middle_len-1)，
        基类将自动提取保留的部分，并与原有的 Sink 和 Recent 重新拼接。
        """
        k, v = self._get_existing_cache(layer_idx)
        if k is None or v is None:
            return

        seq_len = k.shape[-2]
        middle_start = self.sink_size
        middle_end = max(self.sink_size, seq_len - self.recent_size)

        # 提取 Sink 和 Recent 区域
        sink_k, sink_v = k[..., :middle_start, :], v[..., :middle_start, :]
        recent_k, recent_v = k[..., middle_end:, :], v[..., middle_end:, :]

        # 提取并修剪 Middle 区域
        if middle_start < middle_end:
            middle_k, middle_v = k[..., middle_start:middle_end, :], v[..., middle_start:middle_end, :]
            keep_middle_indices = keep_middle_indices.to(device=k.device, dtype=torch.long)
            pruned_middle_k = middle_k.index_select(-2, keep_middle_indices)
            pruned_middle_v = middle_v.index_select(-2, keep_middle_indices)
        else:
            # 异常兜底：若 Middle 为空，则创建空张量
            pruned_middle_k = k.new_empty((*k.shape[:-2], 0, k.shape[-1]))
            pruned_middle_v = v.new_empty((*v.shape[:-2], 0, v.shape[-1]))

        # 重新拼接
        new_k = torch.cat([sink_k, pruned_middle_k, recent_k], dim=-2)
        new_v = torch.cat([sink_v, pruned_middle_v, recent_v], dim=-2)

        self._replace_existing_cache(layer_idx, new_k, new_v)

    def replace_middle_cache(self, layer_idx: int, new_middle_k: torch.Tensor, new_middle_v: torch.Tensor):
        """
        如果你需要对 Middle 做复杂的非线性压缩（如 Token 合并/池化），
        可以直接通过此接口用全新的 Middle KV 替换掉原有的 Middle 区域。
        """
        k, v = self._get_existing_cache(layer_idx)
        if k is None or v is None:
            return

        seq_len = k.shape[-2]
        middle_start = self.sink_size
        middle_end = max(self.sink_size, seq_len - self.recent_size)

        sink_k, sink_v = k[..., :middle_start, :], v[..., :middle_start, :]
        recent_k, recent_v = k[..., middle_end:, :], v[..., middle_end:, :]

        new_k = torch.cat([sink_k, new_middle_k, recent_k], dim=-2)
        new_v = torch.cat([sink_v, new_middle_v, recent_v], dim=-2)

        self._replace_existing_cache(layer_idx, new_k, new_v)

    # ==========================================
    # 底层 HuggingFace 兼容层
    # ==========================================

    def _get_existing_cache(self, layer_idx: int):
        # 适配 Transformers 5.x
        if hasattr(self, "layers"):
            if layer_idx >= len(self.layers):
                return None, None
            layer = self.layers[layer_idx]
            return layer.keys, layer.values

        # 适配 Transformers 4.x (兼容旧版逻辑)
        if hasattr(self, "key_cache"):
            if layer_idx >= len(self.key_cache):
                return None, None
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        return None, None

    def _replace_existing_cache(self, layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor):
        if key_states is None or value_states is None:
            return

        # 适配 Transformers 5.x
        if hasattr(self, "layers"):
            if layer_idx >= len(self.layers):
                return
            layer = self.layers[layer_idx]
            layer.keys = key_states
            layer.values = value_states
            self._set_layer_length(layer, key_states.shape[-2])
            return

        # 适配 Transformers 4.x (兼容旧版逻辑)
        if hasattr(self, "key_cache"):
            if layer_idx >= len(self.key_cache):
                return
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states

    @staticmethod
    def _set_layer_length(layer, seq_len: int):
        if hasattr(layer, "cumulative_length"):
            if isinstance(layer.cumulative_length, torch.Tensor):
                layer.cumulative_length.fill_(seq_len)
            else:
                layer.cumulative_length = seq_len
        if hasattr(layer, "cumulative_length_int"):
            layer.cumulative_length_int = seq_len
