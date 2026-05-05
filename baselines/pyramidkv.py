# baselines/pyramidkv.py
import math
from baselines.snapkv import SnapKVCache


class PyramidKVCache(SnapKVCache):
    """
    基于 SnapKV 继承实现的 PyramidKV。
    仅仅重写了 get_middle_budget 方法，实现越深的层截断越狠的金字塔结构。
    """

    def __init__(self, pyramid_low_scale: float = 1.5, pyramid_high_scale: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.pyramid_low_scale = float(pyramid_low_scale)
        self.pyramid_high_scale = float(pyramid_high_scale)
        self.num_layers = None

    def get_middle_budget(self, layer_idx: int, total_tokens: int) -> int:
        # 探测当前模型的总层数
        self.num_layers = max(self.num_layers or 0, layer_idx + 1)

        num_layers = max(1, self.num_layers)
        if num_layers == 1:
            scale = 1.0
        else:
            depth = layer_idx / float(num_layers - 1)
            scale = self.pyramid_low_scale + depth * (self.pyramid_high_scale - self.pyramid_low_scale)

        mean_scale = 0.5 * (self.pyramid_low_scale + self.pyramid_high_scale)
        if mean_scale <= 0:
            mean_scale = 1.0

        # 同时支持整数（绝对预算）和浮点（比例）的 compression_size，
        # 与 BaseCompressCache._update_budget 中的语义保持一致。
        if isinstance(self.compression_size, int):
            base_layer_budget = float(self.compression_size)
        else:
            base_layer_budget = float(total_tokens) * float(self.compression_size)

        layer_budget = math.floor(base_layer_budget * scale / mean_scale)
        # 不允许超过物理 token 数量
        layer_budget = min(layer_budget, total_tokens)

        return max(0, layer_budget - self.sink_size - self.recent_size)