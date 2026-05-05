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
            effective_ratio = self.compression_size
        else:
            depth = layer_idx / float(num_layers - 1)
            scale = self.pyramid_low_scale + depth * (self.pyramid_high_scale - self.pyramid_low_scale)
            mean_scale = 0.5 * (self.pyramid_low_scale + self.pyramid_high_scale)
            # 根据基准的 compression_size 调节当前层比例
            effective_ratio = min(1.0, max(0.0, self.compression_size * scale / mean_scale))

        # 计算层绝对 budget，并扣除固定的 sink 和 recent 得到 middle 的配额
        layer_budget = math.floor(total_tokens * effective_ratio)
        return max(0, layer_budget - self.sink_size - self.recent_size)