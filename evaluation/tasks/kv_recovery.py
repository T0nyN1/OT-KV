# evaluation/tasks/kv_recovery.py
from typing import Dict, Any

import scipy.stats
import torch
from transformers.cache_utils import DynamicCache

from .base_evaluator import BaseEvaluator
from .registry import register_task


@register_task("kv_recovery")
class KVRecoveryEvaluator(BaseEvaluator):
    """
    KV 状态恢复实验。
    计算 Baseline (原生全量 Cache) 和 启用的 Cache 策略 之间的 1D Wasserstein 距离。
    """

    def evaluate(self) -> Dict[str, Any]:
        print("\n[*] Running KV State Recovery (Wasserstein Distance) Experiment...")
        tokenizer = self.model_wrapper.tokenizer
        model = self.model_wrapper._model

        # 截取一段较短的文本用于分析
        sample_text = "Machine learning focuses on the development of computer programs that can access data and use it learn for themselves. The process of learning begins with observations or data."
        inputs = tokenizer(sample_text, return_tensors="pt").to(model.device)

        # ==================================================
        # 1. 运行当前压缩策略的模型 (Compressed)
        # ==================================================
        custom_cache = self.model_wrapper._setup_cache_and_hooks()
        try:
            with torch.no_grad():
                outputs_compressed = model(
                    inputs.input_ids,
                    use_cache=True,
                    past_key_values=custom_cache,
                )
                cache_compressed = outputs_compressed.past_key_values
                if hasattr(cache_compressed, "on_prefill_end"):
                    cache_compressed.on_prefill_end()
        finally:
            self._cleanup_cache_and_hooks(custom_cache)

        # ==================================================
        # 2. 运行基线原生模型 (Dense)
        # ==================================================
        dense_cache = DynamicCache()  # HuggingFace 原生全量 Cache
        with torch.no_grad():
            outputs_dense = model(
                inputs.input_ids,
                use_cache=True,
                past_key_values=dense_cache
            )
            cache_dense = outputs_dense.past_key_values

        # ==================================================
        # 3. 提取特征计算 Wasserstein 距离
        # ==================================================
        layer_idx = -1  # 取最后一层

        def get_v_matrix(cache, idx):
            # Transformers 4.x: DynamicCache has value_cache as list of tensors
            if hasattr(cache, "value_cache"):
                return cache.value_cache[idx]
            # Transformers 5.x: DynamicCache has layers list with .values per layer
            if hasattr(cache, "layers"):
                return cache.layers[idx].values
            # Legacy tuple-of-tuples format: ((k0, v0), (k1, v1), ...)
            if isinstance(cache, (tuple, list)):
                return cache[idx][1]
            raise TypeError(f"Unsupported cache type: {type(cache)}")

        # 展平矩阵
        v_dense = get_v_matrix(cache_dense, layer_idx).cpu().float().numpy().flatten()
        v_compressed = get_v_matrix(cache_compressed, layer_idx).cpu().float().numpy().flatten()

        w_dist = scipy.stats.wasserstein_distance(v_dense, v_compressed)

        print(f"-> 最后一层 V Matrix 1D Wasserstein Distance: {w_dist:.4f}")

        return {"kv_recovery": {"wasserstein_distance_1d": w_dist.item()}}
