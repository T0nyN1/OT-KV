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
                # 传入自定义 cache 并开启 output_attentions 激活压缩逻辑
                outputs_compressed = model(
                    inputs.input_ids,
                    use_cache=True,
                    past_key_values=custom_cache,
                    output_attentions=True
                )
                cache_compressed = outputs_compressed.past_key_values
        finally:
            for h in getattr(self.model_wrapper, '_hooks', []):
                h.remove()
            self.model_wrapper._hooks.clear()

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

        # 因为新架构完全继承自 DynamicCache，提取变得非常简单统一
        def get_v_matrix(cache, idx):
            if hasattr(cache, "value_cache"):
                return cache.value_cache[idx]
            raise TypeError("Expected a subclass of DynamicCache")

        # 展平矩阵
        v_dense = get_v_matrix(cache_dense, layer_idx).cpu().float().numpy().flatten()
        v_compressed = get_v_matrix(cache_compressed, layer_idx).cpu().float().numpy().flatten()

        w_dist = scipy.stats.wasserstein_distance(v_dense, v_compressed)

        print(f"-> 最后一层 V Matrix 1D Wasserstein Distance: {w_dist:.4f}")

        return {"kv_recovery": {"wasserstein_distance_1d": w_dist.item()}}