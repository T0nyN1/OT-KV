# evaluation/tasks/kv_recovery.py
from typing import Dict, Any

import scipy.stats
import torch

from .base_evaluator import BaseEvaluator
from .registry import register_task


@register_task("kv_recovery")
class KVRecoveryEvaluator(BaseEvaluator):
    """
    KV 状态恢复实验。
    通过对比 Baseline (全量Cache) 和 当前启用的 Policy (压缩Cache) 的特征分布，
    计算 1D Wasserstein 距离。
    """

    def evaluate(self) -> Dict[str, Any]:
        print("\n[*] Running KV State Recovery (Wasserstein Distance) Experiment...")
        tokenizer = self.model_wrapper.tokenizer
        model = self.model_wrapper._model

        # 截取一段较短的文本用于分析 (避免 OOM)
        sample_text = "Machine learning focuses on the development of computer programs that can access data and use it learn for themselves. The process of learning begins with observations or data."
        inputs = tokenizer(sample_text, return_tensors="pt").to(model.device)

        # 1. 运行当前压缩策略的模型 (Compressed)
        with torch.no_grad():
            outputs_compressed = model(inputs.input_ids, use_cache=True)
            cache_compressed = outputs_compressed.past_key_values

        # 2. 临时禁用策略，运行全量模型 (Dense)
        original_policy = self.model_wrapper.cache_manager.policy
        self.model_wrapper.cache_manager.policy = None  # 临时关闭压缩

        with torch.no_grad():
            outputs_dense = model(inputs.input_ids, use_cache=True)
            cache_dense = outputs_dense.past_key_values

        # 恢复压缩策略
        self.model_wrapper.cache_manager.policy = original_policy

        # 3. 计算最后一层的 V 矩阵特征分布的 Wasserstein 距离
        # 这里的实现采用扁平化后的 1D 近似 Wasserstein 距离
        layer_idx = -1  # 取最后一层
        def get_v_matrix(cache, idx):
            if hasattr(cache, "value_cache"):  # Transformers 4.36+ 引入的 DynamicCache
                return cache.value_cache[idx]
            elif hasattr(cache, "layers"):     # 某些自定义模型 Cache
                return cache.layers[idx].values
            elif isinstance(cache, (tuple, list)):  # 旧版的 Tuple 格式
                return cache[idx][1]
            else:
                raise TypeError(f"Unsupported cache type: {type(cache)}")

        # 使用兼容函数安全提取 V 矩阵
        v_dense = get_v_matrix(cache_dense, layer_idx).cpu().float().numpy().flatten()
        v_compressed = get_v_matrix(cache_compressed, layer_idx).cpu().float().numpy().flatten()

        # scipy.stats.wasserstein_distance 接受两个 1D 分布
        w_dist = scipy.stats.wasserstein_distance(v_dense, v_compressed)

        print(f"-> 最后一层 V Matrix 1D Wasserstein Distance: {w_dist:.4f}")

        return {"kv_recovery": {"wasserstein_distance_1d": w_dist.item()}}
