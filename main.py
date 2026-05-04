# main.py
import argparse

from evaluation.models.wrapper import EvaluatorHFLM
from evaluation.tasks.registry import get_evaluator
from utils import set_device


def main(model_id, method, task, prefill_fraction, max_length, **kwargs):
    """
    新架构下的主入口。
    不再实例化 Policy 对象，而是配置 cache_class 和 cache_kwargs。
    """

    # 1. 策略配置映射
    # 根据 method 选择对应的 Cache 子类及其特有参数
    cache_class = None
    cache_kwargs = {}

    if method == "baseline":
        from baselines.baseline import BaselineCache
        cache_class = BaselineCache
        cache_kwargs = {}

    elif method == "h2o":
        from baselines.h2o import H2OCache
        cache_class = H2OCache
        cache_kwargs = {
            "compression_ratio": kwargs.get('compression_ratio', 0.5),
            "recent_window": kwargs.get('recent_window', 256),
            "sink_size": kwargs.get('sink_size', 4),
        }

    elif method == "otkv":
        # 预留给你的 OT-KV 实现
        # from baselines.otkv_cache import OTKVCache
        # cache_class = OTKVCache
        # cache_kwargs = {"compression_ratio": kwargs.get('compression_ratio', 0.5)}
        pass

    elif method == "streamingllm":
        from baselines.streamingllm import StreamingLLMCache
        cache_class = StreamingLLMCache
        cache_kwargs = {
            "compression_ratio": kwargs.get('compression_ratio', 1.0),
            "recent_window": kwargs.get('recent_window', 256),
            "sink_size": 4 if kwargs.get('sink_size') is None else kwargs.get('sink_size'),
        }

    elif method == "snapkv":
        from baselines.snapkv import SnapKVCache
        cache_class = SnapKVCache
        cache_kwargs = {
            "compression_ratio": kwargs.get('compression_ratio', 0.5),
            "recent_window": kwargs.get('recent_window', 256),
            "sink_size": 0 if kwargs.get('sink_size') is None else kwargs.get('sink_size'),
            "observation_window": kwargs.get('observation_window', None),
        }

    elif method == "pyramidkv":
        from baselines.pyramidkv import PyramidKVCache
        cache_class = PyramidKVCache
        cache_kwargs = {
            "compression_ratio": kwargs.get('compression_ratio', 0.5),
            "recent_window": kwargs.get('recent_window', 256),
            "sink_size": 0 if kwargs.get('sink_size') is None else kwargs.get('sink_size'),
            "observation_window": kwargs.get('observation_window', None),
        }

    elif method == "echokv":
        from baselines.echokv import EchoKVCache
        cache_class = EchoKVCache
        cache_kwargs = {
            "compression_ratio": kwargs.get('compression_ratio', 0.5),
            "recent_window": kwargs.get('recent_window', 256),
            "sink_size": 0 if kwargs.get('sink_size') is None else kwargs.get('sink_size'),
        }

    else:
        raise ValueError(f"Unknown method: {method}")

    # 2. 初始化封装后的模型 (EvaluatorHFLM)
    # 我们将类本身和参数传进去，让模型在推理循环中动态创建 Cache 实例
    print(f">>> [Init] Loading model: {model_id}")
    print(f">>> [Init] Optimization Method: {method}")

    model_wrapper = EvaluatorHFLM(
        pretrained=model_id,
        cache_class=cache_class,
        cache_kwargs=cache_kwargs,
        prefill_fraction=prefill_fraction,
        max_length=max_length,
        device=set_device(),
        # 也可以在此处传递其他 HF 模型参数，如 torch_dtype
        # torch_dtype=torch.float16
    )

    # 3. 获取评测任务并执行
    print(f">>> [Task] Running evaluation task: {task}...")
    try:
        evaluator_class = get_evaluator(task)
        evaluator_instance = evaluator_class(model_wrapper=model_wrapper, **kwargs)

        # 执行评测
        result = evaluator_instance.evaluate()

        # 4. 打印结果
        print("\n" + "=" * 40)
        print(f"EVALUATION RESULT (Method: {method}, Task: {task})")
        print("=" * 40)
        print(result)
        print("=" * 40 + "\n")

    except Exception as e:
        print(f"[Critical Error] Evaluation failed: {str(e)}")
        raise e


def run():
    parser = argparse.ArgumentParser(description="OT-KV & KV Compression Evaluation Framework (v2: Cache-based)")

    # 基础模型与任务配置
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        help="HuggingFace model repository ID or local path")
    parser.add_argument("--task", type=str, default="wikitext",
                        choices=["wikitext", "niah", "longbench", "ruler"],
                        help="Evaluation task name registered in TASK_REGISTRY")
    parser.add_argument("--method", type=str, default="baseline",
                        choices=["baseline", "otkv", "h2o", "streamingllm", "snapkv", "pyramidkv", "echokv"],
                        help="KV Cache compression method")

    # 压缩相关参数
    parser.add_argument("--compression_ratio", type=float, default=0.5,
                        help="Target KV Cache retention ratio (e.g., 0.5 means keep 50%)")
    parser.add_argument("--recent_window", type=int, default=256,
                        help="Size of the local/recent window for algorithms like H2O or StreamingLLM")
    parser.add_argument("--sink_size", type=int, default=4,
                        help="Number of initial/sink tokens to retain")
    parser.add_argument("--observation_window", type=int, default=None,
                        help="Query window used to score prompt tokens for SnapKV/PyramidKV")

    # 评测流程控制
    parser.add_argument("--prefill_fraction", type=float, default=0.1,
                        help="Fraction of document used for the initial prefill stage in PPL testing")
    parser.add_argument("--max_length", type=int, default=4096,
                        help="Maximum sequence length for the model")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of samples for evaluation (for quick debugging)")

    args = parser.parse_args()

    # 将 args 转换为字典以便透传给具体任务
    main_kwargs = vars(args)

    main(**main_kwargs)


if __name__ == "__main__":
    run()
