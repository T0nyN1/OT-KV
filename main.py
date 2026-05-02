# main.py
import argparse

from baselines.baseline import BaselinePolicy
from evaluation.models.wrapper import EvaluatorHFLM
from evaluation.tasks.registry import get_evaluator
from utils import set_device


def main(model_id, method, task, prefill_fraction, max_length, **kwargs):
    # 1. 策略选择逻辑
    policy = None
    if method == "baseline":
        # 也可以使用 policy = BaselinePolicy()，效果等同
        policy = BaselinePolicy()
    elif method == "otkv":
        pass
    elif method == "h2o":
        from baselines.h2o import H2OPolicy
        policy = H2OPolicy(
            compression_ratio=kwargs.get('compression_ratio', 0.5),
            recent_window=256,
            sink_size=4
        )
    elif method == "streamingllm":
        pass
    elif method == "snapkv":
        pass
    elif method == "pyramidkv":
        pass
    elif method == "echokv":
        pass
    else:
        raise ValueError(f"Unknown method: {method}")

    # 2. 初始化封装后的模型
    print(f">>> Loading model: {model_id} with {method} policy...")
    model_wrapper = EvaluatorHFLM(
        pretrained=model_id,
        policy=policy,
        prefill_fraction=prefill_fraction,
        max_length=max_length,
        device=set_device()
    )

    # 3. 动态获取并执行评测任务
    print(f">>> Running task: {task}...")
    evaluator_class = get_evaluator(task)
    evaluator_instance = evaluator_class(model_wrapper=model_wrapper, **kwargs)

    result = evaluator_instance.evaluate()

    print("\n" + "=" * 30)
    print("EVALUATION RESULT")
    print("=" * 30)
    print(result)


def run():
    parser = argparse.ArgumentParser(description="OT-KV Evaluation Framework")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--task", type=str, default="wikitext",
                        choices=["wikitext", "niah", "longbench", "profile_niah", "kv_recovery", "ruler"])
    parser.add_argument("--method", type=str, default="baseline",
                        choices=["baseline", "otkv", "h2o", "streamingllm", "snapkv", "pyramidkv", "echokv"])
    parser.add_argument("--prefill_fraction", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=None, help="samples limit")

    args = parser.parse_args()
    main(model_id=args.model_id,
         method=args.method,
         task=args.task,
         prefill_fraction=args.prefill_fraction,
         max_length=args.max_length,
         limit=args.limit)


if __name__ == "__main__":
    run()
