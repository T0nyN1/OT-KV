import argparse
import time

from evaluation.models.wrapper import EvaluatorHFLM
from evaluation.tasks.registry import get_evaluator
from utils import set_device, export_results


def get_cache_config(method: str, kwargs: dict):
    match method.lower():
        case "baseline":
            from baselines.baseline import BaselineCache
            return BaselineCache, {}

        case "h2o":
            from baselines.h2o import H2OCache
            return H2OCache, {
                "compression_size": kwargs.get('compression_size', 0.5),
                "recent_size": kwargs.get('recent_size', 0.1),
                "sink_size": kwargs.get('sink_size', 4),
            }

        case "otkv":
            from core.ot_kv import OTKVCache
            return OTKVCache, {
                "compression_size": kwargs.get('compression_size', 0.5),
                "recent_size": kwargs.get('recent_size', 0.1),
                "sink_size": kwargs.get('sink_size', 4),
                "gamma": kwargs.get('otkv_gamma', 1.0),
                "epsilon": kwargs.get('otkv_epsilon', 0.01),
                "transport_mode": kwargs.get('otkv_transport_mode', "soft"),
                "compress_interval": kwargs.get('otkv_compress_interval', 32),
                "target_beta": kwargs.get("target_beta", 0.0),
                "sinkhorn_iters": kwargs.get("sinkhorn_iters", 50)
            }

        case "streamingllm":
            from baselines.streamingllm import StreamingLLMCache
            return StreamingLLMCache, {
                "compression_size": kwargs.get('compression_size', 1.0),
                "recent_size": kwargs.get('recent_size', 0.1),
                "sink_size": 4 if kwargs.get('sink_size') is None else kwargs.get('sink_size'),
            }

        case "snapkv":
            from baselines.snapkv import SnapKVCache
            return SnapKVCache, {
                "compression_size": kwargs.get('compression_size', 0.5),
                "recent_size": kwargs.get('recent_size', 0.1),
                "sink_size": 0 if kwargs.get('sink_size') is None else kwargs.get('sink_size'),
                "observation_window": kwargs.get('observation_window', None),
            }

        case "pyramidkv":
            from baselines.pyramidkv import PyramidKVCache
            return PyramidKVCache, {
                "compression_size": kwargs.get('compression_size', 0.5),
                "recent_size": kwargs.get('recent_size', 0.1),
                "sink_size": 0 if kwargs.get('sink_size') is None else kwargs.get('sink_size'),
                "pyramid_low_scale": kwargs.get('pyramid_low_scale', 1.5),
                "pyramid_high_scale": kwargs.get('pyramid_high_scale', 0.5),
            }

        case "echokv":
            from baselines.echokv import EchoKVCache
            return EchoKVCache, {
                "compression_size": kwargs.get('compression_size', 0.5),
                "recent_size": kwargs.get('recent_size', 0.1),
                "sink_size": 0 if kwargs.get('sink_size') is None else kwargs.get('sink_size'),
                "max_representative_scan": kwargs.get('max_representative_scan', None),
            }

        case _:
            raise ValueError(f"Unknown method: {method}")


def main(model_id, methods, tasks, **kwargs):
    device = set_device()
    print(f"\n{'=' * 60}")
    print(f"🚀 Starting Multi-Evaluation Pipeline")
    print(f"Model  : {model_id}")
    print(f"Tasks  : {', '.join(tasks)}")
    print(f"Methods: {', '.join(methods)}")
    print(f"Using device: {device}")
    print(f"{'=' * 60}\n")

    print(f">>> [Init] Loading Large Language Model ONCE into VRAM...")
    model_wrapper = EvaluatorHFLM(
        pretrained=model_id,
        cache_class=None,
        cache_kwargs={},
        prefill_fraction=kwargs.get("prefill_fraction", 0.2),
        max_length=kwargs.get("max_length", 4096),
        device=device,
    )
    print(f">>> [Init] Model loaded successfully!\n")

    summary_results = {task: {} for task in tasks}

    for task in tasks:
        print(f"\n{'=' * 60}")
        print(f"📌 Task: {task.upper()}")
        print(f"{'=' * 60}")

        try:
            evaluator_class = get_evaluator(task)
        except Exception as e:
            print(f"[Error] Failed to load evaluator for task '{task}': {e}")
            continue

        for method in methods:
            print(f"\n---> Evaluating Method: [{method.upper()}] on [{task}]")

            try:
                cache_class, cache_kwargs = get_cache_config(method, kwargs)
                model_wrapper.cache_class = cache_class
                model_wrapper.cache_kwargs = cache_kwargs
            except Exception as e:
                print(f"[Error] Failed to configure method '{method}': {e}")
                summary_results[task][method] = "Config Error"
                continue

            try:
                start_time = time.time()
                evaluator_instance = evaluator_class(model_wrapper=model_wrapper, **kwargs)
                result = evaluator_instance.evaluate()
                elapsed = time.time() - start_time

                summary_results[task][method] = result
                print(f"     ✅ Done in {elapsed:.2f}s | Result: {result}")
            except Exception as e:
                print(f"     ❌ [Evaluation Failed] {str(e)}")
                summary_results[task][method] = f"Error: {str(e)}"

    print("\n\n" + "=" * 60)
    print("🏆 FINAL EVALUATION SUMMARY")
    print("=" * 60)
    for task, method_res in summary_results.items():
        print(f"\n🔹 TASK: {task}")
        print(f"{'Method':<15} | {'Result':<20}")
        print("-" * 40)
        for method, res in method_res.items():
            res_str = f"{res:.4f}" if isinstance(res, float) else str(res)
            print(f"{method:<15} | {res_str:<20}")
    print("=" * 60 + "\n")

    export_results(summary_results, kwargs.get("save_dir", None), kwargs.get("filename", None))


def run():
    parser = argparse.ArgumentParser(description="OT-KV & KV Compression Evaluation Framework (v2: Multi-Run)")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        help="HuggingFace model repository ID or local path")
    parser.add_argument("--tasks", type=str, nargs='+', default=["wikitext"],
                        choices=["wikitext", "niah", "longbench", "profile_niah", "kv_recovery"],
                        help="Evaluation task names (space separated, e.g., wikitext niah)")
    parser.add_argument("--methods", type=str, nargs='+', default=["baseline"],
                        choices=["baseline", "otkv", "h2o", "streamingllm", "snapkv", "pyramidkv", "echokv"],
                        help="KV Cache compression methods (space separated, e.g., baseline h2o snapkv)")
    parser.add_argument("--compression_size", default=0.5,
                        help="Target KV Cache retention ratio (e.g., 0.5 means keep 50%)")
    parser.add_argument("--recent_size", default=0.1,
                        help="Size of the local/recent window for algorithms like H2O or StreamingLLM")
    parser.add_argument("--sink_size", default=4,
                        help="Number of initial/sink tokens to retain")
    parser.add_argument("--otkv_compress_interval", type=int, default=32,
                        help="Run OTKV decode-time OT compression every N decode steps")
    parser.add_argument("--prefill_fraction", type=float, default=0.1,
                        help="Fraction of document used for the initial prefill stage in PPL testing")
    parser.add_argument("--max_length", type=int, default=4096,
                        help="Maximum sequence length for the model")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of samples for evaluation (for quick debugging)")
    parser.add_argument("--wiki_docs", type=str, default=None,
                        help="Wikitext document numbers to evaluate, e.g. 1-10 or 1,2,3,4")
    parser.add_argument("--haystack_dir", type=str, default="datasets/niah/PaulGrahamEssays", )
    parser.add_argument("--longbench_dir", type=str, default="datasets/LongBench_dataset", )
    parser.add_argument("--longbench_tasks", type=str, default="all")
    parser.add_argument("--save_dir", type=str, default="runs")

    args = parser.parse_args()
    main_kwargs = vars(args)
    main(**main_kwargs)


if __name__ == "__main__":
    run()
