import os
import sys
import modal

app = modal.App("ot-kv-framework-runner")
data_volume = modal.Volume.from_name("ot_kv_data")

# 1. 配置运行环境，并把代码挂载逻辑直接链式写在 Image 里
eval_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "lm-eval",
        "wonderwords",
        "nltk",
        "datasets",
        "tiktoken",
        "hf_transfer"  # 推荐加入：加速 HuggingFace 上的模型下载
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # 开启 hf_transfer
    .add_local_dir(
        local_path=".",
        remote_path="/app",
        ignore=[".git", "__pycache__", ".idea", ".vscode", "venv", "env"]
    )
)


@app.function(
    image=eval_image,
    gpu="A100-80GB",  # 可以根据你的预算和需求调整，如 "H100" 或 "L40S"
    volumes={"/ot_kv_data": data_volume},
    timeout=7200,
)
def run_framework_on_modal(
        model_id: str,
        method: str,
        task: str,
        prefill_fraction: float,
        max_length: int,
        limit: int = None,
        **kwargs
):
    # 将挂载的工作目录加入 Python 路径
    sys.path.append("/app")
    os.chdir("/app")

    # 统一将各种缓存指引到挂载的 Volume，避免每次重启实例重新下载
    os.environ["HF_DATASETS_CACHE"] = "/ot_kv_data/datasets"
    os.environ["HF_HOME"] = "/ot_kv_data/models"

    # 动态导入重构后的主函数
    from main import main as custom_main

    print("=" * 50)
    print("🚀 启动 Modal 远程评测流水线")
    print(f"📦 模型:   {model_id}")
    print(f"🧠 策略:   {method}")
    print(f"🎯 任务:   {task}")
    if method != "baseline":
        print(f"⚙️  超参数: {kwargs}")
    print("=" * 50)

    # 调用新架构的主函数，kwargs 会把 compression_ratio 等自动传给 Cache 类
    custom_main(
        model_id=model_id,
        method=method,
        task=task,
        prefill_fraction=prefill_fraction,
        max_length=max_length,
        limit=limit,
        **kwargs
    )


@app.local_entrypoint()
def run(
        # 基础配置
        model_id: str = "/ot_kv_data/models/Llama-3.1-8B-Instruct",
        method: str = "h2o",
        task: str = "wikitext",
        prefill_fraction: float = 0.1,
        max_length: int = 4096,
        limit: int = 2,
        compression_ratio: float = 0.5,
        recent_window: int = 256,
        sink_size: int = 4,
):
    """
    你可以直接在本地命令行通过 flags 覆盖这些默认参数。
    """

    # 将命令行参数打包为 kwargs 字典
    kwargs = {
        "compression_ratio": compression_ratio,
        "recent_window": recent_window,
        "sink_size": sink_size,
    }

    # 处理 limit 的特殊情况
    eval_limit = limit if limit > 0 else None

    # 触发远程执行
    run_framework_on_modal.remote(
        model_id=model_id,
        method=method,
        task=task,
        prefill_fraction=prefill_fraction,
        max_length=max_length,
        limit=eval_limit,
        **kwargs
    )