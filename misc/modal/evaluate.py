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
        "tiktoken"
    )
    # 核心修改：使用 .add_local_dir 替代原本的 modal.Mount
    .add_local_dir(
        local_path=".",
        remote_path="/app",
        ignore=[".git", "__pycache__", ".idea", ".vscode", "venv", "env"]
    )
)


@app.function(
    image=eval_image,
    gpu="A100-80GB",
    volumes={"/ot_kv_data": data_volume},
    timeout=7200,
)
def run_framework_on_modal(model_id: str, method: str, task: str, prefill_fraction=0.1, max_length=4096,
                           **kwargs):
    # 将挂载的工作目录加入 Python 路径，确保你的本地 import 不会报错
    sys.path.append("/app")
    os.chdir("/app")
    os.environ["HF_DATASETS_CACHE"] = "/ot_kv_data/datasets"

    # 动态导入你的 main 函数
    from main import main as custom_main
    print(f"🚀 在 Modal 上启动评测流水线...")
    print(f"📦 Method: {method} | Task: {task} | Model: {model_id}")

    # 调用你自己的主函数
    custom_main(model_id=model_id, method=method, task=task, prefill_fraction=prefill_fraction, max_length=max_length,
                **kwargs)


@app.local_entrypoint()
def run():
    # 注意：这里传入的是 Modal Volume 里模型的绝对路径
    modal_model_path = "/ot_kv_data/models/Llama-3.1-8B-Instruct"

    # 远程执行
    run_framework_on_modal.remote(
        model_id=modal_model_path,
        method="h2o",
        task="wikitext",
        prefill_fraction=0.1,
        max_length=4096,
        limit=10
    )
