import os
import sys

import modal

app = modal.App("ot-kv-framework-runner")
data_volume = modal.Volume.from_name("ot_kv_data")

eval_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.11.0",
        "transformers==5.7.0",
        "accelerate",
        "lm-eval",
        "wonderwords",
        "nltk",
        "datasets",
        "tiktoken",
        "hf_transfer"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir(
        local_path=".",
        remote_path="/app",
        ignore=[".git", "__pycache__", ".idea", ".vscode", "venv", "env"]
    )
)


@app.function(
    image=eval_image,
    gpu="H200",
    volumes={"/ot_kv_data": data_volume},
    timeout=7200,
)
def run_framework_on_modal(
        model_id: str,
        methods: list[str],
        tasks: list[str],
        prefill_fraction: float,
        max_length: int,
        limit: int = None,
        **kwargs
):
    sys.path.append("/app")
    os.chdir("/app")

    os.environ["HF_DATASETS_CACHE"] = "/ot_kv_data/datasets"
    os.environ["HF_HOME"] = "/ot_kv_data/models"

    from main import main as custom_main

    print("=" * 60)
    print("Launching modal evaluation pipeline...")
    print(f"Model:   {model_id}")
    print(f"Methods:   {', '.join(methods)}")
    print(f"Tasks:   {', '.join(tasks)}")
    print(f"Hyperparameters: {kwargs}")
    print("=" * 60)

    save_dir = kwargs.get("save_dir", None)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    custom_main(
        model_id=model_id,
        methods=methods,
        tasks=tasks,
        prefill_fraction=prefill_fraction,
        max_length=max_length,
        limit=limit,
        **kwargs
    )

    data_volume.commit()


@app.local_entrypoint()
def run(
        model_id: str = "/ot_kv_data/models/Llama-3.1-8B-Instruct",
        methods: str = "otkv",
        tasks: str = "longbench,profile_niah",
        prefill_fraction: float = 0.5,
        max_length: int = 12000,
        limit: int = 0,
        wiki_docs: str = "",
        compression_size=0.5,
        recent_size=0.1,
        sink_size=4,
        mode: str = "prefill",
):
    method_list = [m.strip() for m in methods.split(",")]
    task_list = [t.strip() for t in tasks.split(",")]

    kwargs = {
        "compression_size": compression_size,
        "recent_size": recent_size,
        "sink_size": sink_size,
        "mode": mode,
        "haystack_dir": "/ot_kv_data/datasets/niah/PaulGrahamEssays",
        "longbench_dir": "/ot_kv_data/datasets/LongBench_Dataset",
        "longbench_tasks": "qasper",
        "save_dir": "/ot_kv_data/runs",
        "sinkhorn_iters": 50
    }
    if wiki_docs:
        kwargs["wiki_docs"] = wiki_docs

    eval_limit = limit if limit > 0 else None

    run_framework_on_modal.remote(
        model_id=model_id,
        methods=method_list,
        tasks=task_list,
        prefill_fraction=prefill_fraction,
        max_length=max_length,
        limit=eval_limit,
        **kwargs
    )
