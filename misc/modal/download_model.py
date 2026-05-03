import modal
import os

app = modal.App("download-llama-3-1")
model_volume = modal.Volume.from_name("ot_kv_data")

image = modal.Image.debian_slim().pip_install("huggingface_hub", "hf_transfer")

@app.function(
    image=image,
    volumes={"/ot_kv_data": model_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
)
def download_model():
    import os
    os.makedirs("/ot_kv_data/models", exist_ok=True)
    from huggingface_hub import snapshot_download
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    print("🚀 开始下载 Llama-3.1-8B-Instruct...")
    snapshot_download(
        repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        local_dir="/ot_kv_data/models/Llama-3.1-8B-Instruct",
        ignore_patterns=["*.pt", "*.bin"],  # 只下载 safetensors 版本
        token=os.environ["HF_TOKEN"]
    )
    model_volume.commit()
    print("✅ 模型下载并保存到 Volume 成功！")


@app.local_entrypoint()
def main():
    download_model.remote()