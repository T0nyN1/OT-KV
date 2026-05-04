import modal
import os

app = modal.App("download-ruler")
data_volume = modal.Volume.from_name("ot_kv_data")

# 需要 huggingface_hub 来执行下载
image = modal.Image.debian_slim().pip_install("huggingface_hub")


@app.function(
    image=image,
    volumes={"/ot_kv_data": data_volume},
    timeout=3600,  # RULER 文件较多，建议保留较长超时
)
def download():
    from huggingface_hub import snapshot_download

    # 1. 定义下载路径
    save_path = "/ot_kv_data/datasets/ruler"
    os.makedirs(save_path, exist_ok=True)

    print("🚀 开始从 Hugging Face 下载 RULER 数据集...")

    # 2. 使用 snapshot_download 替代命令行 hf download
    # 这会自动处理断点续传和多线程下载
    snapshot_download(
        repo_id="llamastack/ruler",
        repo_type="dataset",
        local_dir=save_path,
        # 如果你想下载特定的子任务，可以使用 allow_patterns=["*niah*"]
        token=os.environ.get("HF_TOKEN")  # 如果是私有或受限仓库则需要
    )

    # 3. 提交更改到 Volume
    data_volume.commit()
    print(f"✅ RULER 数据集已成功下载并缓存到 {save_path} ！")


@app.local_entrypoint()
def main():
    download.remote()