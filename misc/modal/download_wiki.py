import modal
import os

app = modal.App("download-wikitext")
data_volume = modal.Volume.from_name("ot_kv_data")

# 只需要 datasets 库即可
image = modal.Image.debian_slim().pip_install("datasets")


@app.function(
    image=image,
    volumes={"/ot_kv_data": data_volume},
    timeout=3600,
)
def download():
    import os
    # 关键：将 Hugging Face datasets 的缓存路径指向我们的持久化存储卷
    os.environ["HF_DATASETS_CACHE"] = "/ot_kv_data/datasets"
    os.makedirs("/ot_kv_data/datasets", exist_ok=True)

    from datasets import load_dataset

    print("🚀 开始下载 Wikitext 数据集...")
    # lm-eval 中 "wikitext" task 默认使用的是 wikitext-2-raw-v1
    load_dataset("wikitext", "wikitext-2-raw-v1")

    # 提交更改到 Volume
    data_volume.commit()
    print("✅ Wikitext 数据集已成功下载并永久缓存到 /ot_kv_data/datasets ！")


@app.local_entrypoint()
def main():
    download.remote()