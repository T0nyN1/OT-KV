import modal

app = modal.App("ot-kv-eval")
# 直接挂载你截图里的 ot_kv_data
data_volume = modal.Volume.from_name("ot_kv_data")

eval_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "lm-eval",
        "datasets",
        "tiktoken"
    )
)


@app.cls(
    image=eval_image,
    gpu="A100-80GB",
    volumes={"/ot_kv_data": data_volume},  # 映射到容器内的统一入口
    timeout=7200,
)
class BaselineEvaluator:
    @modal.enter()
    def load_model(self):
        print("⏳ 正在将模型加载到显存中...")
        from lm_eval.models.huggingface import HFLM

        self.model = HFLM(
            # 指向你 Volume 里的模型路径
            pretrained="/ot_kv_data/models/Llama-3.1-8B-Instruct",
            backend="causal",
            device="cuda",
            dtype="bfloat16",
            # 测 Wikitext PPL 通常不需要极致的长上下文，设为 4096 或 8192 足够，能节省大量显存
            max_length=4096,
        )
        print("✅ 模型加载完毕！")

    @modal.method()
    def evaluate(self, tasks: list[str], limit: int = None):
        import lm_eval
        print(f"📊 开始评测任务: {tasks}")

        results = lm_eval.simple_evaluate(
            model=self.model,
            tasks=tasks,
            limit=limit,
            batch_size="auto",  # 自动拉满显存利用率
        )

        # 为了方便你看结果，专门为 wikitext 提取 PPL 打印
        if "wikitext" in tasks:
            # lm-eval v0.4+ 的 metric 命名带有 ,none 后缀
            metrics = results["results"].get("wikitext", {})
            word_ppl = metrics.get("word_perplexity,none")
            byte_ppl = metrics.get("byte_perplexity,none")
            print("\n" + "=" * 40)
            print(f"🎯 Wikitext 评测结果:")
            print(f"Word Perplexity: {word_ppl:.4f}" if word_ppl else "未获取到 Word PPL")
            print(f"Byte Perplexity: {byte_ppl:.4f}" if byte_ppl else "未获取到 Byte PPL")
            print("=" * 40 + "\n")

        return results["results"]


@app.local_entrypoint()
def main():
    evaluator = BaselineEvaluator()

    # 开始执行评估。
    # 提示：如果是验证代码逻辑，建议先把 limit 设为 50 跑个测试；
    # 确认没问题后，再把 limit=None 跑全量数据集。
    evaluator.evaluate.remote(tasks=["wikitext"], limit=10)