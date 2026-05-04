# evaluation/tasks/longbench.py
import json
import os
import urllib.request
import zipfile
from typing import Dict, Any

import torch
from tqdm import tqdm

from .base_evaluator import BaseEvaluator
from .registry import register_task


@register_task("longbench")
class LongBenchEvaluator(BaseEvaluator):
    """LongBench 原生评测任务"""

    def evaluate(self) -> Dict[str, Any]:
        tasks = self.args.get('longbench_tasks', "qasper").split(",")
        max_length = self.args.get('longbench_max_length', 7500)
        tokenizer = self.model_wrapper.tokenizer
        model = self.model_wrapper._model

        data_dir = "datasets/LongBench_dataset"
        data_folder = os.path.join(data_dir, "data")

        if not os.path.exists(data_folder):
            print(f"\n[*] Downloading LongBench directly to {data_dir}...")
            os.makedirs(data_dir, exist_ok=True)
            zip_url = "https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip"
            zip_path = os.path.join(data_dir, "data.zip")
            urllib.request.urlretrieve(zip_url, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)

        all_results = {}
        for task in tasks:
            task = task.strip()
            print(f"\n[*] Running LongBench task: {task}")
            file_path = os.path.join(data_folder, f"{task}.jsonl")

            if not os.path.exists(file_path):
                continue

            dataset = [json.loads(line) for line in open(file_path, 'r', encoding='utf-8')]
            if self.args.get('limit', None):
                dataset = dataset[:self.args.get('limit')]

            task_score = 0
            for item in tqdm(dataset, desc=f"Evaluating {task}"):
                prompt = f"Context:\n{item['context']}\n\nQuestion:\n{item['input']}\n\nAnswer:"
                input_ids = tokenizer.encode(prompt, add_special_tokens=False)

                if len(input_ids) > max_length:
                    half = max_length // 2
                    input_ids = input_ids[:half] + input_ids[-half:]

                input_tensor = torch.tensor([input_ids]).to(model.device)

                # ==================================================
                # [核心适配] 初始化自定义 Cache
                # ==================================================
                custom_cache = self.model_wrapper._setup_cache_and_hooks()

                try:
                    with torch.no_grad():
                        output_ids = model.generate(
                            input_tensor,
                            max_new_tokens=64,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                            past_key_values=custom_cache,  # 传入缓存策略
                            output_attentions=True,        # 开启 Attention 抓取
                            use_cache=True
                        )
                finally:
                    self._cleanup_cache_and_hooks(custom_cache)

                response = tokenizer.decode(output_ids[0][input_tensor.shape[1]:],
                                            skip_special_tokens=True).strip().lower()
                is_correct = any(str(ans).lower() in response for ans in item["answers"])
                task_score += 1 if is_correct else 0

            accuracy = task_score / len(dataset) if dataset else 0
            all_results[task] = {"accuracy": accuracy}
            print(f"-> {task} Accuracy: {accuracy * 100:.2f}%")

        return {"longbench": all_results}
