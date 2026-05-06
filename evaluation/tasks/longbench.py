import re
import string
from collections import Counter
from typing import Dict, Any

from .base_evaluator import BaseEvaluator
from .registry import register_task


def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    return ' '.join(s.split())


def _qa_f1(prediction: str, ground_truths: list) -> float:
    pred_tokens = _normalize(prediction).split()
    best = 0.0
    for gt in ground_truths:
        gold_tokens = _normalize(str(gt)).split()
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        best = max(best, (2 * precision * recall) / (precision + recall))
    return best


LONGBENCH_METRIC = "f1"


@register_task("longbench")
class LongBenchEvaluator(BaseEvaluator):

    def evaluate(self) -> Dict[str, Any]:
        import json
        import os
        import urllib.request
        import zipfile
        import torch
        from tqdm import tqdm

        max_length = self.args.get('max_length', 7500)
        tokenizer = self.model_wrapper.tokenizer
        model = self.model_wrapper._model

        longbench_dir = self.args.get("longbench_dir", "./datasets/LongBench_dataset")
        data_folder = os.path.join(longbench_dir, "data")

        if not os.path.exists(data_folder):
            print(f"\n[*] Downloading LongBench directly to {longbench_dir}...")
            os.makedirs(longbench_dir, exist_ok=True)
            zip_url = "https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip"
            zip_path = os.path.join(longbench_dir, "data.zip")
            urllib.request.urlretrieve(zip_url, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(longbench_dir)

        tasks_arg = self.args.get('longbench_tasks', "all")

        if tasks_arg.strip().lower() == "all" or not tasks_arg:

            tasks = [f.replace(".jsonl", "") for f in os.listdir(data_folder) if f.endswith(".jsonl")]
            print(f"\n[*] No specific tasks provided. Found {len(tasks)} tasks in dataset directory.")
        else:
            tasks = tasks_arg.split(",")

        all_results = {}
        for task in tasks:
            task = task.strip()
            print(f"\n[*] Running LongBench task: {task}")
            file_path = os.path.join(data_folder, f"{task}.jsonl")

            if not os.path.exists(file_path):
                print(f"[!] Warning: Task file {file_path} not found. Skipping...")
                continue

            dataset = [json.loads(line) for line in open(file_path, 'r', encoding='utf-8')]
            limit = self.args.get('limit', None)
            if limit:
                dataset = dataset[:limit]

            task_score = 0
            for item in tqdm(dataset, desc=f"Evaluating {task}"):
                context = item['context']
                query = item['input']
                answers = item["answers"]

                if query.strip():
                    prompt = f"Please read the following context and answer the question.\n\nContext:\n{context}\n\nQuestion:\n{query}"
                else:

                    prompt = context
                messages = [{"role": "user", "content": prompt}]
                input_tensor = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )["input_ids"].to(model.device)

                if input_tensor.shape[1] > max_length:
                    half = max_length // 2
                    input_tensor = torch.cat([input_tensor[:, :half], input_tensor[:, -half:]], dim=1)

                custom_cache = self.model_wrapper._setup_cache_and_hooks()

                with torch.no_grad():
                    output_ids = model.generate(
                        input_tensor,
                        attention_mask=torch.ones_like(input_tensor),
                        max_new_tokens=64,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                        past_key_values=custom_cache,
                        use_cache=True
                    )
                self._cleanup_cache_and_hooks(custom_cache)

                generated_tokens = output_ids[0][input_tensor.shape[1]:]
                response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

                if LONGBENCH_METRIC == "f1":
                    task_score += _qa_f1(response, answers)
                else:
                    response_lower = response.lower()
                    task_score += 1 if any(str(ans).lower() in response_lower for ans in answers) else 0

            score = task_score / len(dataset) if len(dataset) > 0 else 0
            all_results[task] = {
                LONGBENCH_METRIC: score,
                "tested_samples": len(dataset)
            }
            print(f"-> {task} {LONGBENCH_METRIC.upper()}: {score * 100:.2f}%")

        return {"longbench": all_results}
