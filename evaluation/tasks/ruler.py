# evaluation/tasks/ruler.py
import json
import os
import glob
from typing import Dict, Any

import torch
import pandas as pd  # 新增：用于处理 Parquet
from tqdm import tqdm

from .base_evaluator import BaseEvaluator
from .registry import register_task


@register_task("ruler")
class RulerEvaluator(BaseEvaluator):
    """
    RULER 评测任务适配器。
    已优化：支持 Hugging Face 下载的目录结构及 Parquet 格式。
    """

    def evaluate(self) -> Dict[str, Any]:
        # 参数获取
        tasks_str = self.args.get('ruler_tasks', "niah_single_1,niah_multivalue")
        tasks = [t.strip() for t in tasks_str.split(",") if t.strip()]
        max_length = self.args.get('ruler_max_length', 8000)
        tokenizer = self.model_wrapper.tokenizer
        model = self.model_wrapper._model
        data_dir = self.args.get("ruler_data_dir", "./datasets/ruler")

        if not os.path.exists(data_dir):
            print(f"\n[!] 错误: RULER 数据目录未找到: {data_dir}")
            return {"ruler": {"error": "Dataset not found"}}

        all_results = {}
        for task in tasks:
            print(f"\n[*] Running RULER task: {task}")

            # --- 修改部分：路径匹配逻辑 ---
            # 支持模糊匹配文件夹，例如输入 niah_single_1 匹配 niah_single_1_128000 文件夹
            search_pattern = os.path.join(data_dir, f"{task}*")
            matches = glob.glob(search_pattern)

            if not matches:
                print(f" -> 警告: 找不到任务路径 {search_pattern}，已跳过。")
                continue

            task_path = matches[0]
            dataset = []

            # --- 修改部分：加载 Parquet 或 JSONL 数据 ---
            if os.path.isdir(task_path):
                # 1. 优先查找 Parquet 文件 (Hugging Face 默认格式)
                parquet_files = glob.glob(os.path.join(task_path, "**/*.parquet"), recursive=True)
                if parquet_files:
                    for pf in parquet_files:
                        df = pd.read_parquet(pf)
                        dataset.extend(df.to_dict(orient='records'))
                else:
                    # 2. 如果没有 Parquet，尝试查找 JSONL
                    jsonl_files = glob.glob(os.path.join(task_path, "**/*.jsonl"), recursive=True)
                    for jf in jsonl_files:
                        dataset.extend([json.loads(line) for line in open(jf, 'r', encoding='utf-8')])
            else:
                # 3. 处理单文件情况
                if task_path.endswith(".parquet"):
                    dataset = pd.read_parquet(task_path).to_dict(orient='records')
                else:
                    dataset = [json.loads(line) for line in open(task_path, 'r', encoding='utf-8')]

            if not dataset:
                print(f" -> 警告: 任务 {task} 数据为空，已跳过。")
                continue

            limit = self.args.get('limit', None)
            if limit:
                dataset = dataset[:limit]

            task_score = 0
            valid_samples = 0

            # 遍历评测
            for item in tqdm(dataset, desc=f"Evaluating {task}"):
                # 适配字段名：Hugging Face 版本的 RULER 字段通常也是 input 和 outputs
                prompt = item.get("input", "")
                answers = item.get("outputs", [])

                if not prompt or not answers:
                    continue

                input_ids = tokenizer.encode(prompt, add_special_tokens=False)

                # 截断策略：保留前 100 和后 (max-100) 以确保指令完整性
                if len(input_ids) > max_length:
                    input_ids = input_ids[:100] + input_ids[-(max_length - 100):]

                input_tensor = torch.tensor([input_ids]).to(model.device)
                custom_cache = self.model_wrapper._setup_cache_and_hooks()

                try:
                    with torch.no_grad():
                        output_ids = model.generate(
                            input_tensor,
                            max_new_tokens=128,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                            past_key_values=custom_cache,
                            output_attentions=True,
                            use_cache=True
                        )
                finally:
                    self._cleanup_cache_and_hooks(custom_cache)

                generated_tokens = output_ids[0][input_tensor.shape[1]:]
                response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip().lower()

                # 评分：模型输出包含任意一个标准答案即得分
                is_correct = any(str(ans).lower() in response for ans in answers)
                task_score += 1 if is_correct else 0
                valid_samples += 1

            accuracy = task_score / valid_samples if valid_samples > 0 else 0
            all_results[task] = {"accuracy": accuracy}
            print(f"-> {task} Accuracy: {accuracy * 100:.2f}%")

        overall_acc = sum(res["accuracy"] for res in all_results.values()) / len(all_results) if all_results else 0
        all_results["overall_accuracy"] = overall_acc
        print(f"\n=> RULER Overall Accuracy: {overall_acc * 100:.2f}%")

        return {"ruler": all_results}