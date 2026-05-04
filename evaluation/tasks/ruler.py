# evaluation/tasks/ruler.py
import json
import os
from typing import Dict, Any

import torch
from tqdm import tqdm

from .base_evaluator import BaseEvaluator
from .registry import register_task


@register_task("ruler")
class RulerEvaluator(BaseEvaluator):
    """
    RULER (Real Context Size) 评测任务适配器。
    支持 RULER 的各类子任务，如 niah_single, niah_multivalue, vt (Variable Tracking), cwe 等。
    """

    def evaluate(self) -> Dict[str, Any]:
        # 从命令行参数获取要运行的 RULER 任务列表，逗号分隔
        # 默认运行单针和多针测试作为演示
        tasks_str = self.args.get('ruler_tasks', "niah_single_1,niah_multivalue")
        tasks = [t.strip() for t in tasks_str.split(",") if t.strip()]

        max_length = self.args.get('ruler_max_length', 8000)  # 根据你的显存调整

        tokenizer = self.model_wrapper.tokenizer
        model = self.model_wrapper._model

        # RULER 数据集目录约定
        data_dir = self.args.get("ruler_data_dir", "./datasets/ruler")

        if not os.path.exists(data_dir):
            print(f"\n[!] 错误: RULER 数据目录未找到: {data_dir}")
            print("请先使用 RULER 官方脚本生成数据，并存放在该目录下。")
            print("数据格式应为每行一个 JSON: {\"input\": \"...\", \"outputs\": [\"ans1\", \"ans2\"]}")
            return {"ruler": {"error": "Dataset not found"}}

        all_results = {}
        for task in tasks:
            print(f"\n[*] Running RULER task: {task}")
            file_path = os.path.join(data_dir, f"{task}.jsonl")

            if not os.path.exists(file_path):
                print(f" -> 警告: 找不到任务文件 {file_path}，已跳过。")
                continue

            # 加载数据
            dataset = [json.loads(line) for line in open(file_path, 'r', encoding='utf-8')]
            limit = self.args.get('limit', None)
            if limit:
                dataset = dataset[:limit]

            task_score = 0
            valid_samples = 0

            # 遍历评测
            for item in tqdm(dataset, desc=f"Evaluating {task}"):
                # RULER 数据集通常包含 'input' (长文本提示) 和 'outputs' (正确答案列表)
                prompt = item.get("input", "")
                answers = item.get("outputs", [])

                if not prompt or not answers:
                    continue

                input_ids = tokenizer.encode(prompt, add_special_tokens=False)

                # 如果超长，进行简单的从后截断 (RULER 的指令通常在前面和最后，中间是 Context)
                if len(input_ids) > max_length:
                    # 保留前 100 个 token（通常是系统指令），剩下的从末尾截取
                    input_ids = input_ids[:100] + input_ids[-(max_length - 100):]

                input_tensor = torch.tensor([input_ids]).to(model.device)

                # ==================================================
                # [核心适配] 初始化当前测试的自定义 Cache 并挂载 Hook
                # ==================================================
                custom_cache = self.model_wrapper._setup_cache_and_hooks()

                try:
                    with torch.no_grad():
                        # RULER 任务的答案通常不长，128 个 token 足够覆盖绝大多数任务
                        output_ids = model.generate(
                            input_tensor,
                            max_new_tokens=128,
                            do_sample=False,  # 贪心解码
                            pad_token_id=tokenizer.eos_token_id,
                            past_key_values=custom_cache,  # 传入当前压缩 Cache 实例
                            output_attentions=True,  # 必须开启以供 Hook 抓取分数
                            use_cache=True
                        )
                finally:
                    self._cleanup_cache_and_hooks(custom_cache)

                # 提取模型新生成的 token
                generated_tokens = output_ids[0][input_tensor.shape[1]:]
                response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip().lower()

                # RULER 的评测逻辑通常是：模型输出包含任意一个 target answer 即算正确
                is_correct = any(str(ans).lower() in response for ans in answers)
                task_score += 1 if is_correct else 0
                valid_samples += 1

            accuracy = task_score / valid_samples if valid_samples > 0 else 0
            all_results[task] = {"accuracy": accuracy}
            print(f"-> {task} Accuracy: {accuracy * 100:.2f}%")

        # 计算一个宏观平均分
        overall_acc = sum(res["accuracy"] for res in all_results.values()) / len(all_results) if all_results else 0
        all_results["overall_accuracy"] = overall_acc

        print(f"\n=> RULER Overall Accuracy: {overall_acc * 100:.2f}%")

        return {"ruler": all_results}
