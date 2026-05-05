import os
import ast
from datetime import datetime
from typing import Dict, Any, Optional

import pandas as pd
import torch


def set_device():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    return device


def export_results(summary_results: Dict[str, Dict[str, Any]], save_dir: Optional[str] = None,
                   filename: Optional[str] = None) -> None:
    def _normalize_metrics(result: Any, task_name: str) -> Any:
        if isinstance(result, str):
            try:
                parsed = ast.literal_eval(result)
                if isinstance(parsed, dict):
                    result = parsed
            except (ValueError, SyntaxError):
                pass

        if isinstance(result, dict):
            if task_name in result:
                return _normalize_metrics(result[task_name], task_name)

            parsed_dict = {}
            for k, v in result.items():
                if isinstance(v, str) and v.strip().startswith("{"):
                    try:
                        v_parsed = ast.literal_eval(v)
                        parsed_dict[k] = v_parsed if isinstance(v_parsed, dict) else v
                    except (ValueError, SyntaxError):
                        parsed_dict[k] = v
                else:
                    parsed_dict[k] = v
            return parsed_dict

        return result

    print("Exporting results...")
    formatted_data = {}

    for task_name, method_res in summary_results.items():
        for method_name, result in method_res.items():
            if method_name not in formatted_data:
                formatted_data[method_name] = {}

            metrics_dict = _normalize_metrics(result, task_name)

            if isinstance(metrics_dict, dict):
                for metric_name, metric_value in metrics_dict.items():
                    formatted_data[method_name][(task_name, metric_name)] = metric_value
            else:
                formatted_data[method_name][(task_name, "Score")] = metrics_dict

    if not formatted_data:
        print("No results found. Export aborted.")
        return

    df = pd.DataFrame.from_dict(formatted_data, orient='index')

    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df = df.reset_index()
    new_columns = [("Task", "Model")] + df.columns.tolist()[1:]
    df.columns = pd.MultiIndex.from_tuples(new_columns)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir if save_dir is not None else "",
                             f"{filename if filename is not None else "eval_results"}_{timestamp}.csv")
    df.to_csv(save_path, index=False, encoding="utf-8")
    print(f"Results saved to: {save_path}")
