# evaluation/tasks/wikitext.py
from typing import Dict, Any

from .base_evaluator import BaseEvaluator
from .registry import register_task


@register_task("wikitext")
class WikitextEvaluator(BaseEvaluator):
    """专门用于运行 lm-eval 的 wikitext PPL 测试"""

    def evaluate(self) -> Dict[str, Any]:
        from lm_eval import simple_evaluate
        print("\n[*] Running lm-eval task: ['wikitext']")

        results = simple_evaluate(
            model=self.model_wrapper,
            tasks=["wikitext"],
            limit=self.args.get('limit', None)
        )
        return results['results']
