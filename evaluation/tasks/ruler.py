# evaluation/tasks/ruler.py
from typing import Dict, Any
from .registry import register_task
from .base_evaluator import BaseEvaluator


@register_task("ruler")
class RulerEvaluator(BaseEvaluator):

    def evaluate(self) -> Dict[str, Any]:
        raise NotImplementedError