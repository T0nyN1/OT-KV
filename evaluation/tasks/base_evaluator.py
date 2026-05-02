# evaluation/tasks/base_evaluator.py
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseEvaluator(ABC):
    """所有评测任务的抽象基类"""

    def __init__(self, model_wrapper, **kwargs):
        self.model_wrapper = model_wrapper
        self.args = kwargs

    @abstractmethod
    def evaluate(self) -> Dict[str, Any]:
        """执行评测并返回包含结果的字典"""
        pass
