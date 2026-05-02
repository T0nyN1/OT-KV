# evaluation/tasks/registry.py
TASK_REGISTRY = {}


def register_task(name: str):
    """任务注册装饰器"""

    def decorator(cls):
        TASK_REGISTRY[name] = cls
        return cls

    return decorator


def get_evaluator(name: str):
    """根据名称获取对应的评测类"""
    if name not in TASK_REGISTRY:
        raise ValueError(f"Task '{name}' not found. Available tasks: {list(TASK_REGISTRY.keys())}")
    return TASK_REGISTRY[name]
