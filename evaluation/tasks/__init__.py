# evaluation/tasks/__init__.py
from .registry import get_evaluator

from . import wikitext
from . import niah
from . import longbench
from . import profile_niah
from . import kv_recovery

__all__ = [
    "get_evaluator",
    "wikitext",
    "niah",
    "longbench",
    "profile_niah",
    "kv_recovery"
]