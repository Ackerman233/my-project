"""Training-free dynamic skill execution components."""

from pathlib import Path
from .infrastructure.compat import install_aliases

install_aliases(Path(__file__).with_name('migration_20261008.json'))

from .execution.checker import CheckResult, CheckStatus, ObservationChecker
from .execution.interrupter import Decision, DecisionType, Interrupter
from .knowledge.skill import SkillPlan, SkillStep
from .execution.skill_executor import ChunkExecutor, ExecutionEvent

__all__ = [
    "CheckResult",
    "CheckStatus",
    "ChunkExecutor",
    "Decision",
    "DecisionType",
    "ExecutionEvent",
    "Interrupter",
    "ObservationChecker",
    "SkillPlan",
    "SkillStep",
]
