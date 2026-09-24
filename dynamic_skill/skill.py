from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class SkillStep:
    """One controller-level chunk in a long-horizon plan."""

    step_id: str
    objective: str
    completion_signals: tuple[str, ...] = ()
    max_actions: int = 4
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.step_id.strip():
            raise ValueError("step_id must be non-empty")
        if not self.objective.strip():
            raise ValueError("objective must be non-empty")
        if self.max_actions < 1:
            raise ValueError("max_actions must be positive")


@dataclass(frozen=True)
class SkillPlan:
    """Serializable plan produced without policy training."""

    name: str
    task: str
    steps: tuple[SkillStep, ...]
    revision: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("a skill plan needs at least one step")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def prompt(self, active_index: int = 0) -> str:
        active_index = max(0, min(active_index, len(self.steps) - 1))
        lines = [
            f"Plan: {self.name} (revision {self.revision})",
            f"Task: {self.task}",
            "Controller chunks:",
        ]
        for index, step in enumerate(self.steps):
            marker = "ACTIVE" if index == active_index else "pending"
            lines.append(f"- [{marker}] {step.step_id}: {step.objective}")
        return "\n".join(lines)

