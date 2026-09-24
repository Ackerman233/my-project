from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .checker import CheckResult, ObservationChecker
from .interrupter import Decision, DecisionType, Interrupter
from .skill import SkillPlan, SkillStep
from .skill_induction import SkillInducer


@dataclass(frozen=True)
class ExecutionEvent:
    decision: Decision
    check: CheckResult
    active_step: str
    plan_revision: int
    interruption_count: int


class ChunkExecutor:
    """Stateful chunk controller around a one-action-at-a-time policy."""

    def __init__(
        self,
        *,
        chunk_size: int = 4,
        dynamic: bool = True,
        checker: ObservationChecker | None = None,
        interrupter: Interrupter | None = None,
        inducer: SkillInducer | None = None,
    ) -> None:
        self.dynamic = dynamic
        self.checker = checker or ObservationChecker()
        self.interrupter = interrupter or Interrupter()
        self.inducer = inducer or SkillInducer(chunk_size)
        self.plan: SkillPlan | None = None
        self.active_index = 0
        self.actions_in_chunk = 0
        self.interruption_count = 0
        self.action_history: list[str] = []
        self.observation_history: list[str] = []
        self.last_stable_index = 0

    @property
    def active_step(self) -> SkillStep:
        if self.plan is None:
            raise RuntimeError("executor has not been started")
        return self.plan.steps[min(self.active_index, len(self.plan.steps) - 1)]

    def start(self, initial_observation: str, task: str | None = None) -> SkillPlan:
        self.plan = self.inducer.induce(initial_observation, task)
        self.observation_history = [initial_observation]
        return self.plan

    def guidance(self) -> str:
        if self.plan is None:
            raise RuntimeError("executor has not been started")
        return (
            "## Dynamic controller\n"
            + self.plan.prompt(self.active_index)
            + "\nExecute only one admissible environment action for the ACTIVE chunk. "
            "The controller will check the resulting observation and may interrupt."
        )

    def _step_completed(self, observation: str) -> bool:
        normalized = self.checker.normalize(observation)
        return any(signal.lower() in normalized for signal in self.active_step.completion_signals)

    def observe(
        self,
        action: str,
        observation: str,
        *,
        reward: float = 0.0,
        done: bool = False,
        info: dict[str, Any] | None = None,
    ) -> ExecutionEvent:
        check = self.checker.check(
            action,
            observation,
            self.action_history,
            self.observation_history,
            reward,
            done,
            info,
        )
        self.actions_in_chunk += 1
        step_completed = self._step_completed(observation)
        decision = self.interrupter.decide(
            check,
            interruption_count=self.interruption_count,
            chunk_exhausted=self.actions_in_chunk >= self.active_step.max_actions,
            step_completed=step_completed,
            dynamic=self.dynamic,
        )
        active_step_id = self.active_step.step_id

        if decision.kind is DecisionType.ADVANCE:
            self.last_stable_index = self.active_index
            self.active_index = min(self.active_index + 1, len(self.plan.steps) - 1)  # type: ignore[union-attr]
            self.actions_in_chunk = 0
        elif decision.kind is DecisionType.REPLAN:
            self.interruption_count += 1
            self.plan = self.inducer.replan(
                self.plan,  # type: ignore[arg-type]
                self.active_index,
                decision.reason,
                observation,
            )
            self.active_index = 0
            self.actions_in_chunk = 0

        self.action_history.append(action)
        self.observation_history.append(observation)
        return ExecutionEvent(
            decision,
            check,
            active_step_id,
            self.plan.revision,  # type: ignore[union-attr]
            self.interruption_count,
        )

