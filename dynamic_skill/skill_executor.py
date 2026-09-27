from __future__ import annotations

from dataclasses import dataclass
from collections import deque
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
    discarded_actions: tuple[str, ...] = ()


class ChunkExecutor:
    """Execute a model-generated action queue without calling the model per step."""

    def __init__(
        self,
        *,
        chunk_size: int = 4,
        dynamic: bool = True,
        checker: ObservationChecker | None = None,
        interrupter: Interrupter | None = None,
        inducer: SkillInducer | None = None,
    ) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = chunk_size
        self.pending_actions: deque[str] = deque()
        self.chunk_id = 0
        self.chunk_length = 0
        self.boundary_count = 0
        self.anomaly_replan_count = 0
        self.last_replan_reason = ""
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
            "## Action chunk planning (output contract overrides single-action instructions above)\n"
            + f"Task: {self.plan.task}\n"
            + f"Return 1 to {self.chunk_size} concrete ALFWorld actions in execution order. "
            "Plan multiple actions when their objects and preconditions are known. "
            "Stop the plan before a step that needs information not yet observed; never invent object IDs. "
            "The first action must be currently admissible; later actions must be valid after preceding actions. "
            "These actions will execute without another model call between them. "
            "Output <think>brief rationale</think><actions>[\"action 1\", \"action 2\"]</actions>. "
            "Use a JSON array of strings, not subgoal descriptions, conditionals, or a single multi-command string.\n"
            + (f"Previous anomaly to recover from: {self.last_replan_reason}\n" if self.last_replan_reason else "")
        )

    @property
    def needs_plan(self) -> bool:
        return not self.pending_actions

    def load_chunk(self, actions: list[str]) -> None:
        if self.pending_actions:
            raise RuntimeError("cannot replace an unconsumed action queue")
        if not 1 <= len(actions) <= self.chunk_size:
            raise ValueError("chunk length must be between 1 and chunk_size")
        if any(not isinstance(a, str) or not a.strip() or any(c in a for c in "\n\r<>") for a in actions):
            raise ValueError("each queue entry must be one non-empty action")
        self.pending_actions.extend(a.strip() for a in actions)
        self.chunk_id += 1
        self.chunk_length = len(actions)
        self.actions_in_chunk = 0

    def next_action(self) -> str:
        if self.needs_plan:
            raise RuntimeError("a model-generated chunk is required")
        return self.pending_actions.popleft()

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
            interruption_count=self.anomaly_replan_count,
            chunk_exhausted=not self.pending_actions,
            step_completed=step_completed,
            dynamic=self.dynamic,
        )
        active_step_id = self.active_step.step_id
        discarded: tuple[str, ...] = ()

        if decision.kind is DecisionType.ADVANCE:
            self.boundary_count += 1
            self.last_replan_reason = ""
        elif decision.kind is DecisionType.REPLAN:
            discarded = tuple(self.pending_actions)
            self.pending_actions.clear()
            self.interruption_count += int(bool(discarded))
            self.anomaly_replan_count += 1
            self.last_replan_reason = decision.reason
            self.plan = self.inducer.replan(
                self.plan,  # type: ignore[arg-type]
                self.active_index,
                decision.reason,
                observation,
            )
            self.active_index = 0
        elif decision.kind in {DecisionType.STOP_SUCCESS, DecisionType.STOP_FAILURE}:
            discarded = tuple(self.pending_actions)
            self.pending_actions.clear()

        self.action_history.append(action)
        self.observation_history.append(observation)
        return ExecutionEvent(
            decision,
            check,
            active_step_id,
            self.plan.revision,  # type: ignore[union-attr]
            self.interruption_count,
            discarded,
        )
