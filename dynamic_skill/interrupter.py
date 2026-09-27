from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .checker import CheckResult, CheckStatus


class DecisionType(str, Enum):
    CONTINUE = "continue"
    ADVANCE = "advance_chunk"
    REPLAN = "interrupt_and_replan"
    STOP_SUCCESS = "stop_success"
    STOP_FAILURE = "stop_failure"


@dataclass(frozen=True)
class Decision:
    kind: DecisionType
    reason: str


class Interrupter:
    def __init__(self, max_interrupts: int = 8) -> None:
        self.max_interrupts = max(1, max_interrupts)

    def decide(
        self,
        check: CheckResult,
        *,
        interruption_count: int,
        chunk_exhausted: bool,
        step_completed: bool,
        dynamic: bool,
    ) -> Decision:
        if check.status is CheckStatus.SUCCESS:
            return Decision(DecisionType.STOP_SUCCESS, check.reason)
        if check.status is CheckStatus.ENV_FAILURE:
            return Decision(DecisionType.STOP_FAILURE, check.reason)
        if dynamic and check.should_interrupt:
            if interruption_count >= self.max_interrupts:
                return Decision(DecisionType.STOP_FAILURE, "interrupt budget exhausted")
            return Decision(DecisionType.REPLAN, check.reason)
        if chunk_exhausted:
            return Decision(DecisionType.ADVANCE, "action queue exhausted normally")
        return Decision(DecisionType.CONTINUE, check.reason)
