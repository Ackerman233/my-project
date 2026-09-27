from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any, Sequence


class CheckStatus(str, Enum):
    OK = "ok"
    PROGRESS = "progress"
    SUCCESS = "success"
    INVALID_ACTION = "invalid_action"
    ACTION_LOOP = "action_loop"
    STALLED = "stalled"
    ENV_FAILURE = "env_failure"


@dataclass(frozen=True)
class CheckResult:
    status: CheckStatus
    reason: str
    should_interrupt: bool = False
    signals: tuple[str, ...] = ()


class ObservationChecker:
    """Lightweight checker using only action/observation trajectory signals."""

    invalid_patterns = (
        "not a valid action",
        "nothing happens",
        "can't do that",
        "cannot do that",
        "you cannot",
        "not possible",
    )
    progress_patterns = (
        "you pick up",
        "you take",
        "you open",
        "you put",
        "you place",
        "you clean",
        "you heat",
        "you cool",
        "you turn on",
    )

    def __init__(self, repeated_observation_limit: int = 2) -> None:
        self.repeated_observation_limit = max(1, repeated_observation_limit)

    @staticmethod
    def normalize(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip().lower())

    def check(
        self,
        action: str,
        observation: str,
        action_history: Sequence[str] = (),
        observation_history: Sequence[str] = (),
        reward: float = 0.0,
        done: bool = False,
        info: dict[str, Any] | None = None,
    ) -> CheckResult:
        info = info or {}
        obs = self.normalize(observation)
        act = self.normalize(action)
        if done and (reward > 0 or bool(info.get("won"))):
            return CheckResult(CheckStatus.SUCCESS, "environment reported success", signals=("terminal_success",))
        if done:
            return CheckResult(CheckStatus.ENV_FAILURE, "episode ended without success", True, ("terminal_failure",))
        invalid = tuple(pattern for pattern in self.invalid_patterns if pattern in obs)
        if invalid:
            return CheckResult(CheckStatus.INVALID_ACTION, f"environment rejected action: {invalid[0]}", True, invalid)

        actions = [self.normalize(item) for item in action_history]
        if len(actions) >= 2 and act and actions[-1] == act and actions[-2] == act:
            return CheckResult(CheckStatus.ACTION_LOOP, "same action selected three times", True, ("aaa_loop",))
        if len(actions) >= 3 and act and actions[-2] == act and actions[-1] == actions[-3] != act:
            return CheckResult(CheckStatus.ACTION_LOOP, "alternating two-action loop detected", True, ("abab_loop",))

        observations = [self.normalize(item) for item in observation_history]
        recent = observations[-self.repeated_observation_limit :]
        if obs and len(recent) == self.repeated_observation_limit and all(item == obs for item in recent):
            return CheckResult(CheckStatus.STALLED, "observation did not change", True, ("repeated_observation",))

        progress = tuple(pattern for pattern in self.progress_patterns if pattern in obs)
        if progress or reward > 0:
            return CheckResult(CheckStatus.PROGRESS, "progress signal observed", signals=progress or ("positive_reward",))
        return CheckResult(CheckStatus.OK, "no anomaly detected")
