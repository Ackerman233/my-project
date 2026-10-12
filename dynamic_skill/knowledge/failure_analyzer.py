from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class FailureReport:
    category: str
    evidence: tuple[str, ...]
    recommendation: str


class FailureAnalyzer:
    def analyze(self, trajectory: Iterable[dict]) -> FailureReport:
        rows = list(trajectory)
        statuses = [str(row.get("checker_status", "")) for row in rows]
        counts = Counter(statuses)
        if counts["action_loop"]:
            return FailureReport("loop", (f"action_loop={counts['action_loop']}",), "Track visited transitions and force a novel admissible action.")
        if counts["invalid_action"]:
            return FailureReport("invalid_action", (f"invalid_action={counts['invalid_action']}",), "Ground every action in the current admissible-action list.")
        if counts["stalled"]:
            return FailureReport("stalled", (f"stalled={counts['stalled']}",), "Interrupt earlier when observations repeat.")
        if rows and bool(rows[-1].get("done")) and not bool(rows[-1].get("reward")):
            return FailureReport("terminal_failure", ("episode ended without reward",), "Verify preconditions before committing terminal actions.")
        return FailureReport("timeout_or_unknown", (f"steps={len(rows)}",), "Reduce unproductive exploration and preserve explicit subgoal state.")

