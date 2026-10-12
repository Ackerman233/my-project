"""Opt-in, puzzle-only counterfactual filter. Not connected to online runners.

Uses exact local dynamics to validate only the supplied actions, never search.
Legality is NOT a guarantee of strategic usefulness or Sudoku solvability.
"""
from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter

from dynamic_skill.environments.puzzles import RushHourEnv, SudokuEnv
from dynamic_skill.execution.interrupter import Interrupter, Decision, DecisionType
from dynamic_skill.execution.checker import CheckStatus


class TailFilterInterrupter(Interrupter):
    """One-observation override, armed only after external tail validation."""
    validated_once = False

    def decide(self, check, **kwargs):
        validated = self.validated_once
        self.validated_once = False
        if (validated and kwargs['dynamic'] and not kwargs['chunk_exhausted']
                and check.status is CheckStatus.INVALID_ACTION):
            return Decision(DecisionType.CONTINUE, 'continue independently validated filtered tail')
        return super().decide(check, **kwargs)


@dataclass(frozen=True)
class TailFilterResult:
    decision: str
    reason: str
    retained: tuple[str, ...]
    retained_indices: tuple[int, ...]
    skipped: tuple[tuple[int, str], ...]
    checks: int
    seconds: float


def filter_tail(env, actions, *, failed_valid, before_failed, max_checks=4, recent_states=()):
    started = perf_counter()
    retained, indices, skipped = [], [], []
    checks = 0

    def finish(reason, keep=False):
        return TailFilterResult('continue' if keep and retained else 'replan', reason,
                                tuple(retained) if keep else (), tuple(indices) if keep else (),
                                tuple(skipped), checks, perf_counter() - started)

    # Exact supported classes: arbitrary render() implementations may omit state.
    if type(env) not in (RushHourEnv, SudokuEnv):
        return finish('unsupported_simulator')
    if max_checks < 1 or len(actions) > max_checks:
        return finish('scan_budget')
    original = env.render()
    if failed_valid is not False or before_failed != original or env.solved:
        return finish('failure_not_confirmed_noop')
    probe = deepcopy(env)
    seen = set(recent_states) | {original}
    for index, action in enumerate(actions):
        before = probe.render()
        checks += 1
        result = probe.step(action)
        after = probe.render()
        if not result.valid:
            if after != before:
                return finish('invalid_action_side_effect')
            skipped.append((index, 'illegal_noop'))
            continue
        if after in seen:
            return finish('cycle_replan')
        retained.append(action)
        indices.append(index)
        seen.add(after)
        if result.won:
            break
    assert env.render() == original, 'simulation must not mutate live state'
    return finish('validated_tail' if retained else 'no_legal_tail', keep=bool(retained))
