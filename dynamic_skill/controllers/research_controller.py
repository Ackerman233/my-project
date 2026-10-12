"""Observable-feedback controller policies. No environment/solver access."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Policy:
    name: str
    chunk: int = 4
    interrupt: bool = True
    recovery: int = 0
    memory: bool = False
    patience: int = 1


POLICIES = {
    'react': Policy('react', chunk=1, interrupt=False),
    'fixed': Policy('fixed', interrupt=False),
    'interrupt': Policy('interrupt'),
    'adaptive1': Policy('adaptive1', recovery=1),
    'adaptive3': Policy('adaptive3', recovery=3),
    'memory': Policy('memory', memory=True),
    'memory_adaptive1': Policy('memory_adaptive1', recovery=1, memory=True),
    'memory_adaptive3': Policy('memory_adaptive3', recovery=3, memory=True),
    'patient_memory': Policy('patient_memory', recovery=1, memory=True, patience=2),
}


class Controller:
    def __init__(self, policy):
        self.policy = policy
        self.recovery_left = 0
        self.invalid_streak = 0
        self.failed = {}

    def action_limit(self):
        return 1 if self.recovery_left else self.policy.chunk

    def memory(self, state):
        return sorted(self.failed.get(state, set())) if self.policy.memory else []

    def observe(self, before, action, valid, after, tail):
        if not valid:
            self.failed.setdefault(before, set()).add(action)
            self.invalid_streak += 1
            self.recovery_left = self.policy.recovery
        else:
            self.invalid_streak = 0
            self.recovery_left = max(0, self.recovery_left - 1)
        cut = bool(tail and self.policy.interrupt and not valid
                   and self.invalid_streak >= self.policy.patience)
        return {'interrupt': cut, 'discarded': list(tail) if cut else [],
                'recovery_left': self.recovery_left,
                'invalid_streak': self.invalid_streak,
                'reason': 'observed_invalid_threshold' if cut else 'continue'}
