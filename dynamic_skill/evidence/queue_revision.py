"""Domain-independent, one-head revision from fresh evidence and dependency proofs.

This certifies only the next action's local contract. Future retained actions stay
pending and MUST be reassessed after each execution; no task-success guarantee.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class Truth(str, Enum):
    TRUE = 'true'
    FALSE = 'false'
    UNKNOWN = 'unknown'


@dataclass(frozen=True)
class Proof:
    value: Truth
    snapshot_id: str
    source: str
    authoritative: bool
    reason: str


class EvidenceAdapter(Protocol):
    snapshot_id: str
    def executable(self, action: str) -> Proof: ...
    def independent_after_skip(self, skipped: str, following: str) -> Proof: ...


@dataclass(frozen=True)
class Revision:
    decision: str
    skipped: tuple[str, ...]
    pending: tuple[str, ...]
    next_action: str | None
    reason: str
    proofs: tuple[Proof, ...]


def verified(proof, adapter):
    if not proof.authoritative or not proof.source or proof.snapshot_id != adapter.snapshot_id:
        return Truth.UNKNOWN
    return proof.value


def revise_head(actions, adapter):
    """Skip at most ONE unexecuted head; stop on unknown/failed dependency evidence."""
    actions = tuple(actions)
    if not actions:
        return Revision('replan', (), (), None, 'empty queue', ())
    head = adapter.executable(actions[0])
    verdict = verified(head, adapter)
    if verdict is Truth.TRUE:
        return Revision('execute', (), actions, actions[0], 'current head supported', (head,))
    if verdict is Truth.UNKNOWN:
        return Revision('replan', (), actions, None, 'head not proven executable or falsified', (head,))
    if len(actions) == 1:
        return Revision('replan', (actions[0],), (), None, 'falsified head; no remaining action', (head,))
    dependency = adapter.independent_after_skip(actions[0], actions[1])
    next_proof = adapter.executable(actions[1])
    proofs = (head, dependency, next_proof)
    if verified(dependency, adapter) is not Truth.TRUE:
        return Revision('replan', (actions[0],), actions[1:], None,
                        'following contract depends on missing effect or dependency is unknown', proofs)
    if verified(next_proof, adapter) is not Truth.TRUE:
        return Revision('replan', (actions[0],), actions[1:], None,
                        'following head not freshly proven executable', proofs)
    return Revision('skip_head', (actions[0],), actions[1:], actions[1],
                    'falsified head and independently supported next action', proofs)
