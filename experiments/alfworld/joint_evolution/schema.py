"""Bounded joint genome and real upstream SkillOpt text patch application."""
from dataclasses import asdict, dataclass, field
import hashlib
import json
import re

from dynamic_skill.controllers.control_only_evolution import ControlPolicy


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class InterruptPolicy:
    max_actions: int = 4
    trigger: str = 'anomaly'
    invalid_patience: int = 1
    recovery_steps: int = 0
    state_check: bool = True
    discovery_interrupt: bool = False
    history_evidence: bool = True

    def __post_init__(self):
        self.executable()
        for name in ('state_check', 'discovery_interrupt', 'history_evidence'):
            if type(getattr(self, name)) is not bool:
                raise ValueError('Boolean policy field required: ' + name)

    def executable(self):
        return ControlPolicy(max_actions=self.max_actions, trigger=self.trigger,
                             invalid_patience=self.invalid_patience, recovery_steps=self.recovery_steps)


@dataclass(frozen=True)
class Genome:
    skill_text: str
    policy: InterruptPolicy = field(default_factory=InterruptPolicy)
    revision: int = 0
    parent_hash: str = ''
    evidence_ids: tuple = ()

    def __post_init__(self):
        if not isinstance(self.skill_text, str) or not self.skill_text.strip() or len(self.skill_text) > 16000:
            raise ValueError('Skill text must be nonempty and at most 16000 characters')
        if not isinstance(self.policy, InterruptPolicy):
            raise ValueError('Executable InterruptPolicy required')
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError('Invalid revision')

    @property
    def hash(self):
        return fingerprint(asdict(self))

    @classmethod
    def from_dict(cls, data):
        return cls(**{**data, 'policy': InterruptPolicy(**data.get('policy', {})),
                      'evidence_ids': tuple(data.get('evidence_ids', []))})


def apply_proposal(raw, parent, evidence):
    """Generated text is data, never Python. Terminal/budget logic is not editable."""
    from skillopt.optimizer.skill import apply_patch
    raw = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(raw, dict) or set(raw) != {'parent_hash', 'edits', 'policy', 'evidence_ids'}:
        raise ValueError('Joint proposal schema mismatch')
    if raw['parent_hash'] != parent.hash:
        raise ValueError('Stale parent hash')
    known = {e['id'] for e in evidence}
    cited = raw['evidence_ids']
    if not isinstance(cited, list) or not cited or any(not isinstance(x, str) or x not in known for x in cited):
        raise ValueError('Cite supplied training evidence only')
    if not isinstance(raw['policy'], dict) or set(raw['policy']) != set(asdict(parent.policy)):
        raise ValueError('Policy schema mismatch')
    policy = InterruptPolicy(**raw['policy'])
    edits = raw['edits']
    if not isinstance(edits, list) or not 1 <= len(edits) <= 4:
        raise ValueError('One to four SkillOpt edits required')
    text = parent.skill_text
    for edit in edits:
        if not isinstance(edit, dict) or set(edit) != {'op', 'target', 'content'}:
            raise ValueError('Edit requires op, target and content only')
        if any(not isinstance(v, str) for v in edit.values()):
            raise ValueError('Edit fields must be strings')
        if edit['op'] not in ('append', 'replace', 'delete') or len(edit['content']) > 4000:
            raise ValueError('Unsupported/unbounded edit')
        if edit['op'] == 'append' and edit['target']:
            raise ValueError('Append must have empty target')
        if edit['op'] != 'append' and (not edit['target'] or text.count(edit['target']) != 1):
            raise ValueError('Edit target must match exactly once')
        updated = apply_patch(text, {'edits': [edit]})
        if updated == text:
            raise ValueError('SkillOpt skipped edit (possibly protected region)')
        text = updated
    if text == parent.skill_text or policy == parent.policy:
        raise ValueError('Joint candidate must change both skill text and executable policy')
    candidate = Genome(text, policy, parent.revision + 1, parent.hash, tuple(cited))
    return candidate


def validate_splits(splits, generations):
    if set(splits) != {'train', 'selection', 'confirmation', 'test'}:
        raise ValueError('Require train/selection/confirmation/test splits')
    if not isinstance(splits['confirmation'], list) or len(splits['confirmation']) != generations:
        raise ValueError('Predeclare one fresh confirmation fold per generation')
    groups = [splits['train'], splits['selection'], *splits['confirmation'], splits['test']]
    ids, identities = set(), set()
    for tasks in groups:
        if not isinstance(tasks, list) or not tasks:
            raise ValueError('Every split/fold must be nonempty')
        for task in tasks:
            name = task.get('id', '')
            if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', name) or name in ids:
                raise ValueError('Unsafe or duplicate task ID')
            # Runtime validates scene_id against the actual ALFWorld game path.
            scene = task.get('scene_id')
            if not isinstance(scene, str) or not scene or scene in identities:
                raise ValueError('Missing or duplicate physical scene across splits/folds')
            ids.add(name)
            identities.add(scene)
