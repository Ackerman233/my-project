import json
import re
from dataclasses import asdict
from experiments.puzzle_pilot import Settings, save
from experiments.interrupt_recovery_ablation import config_hash, digest
from experiments.queue_revision_transfer_trial import replay_prefix, run_suffix


class PublicEnv:
    def __init__(self):
        self.envs = self
        self.get_admissible_commands = [['go to drawer 1', 'go to shelf 1']]
        self.actions = []
    def reset(self, _):
        self.get_admissible_commands = [['go to drawer 1', 'go to shelf 1']]
        return {'anchor': ['Room with drawer 1 and shelf 1.']}, [{}]
    def step(self, actions):
        action = re.search(r'<action>(.*?)</action>', actions[0])[1]
        self.actions.append(action)
        valid = action in self.get_admissible_commands[0]
        if action == 'go to drawer 1':
            feedback = 'On the drawer 1, you see nothing.'
            self.get_admissible_commands = [['go to shelf 1', 'examine drawer 1']]
        elif valid:
            feedback = 'On the shelf 1, you see nothing.'
            self.get_admissible_commands = [['go to drawer 1', 'examine shelf 1']]
        else:
            feedback = 'Nothing happens.'
        return {'anchor': [feedback]}, [0], [False], [{}]


def fixture(tmp_path, tail):
    (tmp_path / 'game').write_text('fixture only')
    save(tmp_path / 'initial.json', {'observation': 'Room with drawer 1 and shelf 1.',
        'task': 'put box on shelf.', 'gamefile': str(tmp_path / 'game')})
    path = tmp_path / 'conversation.json'
    save(path, [{'step': 0, 'action': 'go to drawer 1',
        'admissible_before': ['go to drawer 1', 'go to shelf 1'],
        'env_feedback': 'On the drawer 1, you see nothing.', 'done': False,
        'remaining_before_interrupt': tail}])
    case = {'id': 'fixture', 'source_trace': str(path), 'source_sha256': digest(path),
        'source_initial_sha256': digest(tmp_path / 'initial.json'),
        'game_sha256': digest(tmp_path / 'game'), 'step': 0, 'tail': tail,
        'reason': 'observed contradiction', 'kind': 'falsified_head'}
    _, _, checkpoint = replay_prefix(PublicEnv(), case)
    case['checkpoint_sha256'] = config_hash(checkpoint)
    cfg = {'policy': {'max_actions': 4, 'trigger': 'anomaly', 'invalid_patience': 1, 'recovery_steps': 0},
        'task_skill': 'use public commands', 'settings': asdict(Settings(max_calls=0, max_steps=4))}
    return case, cfg


def test_three_arms_same_checkpoint_original_queue_only_differs(tmp_path):
    case, cfg = fixture(tmp_path, ['open drawer 1', 'go to shelf 1'])
    rows = [run_suffix(PublicEnv(), case, arm, cfg, None, tmp_path / arm)
        for arm in ('continue_tail', 'interrupt_replan', 'evidence_repair')]
    assert len({r['checkpoint_sha256'] for r in rows}) == 1
    assert [r['model_calls'] for r in rows] == [0, 0, 0]
    assert [r['suffix_steps'] for r in rows] == [2, 0, 1]
    assert [r['invalid_actions'] for r in rows] == [1, 0, 0]
    assert rows[2]['skipped_falsified_heads'] == 1
    assert rows[2]['original_tail_executed'] == 1


def test_dependency_rejection_requires_replan_not_tail_execution(tmp_path):
    case, cfg = fixture(tmp_path, ['open drawer 1', 'take box 1 from drawer 1'])
    env = PublicEnv()
    row = run_suffix(env, case, 'evidence_repair', cfg, None, tmp_path / 'repair')
    assert row['suffix_steps'] == 0
    assert row['revision_replan_discarded'] == 2
    assert env.actions == ['go to drawer 1']


def test_repair_rechecks_after_navigation_and_does_not_certify_far_tail(tmp_path):
    case, cfg = fixture(tmp_path, ['open drawer 1', 'go to shelf 1', 'take box 1 from drawer 1'])
    row = run_suffix(PublicEnv(), case, 'evidence_repair', cfg, None, tmp_path / 'repair')
    assert row['suffix_steps'] == 1
    decisions = json.loads((tmp_path / 'repair' / 'queue_revisions.json').read_text())
    assert [r['revision']['decision'] for r in decisions] == ['skip_head', 'replan']
