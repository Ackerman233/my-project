import json
import re
from dataclasses import asdict
import pytest
from experiments.puzzle_pilot import Settings, save
from experiments.interrupt_recovery_ablation import digest, config_hash
from experiments.prefix_recovery_trial import replay_prefix, run_suffix


class PublicEnv:
    tasks = ['put some box on dresser.']
    def __init__(self):
        self.envs = self
        self.actions = []
        self.get_admissible_commands = [['go to drawer 1', 'go to drawer 2', 'look']]
    def reset(self, _):
        return {'anchor': ['Room has drawer 1, drawer 2.']}, [{}]
    def step(self, actions):
        action = re.search(r'<action>(.*?)</action>', actions[0])[1]
        self.actions.append(action)
        self.get_admissible_commands = [['examine drawer 1', 'go to drawer 2', 'look']]
        feedback = 'Nothing happens.' if action == 'open drawer 1' else 'On the drawer 1, you see nothing.'
        if action == 'go to drawer 1':
            feedback = 'You arrive at drawer 1. On the drawer 1, you see nothing.'
        return {'anchor': [feedback]}, [0], [False], [{}]


def case_fixture(tmp_path):
    (tmp_path / 'game').write_text('synthetic fixture, not a benchmark solution')
    save(tmp_path / 'initial.json', {'observation': 'Room has drawer 1, drawer 2.',
         'task': PublicEnv.tasks[0], 'gamefile': str(tmp_path / 'game')})
    save(tmp_path / 'conversation.json', [{
        'step': 0, 'action': 'go to drawer 1', 'admissible_before': ['go to drawer 1', 'go to drawer 2', 'look'],
        'env_feedback': 'You arrive at drawer 1. On the drawer 1, you see nothing.', 'done': False,
        'remaining_before_interrupt': ['open drawer 1'], 'valid': True}])
    case = {'id': 'fixture', 'source_trace': str(tmp_path / 'conversation.json'),
            'source_sha256': digest(tmp_path / 'conversation.json'), 'step': 0,
            'tail': ['open drawer 1'], 'kind': 'surface_open_contradiction', 'reason': 'surface open contradiction'}
    _, _, checkpoint = replay_prefix(PublicEnv(), case)
    case['checkpoint_sha256'] = config_hash(checkpoint)
    return case


def test_identical_prefix_different_tail_and_suffix_accounting(tmp_path):
    case = case_fixture(tmp_path)
    cfg = {'policy': {'max_actions': 4, 'trigger': 'anomaly', 'invalid_patience': 1, 'recovery_steps': 0},
           'task_skill': 'Use public actions.', 'settings': asdict(Settings(max_calls=1, max_steps=4))}
    class Planner:
        def before_call(self): pass
        def __call__(self, prompt):
            return {'response': '<thought>Observe.</thought><actions>["look"]</actions>',
                    'usage': {'prompt_tokens': 0, 'completion_tokens': 0}}
    rows = [run_suffix(PublicEnv(), case, arm, cfg, Planner(), tmp_path / arm)
            for arm in ('continue_tail', 'interrupt_replan')]
    assert rows[0]['checkpoint_sha256'] == rows[1]['checkpoint_sha256']
    assert rows[0]['prefix_steps'] == rows[1]['prefix_steps'] == 1
    assert rows[0]['model_calls'] == rows[1]['model_calls'] == 1
    assert [r['suffix_steps'] for r in rows] == [2, 1]
    assert [r['invalid_actions'] for r in rows] == [1, 0]
    assert rows[0]['original_tail_executed'] == 1
    assert rows[1]['initial_discarded_actions'] == ['open drawer 1']


def test_changed_prefix_refused_before_planning(tmp_path):
    case = case_fixture(tmp_path)
    trace = json.loads((tmp_path / 'conversation.json').read_text())
    trace[0]['env_feedback'] = 'different state'
    save(tmp_path / 'conversation.json', trace)
    with pytest.raises(AssertionError, match='source trace changed'):
        replay_prefix(PublicEnv(), case)
