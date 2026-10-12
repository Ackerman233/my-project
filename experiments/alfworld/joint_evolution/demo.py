"""Offline integration fixture; no model, no benchmark gain claim."""
from dataclasses import asdict
import json
import re
from pathlib import Path

from experiments.common.puzzle_pilot import Settings, save
from .schema import Genome
from .runtime import evaluate_episode
from .engine import evolve


class PublicFixture:
    tasks = ['Reach desk 1.']
    def __init__(self):
        self.envs = self

    @property
    def get_admissible_commands(self):
        return [['go to desk 1']]

    def reset(self, _):
        return {'anchor': ['There is a surface named drawer 1 and a desk 1.']}, [{}]

    def step(self, wrapped):
        action = re.search(r'<action>(.*?)</action>', wrapped[0])[1]
        done = action == 'go to desk 1'
        return {'anchor': ['You arrive at desk 1.' if done else 'Nothing happens.']}, [float(done)], [done], [{'won': done}]


def splits(generations=2):
    def task(name):
        return {'id': name, 'scene_id': name, 'gamefile': 'offline-fixture://' + name, 'benchmark': 'alfworld'}
    return {'train': [task('train-public')], 'selection': [task('select-private')],
            'confirmation': [[task(f'confirm-private-{i}')] for i in range(generations)],
            'test': [task('test-private')]}


def scripted_evaluator(tasks, genome, directory):
    rows = []
    for task in tasks:
        class Actor:
            count = 0
            def before_call(self):
                pass
            def __call__(self, prompt):
                self.count += 1
                actions = ['open drawer 1', 'go to desk 1'] if self.count == 1 else ['go to desk 1']
                return {'response': '<thought>Navigate.</thought><actions>' + json.dumps(actions) + '</actions>',
                        'usage': {'prompt_tokens': 0, 'completion_tokens': 0}}
        row = evaluate_episode(PublicFixture(), task, genome,
            Settings(model='offline-scripted-fixture', max_steps=4, max_calls=3), Actor(), directory)
        rows.append(row)
    return rows


def scripted_proposer(prompt, directory):
    assert 'select-private' not in prompt and 'confirm-private' not in prompt and 'test-private' not in prompt
    data = json.loads(prompt.split('\nCURRENT\n', 1)[1].split('\nTRAINING_ONLY\n', 1)[0])
    evidence = json.loads(prompt.split('\nTRAINING_ONLY\n', 1)[1])
    policy = data['policy']
    if policy['invalid_patience'] == 1:
        policy['invalid_patience'] = 2
    else:
        policy['history_evidence'] = not policy['history_evidence']
    proposal = {'parent_hash': data['parent_hash'],
        'edits': [{'op': 'append', 'target': '', 'content': 'Verify rejected actions against the dependency of the remaining navigation.'}],
        'policy': policy, 'evidence_ids': [evidence[0]['id']]}
    return {'response': json.dumps(proposal), 'cost_usd': 0, 'fixture_not_model': True}


def run_demo(output):
    initial = Genome('Use public observations to navigate. Never assume a surface opens.')
    best, result = evolve(splits(), initial, scripted_evaluator, scripted_proposer,
                          output, generations=2, candidates_per_generation=1)
    save(Path(output) / 'DEMO_ONLY.json', {'offline_scripted_fixture': True, 'api_calls': 0,
        'not_benchmark_gain': True, 'best_revision': best.revision,
        'what_is_verified': 'Actual queue truncation changes from one to zero after policy edit; fresh confirmation accepts first round, no-gain second round is rejected. Final test never enters proposal prompts.'})
    return best, result
