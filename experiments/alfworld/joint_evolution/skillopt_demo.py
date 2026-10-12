"""Zero-API integration fixture exercising the actual upstream core chain.

Actor/optimizer replies are scripted; upstream analysis, merge, ranking, patch
application, executable interruption, selection and confirmation are real code.
This verifies integration, not learned policy quality or benchmark performance.
"""
from dataclasses import replace
import json
from pathlib import Path

from experiments.common.puzzle_pilot import Settings, save
from .demo import PublicFixture, splits
from .engine import evolve
from .runtime import evaluate_episode
from .schema import Genome
from .skillopt_core import SkillOptCore, parse_joint, policy_json


def scripted_evaluator(tasks, genome, directory):
    rows = []
    for task in tasks:
        class Actor:
            def before_call(self):
                pass

            def __call__(self, prompt):
                actions = ['go to desk 1'] if task['id'] == 'train-success' else ['open drawer 1', 'go to desk 1']
                return {'response': '<thought>Navigate.</thought><actions>' + json.dumps(actions) + '</actions>',
                        'usage': {'prompt_tokens': 0, 'completion_tokens': 0}}
        # One planning call makes discarding a still-usable navigation tail a
        # real observed task failure in the first round of this fixture.
        rows.append(evaluate_episode(PublicFixture(), task, genome,
            Settings(model='offline-scripted-fixture', max_steps=4, max_calls=1), Actor(), directory))
    return rows


class ScriptedOptimizer:
    def __init__(self):
        self.calls = []

    def __call__(self, prompt, *, system, max_tokens, label):
        assert all(name not in system + prompt for name in ('select-private', 'confirm-private', 'test-private'))
        phase = Path(label).parent.name
        self.calls.append(phase)
        document = prompt.split('## Current Skill\n', 1)[1].split('\n\n## ', 1)[0]
        _, policy = parse_joint(document)
        known = json.loads(system.rsplit('Supplied evidence IDs: ', 1)[1])

        def skill_edit(content):
            return {'op': 'append', 'content': content, 'evidence_ids': known}

        def policy_edit(new_policy):
            return {'op': 'replace', 'target': policy_json(policy), 'content': policy_json(new_policy),
                    'evidence_ids': known}

        if phase == 'reflect_failure':
            payload = {'failure_summary': [{'failure_type': 'premature_stop', 'count': 1,
                'description': 'An invalid surface action discarded navigation before a one-call limit.'}],
                'patch': {'reasoning': 'Test whether a later navigation action can survive one rejection.',
                    'edits': [policy_edit(replace(policy, invalid_patience=2)),
                              skill_edit('Use rejection feedback to reconsider assumptions and remaining navigation.')]}}
        elif phase == 'reflect_success':
            if policy.invalid_patience == 1:
                edits = [skill_edit(text) for text in (
                    'Navigate using public admissible commands.', 'Keep the destination subgoal explicit.',
                    'Stop acting when the environment reports success.')]
            else:
                edits = [policy_edit(replace(policy, history_evidence=not policy.history_evidence)),
                         skill_edit('Recheck the public task after each observation.')]
            payload = {'patch': {'reasoning': 'Retain reusable behavior seen in successful training runs.', 'edits': edits}}
        elif phase == 'aggregate':
            patches = json.loads('[' + prompt.split('\n\n[', 1)[1])
            payload = {'reasoning': 'Preserve failure repair and complementary success guidance.',
                       'edits': [edit for patch in patches for edit in patch['edits']]}
        elif phase == 'rank':
            payload = {'reasoning': 'Keep policy repair and complementary skill changes.', 'selected_indices': [0, 1, 2, 3]}
        else:
            raise AssertionError('Unexpected optimizer stage: ' + phase)
        return {'response': json.dumps(payload), 'usage': {'prompt_tokens': 0, 'completion_tokens': 0},
                'cost_usd': 0, 'fixture_not_model': True}


def run_demo(output):
    data = splits()
    data['train'].append({'id': 'train-success', 'scene_id': 'train-success',
                          'gamefile': 'offline-fixture://train-success', 'benchmark': 'alfworld'})
    optimizer = ScriptedOptimizer()
    initial = Genome('Use public observations to navigate. Never assume a surface opens.')
    best, final = evolve(data, initial, scripted_evaluator, None, output,
        generations=2, candidates_per_generation=1, candidate_builder=SkillOptCore(optimizer))
    save(Path(output) / 'DEMO_ONLY.json', {'offline_scripted_fixture': True, 'api_calls': 0,
        'not_benchmark_gain': True, 'best_revision': best.revision,
        'actual_upstream_core_used': True, 'scripted_optimizer_stages': optimizer.calls,
        'what_is_verified': 'Actual training failure feeds upstream Reflect, merge and rank retain an executable policy edit, Apply changes queue behavior; fresh confirmation accepts, a later no-gain candidate is rejected.'})
    return best, final
