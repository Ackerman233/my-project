from dataclasses import asdict, replace
import json
import pytest

from dynamic_skill.evolution import (EvolvedSkill, LearnedInterrupter, validate_splits,
    parse_candidate, proposal_prompt, validation_gate, evolve, skillopt_document)
from dynamic_skill.checker import CheckResult, CheckStatus
from dynamic_skill.interrupter import DecisionType
from experiments.puzzle_pilot import run_episode, Settings


def task(name, board=None):
    return {'id': name, 'benchmark': 'rush_hour', 'minimum_actions': 1,
            'board': board or '......' '......' 'AA....' '......' '......' '......'}


def splits():
    return {'train': [task('train_good'), task('train_bad', 'x.....' '......' '.AA...' '......' '......' '......')],
            'validation': [task('validation', '......' 'x.....' 'AA....' '......' '......' '......')],
            'test': [task('test', '......' '......' 'AA....' '......' '......' 'x.....')]}


def evidence():
    return [{'id': 'good', 'success': True}, {'id': 'bad', 'success': False}]


def proposal():
    return {'chunk_guidance': 'Verify clear path before grouping slides.',
            'recovery_guidance': 'Recompute from current coordinates after failed movement.',
            'max_actions': 2, 'interrupt_after': {'invalid_action': 2, 'action_loop': 1, 'stalled': 1},
            'evidence_ids': ['good', 'bad']}


def test_split_rejects_renamed_duplicate_and_path_traversal():
    data = splits(); validate_splits(data)
    data['test'][0]['board'] = data['train'][0]['board']
    with pytest.raises(ValueError): validate_splits(data)
    data = splits(); data['test'][0]['id'] = '../escape'
    with pytest.raises(ValueError): validate_splits(data)


@pytest.mark.parametrize('change', [ {'max_actions': 99}, {'evidence_ids': ['test-answer']},
    {'evidence_ids': ['good']}, {'interrupt_after': {'execute_python': 1}}, {'max_actions': True}])
def test_candidate_rejects_unbounded_or_unproven_updates(change):
    with pytest.raises((ValueError, TypeError)):
        parse_candidate(json.dumps({**proposal(), **change}), EvolvedSkill(), evidence())


def test_candidate_provenance_and_upstream_patch():
    parent = EvolvedSkill()
    child = parse_candidate(json.dumps(proposal()), parent, evidence())
    assert child.parent_hash == parent.hash and child.revision == 1
    assert 'Chunk formation' in skillopt_document(child)
    assert 'TRAIN_EVIDENCE' in proposal_prompt(parent, evidence())


def test_interrupt_threshold_is_executable():
    policy = LearnedInterrupter(replace(EvolvedSkill(), interrupt_after={'invalid_action': 2, 'action_loop': 1, 'stalled': 1}))
    args = dict(interruption_count=0, chunk_exhausted=False, step_completed=False, dynamic=True)
    check = CheckResult(CheckStatus.INVALID_ACTION, 'invalid', True)
    assert policy.decide(check, **args).kind is DecisionType.CONTINUE
    assert policy.decide(check, **args).kind is DecisionType.REPLAN
    assert policy.decide(CheckResult(CheckStatus.SUCCESS, 'won'), **args).kind is DecisionType.STOP_SUCCESS


def test_skill_changes_real_queue_and_prompt(tmp_path):
    skill = replace(EvolvedSkill(), interrupt_after={'invalid_action': 2, 'action_loop': 1, 'stalled': 1})
    prompts = []
    def actor(prompt):
        prompts.append(prompt)
        return {'response': '<actions>["A-1", "A+4"]</actions>', 'usage': {}}
    result = run_episode(task('queue'), 'dynamic_interrupt', Settings(action_only=True), tmp_path, actor, skill=skill)
    assert result['success'] and result['model_calls'] == 1 and result['environment_steps'] == 2
    assert skill.chunk_guidance in prompts[0] and result['skill_hash'] == skill.hash


def row(success=True, calls=3, **kwargs):
    return dict(task_id='a', success=success, model_calls=calls, invalid_actions=0,
                environment_steps=2, status='success' if success else 'step_limit', **kwargs)


def test_gate_rejects_cheap_failure_infrastructure_and_mismatches():
    assert not validation_gate([row()], [row(False, 1)])['accept']
    assert not validation_gate([row(False)], [row(False, 1)])['accept']
    assert not validation_gate([row()], [{**row(), 'status': 'api_error'}])['accept']
    assert not validation_gate([row()], [])['accept']
    assert validation_gate([row(calls=3)], [row(calls=2)])['accept']
    assert validation_gate([row(False)], [row()])['accept']


def demo_run(output, improve=True):
    """Scripted interface fixture, NOT a model benchmark or learned capability."""
    seen = []
    def evaluator(tasks, skill, mode, directory):
        rows = []
        for item in tasks:
            seen.append((item['id'], skill.revision if skill else None, mode))
            def actor(prompt):
                # Deliberately scripted to exercise accept/reject plumbing.
                action = 'A+3' if item['id'] == 'train_bad' else 'A+4'
                if skill is not None and (skill.revision == 0 or not improve) and item['id'] != 'train_good':
                    action = 'A-9'
                return {'response': '<actions>' + json.dumps([action]) + '</actions>',
                        'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}
            rows.append(run_episode(item, mode, Settings(action_only=True, max_steps=2, max_calls=2), directory, actor, skill=skill))
        return rows
    def proposer(prompt):
        assert 'validation' not in prompt and '"task_id": "test"' not in prompt
        return {'response': json.dumps({**proposal(), 'evidence_ids': ['train_good__dynamic_interrupt', 'train_bad__dynamic_interrupt']})}
    results = evolve(splits(), EvolvedSkill(), evaluator, proposer, output, generations=1)
    return results, seen


def test_complete_offline_loop_freezes_skill_before_test(tmp_path):
    results, seen = demo_run(tmp_path / 'demo')
    assert set(results) == {'react', 'initial_dynamic', 'evolved_dynamic'}
    history = json.loads((tmp_path / 'demo/history.json').read_text())
    assert history[0]['accept']
    assert seen[-3:] == [('test', None, 'react'), ('test', 0, 'dynamic_interrupt'), ('test', 1, 'dynamic_interrupt')]
    assert (tmp_path / 'demo/best_skill.md').exists()


def test_rejected_candidate_does_not_replace_skill(tmp_path):
    _, seen = demo_run(tmp_path / 'rejected', improve=False)
    history = json.loads((tmp_path / 'rejected/history.json').read_text())
    assert not history[0]['accept']
    assert seen[-1] == ('test', 0, 'dynamic_interrupt')
    assert json.loads((tmp_path / 'rejected/best_skill.json').read_text())['revision'] == 0


def test_split_family_leakage(tmp_path):
    data = splits()
    data['train'][0]['family_id'] = 'same-family'
    data['test'][0]['family_id'] = 'same-family'
    with pytest.raises(ValueError, match='family'):
        validate_splits(data)


def test_live_entry_preflight_and_closed_cap_make_no_requests(tmp_path, monkeypatch):
    from dynamic_skill.campaign_budget import CampaignBudget
    from experiments import skill_evolution
    from experiments.puzzle_pilot import save
    import time
    save(tmp_path / 'splits.json', splits())
    budget = CampaignBudget(tmp_path / 'budget.json', deadline=time.time() + 100)
    def forbidden(*args, **kwargs):
        raise AssertionError('preflight must never construct an API client')
    monkeypatch.setattr(skill_evolution, 'ApiPlanner', forbidden)
    args = ['--splits', str(tmp_path / 'splits.json'), '--budget-json', str(budget.path),
            '--out', str(tmp_path / 'never_created'), '--preflight']
    before = budget.read()
    skill_evolution.main(args)
    assert budget.read() == before and not (tmp_path / 'never_created').exists()
    closed = budget.read(); closed['max_requests'] = 0; budget.write(closed)
    with pytest.raises(SystemExit):
        skill_evolution.main(args)


def test_puzzle_api_gpt_wire_contract_without_network(monkeypatch):
    from experiments.puzzle_pilot import ApiPlanner
    from types import SimpleNamespace
    sent = []
    def create(**kwargs):
        sent.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='<actions>["A+4"]</actions>'),
                                                        finish_reason='stop')], usage=None, id='fake', model='gpt-5.4')
    planner = object.__new__(ApiPlanner)
    planner.settings = Settings(model='gpt-5.4', max_tokens=512, action_only=True)
    planner.action_limit = 1; planner.system_prompt = None
    planner.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    planner('test')
    assert sent[0]['reasoning_effort'] == 'none' and sent[0]['max_completion_tokens'] == 512
    assert 'extra_body' not in sent[0] and 'max_tokens' not in sent[0]
