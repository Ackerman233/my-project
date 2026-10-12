from dataclasses import asdict, replace
import json
from pathlib import Path
import time
import pytest

from dynamic_skill.infrastructure.campaign_budget import CampaignBudget, CampaignHalt
from experiments.alfworld.joint_evolution.engine import evolve, training_evidence
from experiments.alfworld.joint_evolution.schema import Genome
from experiments.alfworld.joint_evolution.skillopt_core import (
    SkillOptCore, CandidateRejected, apply_joint_patch, policy_json, PIPELINE)
from experiments.alfworld.joint_evolution.skillopt_demo import (
    ScriptedOptimizer, scripted_evaluator, run_demo)
from experiments.alfworld.joint_evolution.demo import splits
from experiments.alfworld.joint_evolution.runtime import LiveTransport
from experiments.common.puzzle_pilot import Settings


def evidence():
    return [{'id': name, 'task_id': name, 'success': success,
        'status': 'success' if success else 'model_call_limit', 'model_calls': 1,
        'environment_steps': 1, 'invalid_actions': int(not success), 'cost_usd': .01,
        'public_task': 'Reach a visible destination.', 'trajectory': [
            {'step': 0, 'action': 'go to desk 1' if success else 'open drawer 1',
             'env_feedback': 'Arrived.' if success else 'Nothing happens.', 'valid': success,
             'remaining_before_interrupt': [] if success else ['go to desk 1'],
             'actual_discarded_actions': [] if success else ['go to desk 1']}]
    } for name, success in [('train-failed', False), ('train-succeeded', True)]]


def test_actual_upstream_pipeline_failure_edit_changes_execution_and_passes_confirmation(tmp_path, monkeypatch):
    from skillopt.gradient import reflect, aggregate
    from skillopt.optimizer import clip, skill
    seen = []
    for module, name in [(reflect, 'run_error_analyst_minibatch'), (reflect, 'run_success_analyst_minibatch'),
                         (aggregate, 'merge_patches'), (clip, 'rank_and_select'), (skill, 'apply_patch_with_report')]:
        original = getattr(module, name)
        def spy(*args, _fn=original, _name=name, **kwargs):
            seen.append(_name)
            return _fn(*args, **kwargs)
        monkeypatch.setattr(module, name, spy)
    originals = [m.chat_optimizer for m in (reflect, aggregate, clip)]
    best, final = run_demo(tmp_path / 'run')
    assert {'run_error_analyst_minibatch', 'run_success_analyst_minibatch', 'merge_patches',
            'rank_and_select', 'apply_patch_with_report'} <= set(seen)
    assert [m.chat_optimizer for m in (reflect, aggregate, clip)] == originals
    history = json.loads((tmp_path / 'run/history.json').read_text())
    assert [r['accepted'] for r in history] == [True, False]
    assert best.policy.invalid_patience == 2 and best.revision == 1
    assert not final['initial'][0]['success'] and not final['skill_only'][0]['success']
    assert final['policy_only'][0]['success'] and final['evolved'][0]['success']
    assert final['initial'][0]['actual_tail_interruptions'] == 1
    assert final['evolved'][0]['actual_tail_interruptions'] == 0
    folder = tmp_path / 'run/generations/000/candidates/00/optimizer'
    application = json.loads((folder / 'apply.json').read_text())
    assert application['failure_evidence_for_policy'] == ['g0-train-public']
    assert application['changed_policy_fields'] == ['invalid_patience']
    assert json.loads((folder / 'status.json').read_text())['optimizer_calls'] == 4
    assert json.loads((folder / 'rank.json').read_text())['optimizer_calls'] == 1
    assert json.loads((tmp_path / 'run/manifest.json').read_text())['optimizer_pipeline'] == PIPELINE
    for path in (tmp_path / 'run/generations').glob('*/candidates/*/optimizer/calls/*.json'):
        text = path.read_text()
        assert all(x not in text for x in ('select-private', 'confirm-private', 'test-private'))
        call = json.loads(text)
        assert call['effective_retries'] == 0 and call['effective_max_tokens'] == 1536


@pytest.mark.parametrize('failed_stage', ['reflect_failure', 'reflect_success', 'aggregate', 'rank'])
def test_upstream_cannot_swallow_api_error_and_apply_fallback(tmp_path, failed_stage):
    from skillopt.gradient import reflect, aggregate
    from skillopt.optimizer import clip
    originals = [m.chat_optimizer for m in (reflect, aggregate, clip)]
    scripted = ScriptedOptimizer()
    attempted = []
    def request(prompt, **kwargs):
        phase = Path(kwargs['label']).parent.name
        attempted.append(phase)
        if phase == failed_stage:
            raise CampaignHalt('api_error')
        return scripted(prompt, **kwargs)
    with pytest.raises(CampaignHalt, match='api_error'):
        SkillOptCore(request)(Genome('Original skill.'), evidence(), tmp_path / 'candidate')
    assert attempted[-1] == failed_stage and attempted.count(failed_stage) == 1
    assert [m.chat_optimizer for m in (reflect, aggregate, clip)] == originals
    assert not (tmp_path / 'candidate/optimizer/apply.json').exists()
    assert json.loads((tmp_path / 'candidate/optimizer/status.json').read_text())['status'] == 'stopped'


@pytest.mark.parametrize('failed_stage', ['reflect_failure', 'aggregate', 'rank'])
def test_bad_optimizer_output_is_rejected_instead_of_upstream_fallback(tmp_path, failed_stage):
    scripted = ScriptedOptimizer()
    def request(prompt, **kwargs):
        if Path(kwargs['label']).parent.name == failed_stage:
            return {'response': '{}', 'cost_usd': 0}
        return scripted(prompt, **kwargs)
    with pytest.raises(CandidateRejected):
        SkillOptCore(request)(Genome('Original skill.'), evidence(), tmp_path / 'candidate')
    assert not (tmp_path / 'candidate/optimizer/apply.json').exists()
    assert json.loads((tmp_path / 'candidate/optimizer/status.json').read_text())['status'] == 'rejected'


def test_failure_only_uses_upstream_fast_paths_without_fake_merge_or_rank_calls(tmp_path):
    request = ScriptedOptimizer()
    best = SkillOptCore(request)(Genome('Original skill.'), evidence()[:1], tmp_path / 'candidate')
    assert best.policy.invalid_patience == 2
    assert request.calls == ['reflect_failure']
    for name in ('aggregate', 'rank'):
        assert json.loads((tmp_path / f'candidate/optimizer/{name}.json').read_text())['fast_path_no_model_call']


def test_policy_edit_cannot_cite_success_when_only_failure_evidence_was_supplied(tmp_path):
    request = ScriptedOptimizer()
    def forged(prompt, **kwargs):
        reply = request(prompt, **kwargs)
        raw = json.loads(reply['response'])
        raw['patch']['edits'][0]['evidence_ids'] = ['test-private']
        return {**reply, 'response': json.dumps(raw)}
    with pytest.raises(CandidateRejected, match='training evidence'):
        SkillOptCore(forged)(Genome('Original skill.'), evidence()[:1], tmp_path / 'candidate')


@pytest.mark.parametrize('mutation', ['out_of_range', 'partial_policy', 'missing_target'])
def test_actual_upstream_apply_rejects_invalid_joint_policy_edits(mutation):
    parent = Genome('Original skill.')
    policy = {**asdict(parent.policy), 'invalid_patience': 2}
    edit = {'op': 'replace', 'target': policy_json(parent.policy),
            'content': json.dumps(policy), 'evidence_ids': ['train-failed']}
    text = {'op': 'append', 'content': 'Use public feedback.', 'evidence_ids': ['train-failed']}
    if mutation == 'out_of_range':
        edit['content'] = json.dumps({**policy, 'invalid_patience': 99})
    elif mutation == 'partial_policy':
        edit.update(target='"invalid_patience": 1', content='"invalid_patience": 2')
    elif mutation == 'missing_target':
        text.update(op='replace', target='does not exist')
    edits = [edit] if mutation == 'policy_only' else [edit, text]
    with pytest.raises(CandidateRejected):
        apply_joint_patch(parent, {'edits': edits}, evidence())


def test_training_reflection_gets_full_public_trace_but_no_hidden_state(tmp_path):
    row = {**evidence()[0], 'trace': [
        {'step': i, 'action': 'look', 'env_feedback': f'public {i}',
         'hidden_state': 'SECRET_ORACLE', 'reference_text': 'SECRET_ORACLE'} for i in range(20)]}
    public = training_evidence([row], 0)
    assert len(public[0]['trajectory']) == 20
    assert 'SECRET_ORACLE' not in json.dumps(public)
    request = ScriptedOptimizer()
    SkillOptCore(request)(Genome('Original skill.'), public, tmp_path / 'candidate')
    call = json.loads(next((tmp_path / 'candidate/optimizer/calls').glob('*.json')).read_text())
    assert 'public 10' in call['prompt'] and 'SECRET_ORACLE' not in call['prompt']


def test_all_core_optimizer_stages_charge_actor_shared_budget(tmp_path, monkeypatch):
    from experiments.alfworld.joint_evolution import runtime
    ledger = CampaignBudget(tmp_path / 'budget.json', deadline=time.time()+600, limit=10)
    scripted = ScriptedOptimizer()
    stages = iter(['reflect_failure', 'reflect_success', 'aggregate', 'rank'])
    class Api:
        def __init__(self, settings, system_prompt):
            self.system = system_prompt
        def close(self):
            pass
        def __call__(self, prompt):
            if prompt == 'actor observation':
                result = {'response': '{}'}
            else:
                phase = next(stages)
                result = scripted(prompt, system=self.system, max_tokens=1536, label=f'{phase}/optimizer')
            return {**result, 'usage': {'prompt_tokens': 10, 'completion_tokens': 5}}
    monkeypatch.setattr(runtime, 'ApiPlanner', Api)
    transport = LiveTransport(tmp_path, ledger, Settings(max_tokens=128))
    transport.actor('actor')('actor observation')
    best = transport.candidate_builder(1536)(Genome('Original skill.'), evidence(), tmp_path / 'candidate')
    assert best.policy.invalid_patience == 2
    rows = ledger.read()['requests']
    assert len(rows) == 5 and all(r['status'] == 'returned_usage' and r['budget_charge_usd'] > 0 for r in rows)
    assert {Path(r['label']).parent.name for r in rows[1:]} == {'reflect_failure','reflect_success','aggregate','rank'}


def test_core_confirmation_no_gain_keeps_old_genome(tmp_path):
    def evaluator(tasks, genome, directory):
        # In the independent fixture fold, navigation cannot survive either
        # policy: replay the incumbent executor to model this negative result.
        actual = Genome(genome.skill_text) if 'confirmation' in directory.parts else genome
        return scripted_evaluator(tasks, actual, directory)
    parent = Genome('Original skill.')
    best, _ = evolve(splits(1), parent, evaluator, None, tmp_path / 'run',
        generations=1, candidates_per_generation=1, candidate_builder=SkillOptCore(ScriptedOptimizer()))
    assert best == parent
    decision = json.loads((tmp_path / 'run/history.json').read_text())[0]
    assert not decision['accepted'] and not decision['confirmation_gate']['accept']


def test_core_transport_error_stops_whole_evolution_before_test(tmp_path):
    count = []
    def failed(*args, **kwargs):
        count.append(kwargs['label'])
        raise CampaignHalt('api_error')
    with pytest.raises(CampaignHalt):
        evolve(splits(1), Genome('Original skill.'), scripted_evaluator, None, tmp_path / 'run',
            generations=1, candidates_per_generation=1, candidate_builder=SkillOptCore(failed))
    assert len(count) == 1 and not (tmp_path / 'run/test').exists()
    assert json.loads((tmp_path / 'run/status.json').read_text())['status'] == 'stopped'


def test_empty_patch_does_not_force_interrupt_update(tmp_path):
    def no_update(*args, **kwargs):
        return {'response': json.dumps({'patch': {'edits': []}})}
    with pytest.raises(CandidateRejected, match='change at least one'):
        SkillOptCore(no_update)(Genome('Original skill.'), evidence()[:1], tmp_path / 'candidate')
    assert json.loads((tmp_path / 'candidate/optimizer/reflect_success.json').read_text())['status'] == 'skipped_empty_group'


def test_truncated_optimizer_reply_is_rejected_even_if_json_parses(tmp_path):
    scripted = ScriptedOptimizer()
    def truncated(*args, **kwargs):
        return {**scripted(*args, **kwargs), 'finish_reason': 'length'}
    with pytest.raises(CandidateRejected, match='Truncated'):
        SkillOptCore(truncated)(Genome('Original skill.'), evidence()[:1], tmp_path / 'candidate')
    assert not (tmp_path / 'candidate/optimizer/apply.json').exists()


def test_joint_schema_is_in_system_not_misrepresented_as_previous_skill_steps(tmp_path):
    SkillOptCore(ScriptedOptimizer())(Genome('Original skill.'), evidence(), tmp_path / 'candidate')
    call = json.loads((tmp_path / 'candidate/optimizer/calls/01_reflect_failure.json').read_text())
    assert 'Authoritative joint output schema' in call['system']
    assert '"evidence_ids"' in call['system'] and 'train-failed' in call['system']
    assert json.dumps(policy_json(Genome('Original skill.').policy)) in call['system']
    assert 'Previous Steps in This Epoch' not in call['prompt']
    assert 'Joint skill and executable interrupt optimization contract' not in call['prompt']


def test_footer_contract_reaches_each_actual_upstream_stage(tmp_path):
    scripted=ScriptedOptimizer()
    def request(prompt,**kwargs):
        assert prompt.rfind('FINAL OUTPUT CONTRACT') > prompt.find('Current Skill')
        assert 'evidence_ids is REQUIRED' in prompt
        return scripted(prompt.split("\n\n## FINAL OUTPUT CONTRACT")[0],**kwargs)
    candidate=SkillOptCore(request,footer_contract=True)(Genome('Original skill.'),evidence(),tmp_path/'candidate')
    assert candidate.policy.invalid_patience==2
    assert scripted.calls==['reflect_failure','reflect_success','aggregate','rank']


def test_batch_provenance_keeps_missing_model_citations_missing(tmp_path):
    reply={'patch':{'edits':[{'op':'append','target':'','content':'Check publicly observed possession before delivery.'}]}}
    original=json.dumps(reply)
    def request(prompt,**kwargs):return {'response':original}
    candidate=SkillOptCore(request,citation_mode='batch_provenance',footer_contract=True)(Genome('Original skill.'),evidence()[:1],tmp_path/'candidate')
    assert candidate.evidence_ids==()
    app=json.loads((tmp_path/'candidate/optimizer/apply.json').read_text())
    assert app['training_input_ids']==['train-failed'] and app['model_citation_ids']==[]
    assert app['missing_citations_were_not_filled'] and not app['per_edit_causal_attribution_established']
    raw=json.loads((tmp_path/'candidate/optimizer/calls/01_reflect_failure.json').read_text())
    assert raw['reply']['response']==original
    with pytest.raises(CandidateRejected):
        apply_joint_patch(Genome('Skill.'),{'edits':[{'op':'append','content':'New.','evidence_ids':['heldout-secret']}]},evidence(),citation_mode='batch_provenance')
