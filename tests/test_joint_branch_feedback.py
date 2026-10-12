from dataclasses import asdict
import json
from pathlib import Path
import pytest
from experiments.alfworld.joint_evolution.branch_feedback import compare
from experiments.alfworld.joint_evolution.skillopt_core import apply_joint_patch, policy_json, CandidateRejected
from experiments.alfworld.joint_evolution.schema import Genome
from experiments.alfworld.joint_evolution.engine import evolve
from experiments.alfworld.joint_evolution.demo import splits, scripted_evaluator


def test_component_updates_are_independent_but_noop_is_rejected():
    parent = Genome('Original skill.')
    evidence = [{'id':'training'}]
    skill = {'op':'append', 'target':'', 'content':'Check public feedback.', 'evidence_ids':['training']}
    policy = {'op':'replace', 'target':policy_json(parent.policy),
              'content':json.dumps({**asdict(parent.policy), 'invalid_patience':2}), 'evidence_ids':['training']}
    a, _ = apply_joint_patch(parent, {'edits':[skill]}, evidence)
    b, _ = apply_joint_patch(parent, {'edits':[policy]}, evidence)
    c, _ = apply_joint_patch(parent, {'edits':[policy,skill]}, evidence)
    assert a.policy == parent.policy and a.skill_text != parent.skill_text
    assert b.policy != parent.policy and b.skill_text == parent.skill_text
    assert c.policy != parent.policy and c.skill_text != parent.skill_text
    with pytest.raises(CandidateRejected):apply_joint_patch(parent, {'edits':[]}, evidence)


def test_branch_signal_does_not_reward_cheap_failure_or_mismatched_prefix():
    case = {'kind':'ordinary_nonempty_queue','step':2,'tail':['go to desk 1']}
    def row(success, calls):
        return {'success':success, 'status':'success' if success else 'model_call_limit',
                'checkpoint_sha256':'same','model_calls':calls, 'environment_steps':calls,
                'invalid_actions':0,'cost_usd':calls*.1}
    rows = {'continue_tail':row(False,4), 'interrupt_replan':row(False,1)}
    assert compare(case,rows)['label'] == 'unknown'
    rows['continue_tail'] = row(True,4)
    assert compare(case,rows)['label'] == 'continue_win'
    rows['interrupt_replan'] = row(True,2)
    assert compare(case,rows)['label'] == 'interrupt_efficiency_win'
    rows['interrupt_replan']['checkpoint_sha256'] = 'different'
    assert not compare(case,rows)['valid_pair'] and compare(case,rows)['label'] == 'unknown'


def test_rejected_candidates_buy_no_selection_or_duplicate_final_episodes(tmp_path):
    class Reject:
        pipeline='test'
        def __call__(self,*args):raise CandidateRejected('No supported change')
    visits=[]
    def evaluator(tasks, genome, directory):
        visits.append(str(directory))
        return scripted_evaluator(tasks, genome, directory)
    evolve(splits(1),Genome('Skill.'),evaluator,None,tmp_path/'run',generations=1,
           candidates_per_generation=2,candidate_builder=Reject())
    assert len(visits)==2
    result=json.loads((tmp_path/'run/test/comparison.json').read_text())
    assert result['gate']['reason']=='identical_configuration'
    assert result['reused_arms']=={'skill_only':'initial','policy_only':'initial','evolved':'initial'}


def test_transport_retry_retains_unknown_cost_and_repeats_exact_prompt(tmp_path, monkeypatch):
    import time
    from experiments.alfworld.joint_evolution import runtime
    from dynamic_skill.infrastructure.campaign_budget import CampaignBudget
    from experiments.common.puzzle_pilot import Settings
    APIConnectionError = type('APIConnectionError',(Exception,),{})
    seen=[]
    class Api:
        def __init__(self, settings, system_prompt):pass
        def close(self):pass
        def __call__(self,prompt):
            seen.append(prompt)
            if len(seen)==1:raise APIConnectionError()
            return {'response':'ok','usage':{'prompt_tokens':10,'completion_tokens':2}}
    monkeypatch.setattr(runtime,'ApiPlanner',Api)
    budget=CampaignBudget(tmp_path/'budget.json',deadline=time.time()+60,limit=5)
    transport=runtime.LiveTransport(tmp_path,budget,Settings(),connection_retry=True)
    result=transport.request('same',system='system',max_tokens=128,label='actor')
    state=budget.read()
    assert seen==['same','same'] and result['response']=='ok'
    assert [r['status'] for r in state['requests']]==['unknown_usage','returned_usage']
    assert state['requests'][0]['budget_charge_usd']==.5
    assert state['consecutive_errors']==0 and (tmp_path/'transport/000001.retry.json').exists()


def test_unreachable_pooled_connection_makes_no_paid_reservation(tmp_path):
    import time
    from experiments.alfworld.joint_evolution.runtime import LiveTransport
    from dynamic_skill.infrastructure.campaign_budget import CampaignBudget,CampaignHalt
    from experiments.common.puzzle_pilot import Settings
    class BadHTTP:
        calls=0
        def head(self,*args,**kwargs):self.calls+=1;raise ConnectionError()
    budget=CampaignBudget(tmp_path/'budget.json',deadline=time.time()+60,limit=5)
    transport=LiveTransport(tmp_path,budget,Settings(),pooled_transport=True)
    transport._shared_http=BadHTTP()
    with pytest.raises(CampaignHalt,match='network_preflight_failed'):
        transport.request('prompt',system='system',max_tokens=128,label='actor')
    assert transport._shared_http.calls==3 and budget.read()['requests']==[]
    assert len(list((tmp_path/'connection_checks').glob('*.json')))==3


def test_footer_enforces_scope_and_citations_after_trajectory_data():
    from experiments.alfworld.joint_evolution.skillopt_core import output_footer
    footer=output_footer('reflect_failure',{'train-a'},'skill_only',Genome('Skill.'))
    assert 'evidence_ids is REQUIRED' in footer and 'Leave the executable policy EXACTLY unchanged' in footer
    assert 'train-a' in footer and 'outside trajectory data' in footer
    rank=output_footer('rank',{'train-a'},'policy_or_joint',Genome('Skill.'))
    assert 'selected_indices' in rank and 'Only select existing edit indices' in rank
