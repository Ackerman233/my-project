import json
import os
from pathlib import Path
import pytest
from experiments.alfworld_campaign import prompt_for, run_episode, SYSTEM
from experiments.puzzle_pilot import Settings


def test_billing_checkpoint_preserves_unknown_and_guard(tmp_path):
    from experiments.alfworld_campaign import authorize_billing_resume
    from dynamic_skill.campaign_budget import CampaignBudget, CampaignHalt
    budget=CampaignBudget(tmp_path/'b.json',deadline=100,clock=lambda:1)
    budget.settle(budget.reserve('unknown'),error=ConnectionError())
    original=budget.read()['requests']
    authorize_billing_resume(budget,9)
    state=budget.read()
    assert state['requests']==original and state['limit_usd']==5
    assert CampaignBudget.effective_spend(state)==pytest.approx(.94)
    with pytest.raises(AssertionError): authorize_billing_resume(budget,9)
    state['limit_usd']=1
    budget.write(state)
    with pytest.raises(CampaignHalt,match='budget_stop'): budget.reserve('over')


def test_twenty_authorization(tmp_path):
    from experiments.alfworld_campaign import authorize_twenty_resume
    from dynamic_skill.campaign_budget import CampaignBudget
    budget=CampaignBudget(tmp_path/'b.json',deadline=100,clock=lambda:1)
    budget.settle(budget.reserve('unknown'),error=ConnectionError())
    before=budget.read()
    authorize_twenty_resume(budget,3)
    after=budget.read()
    assert after['limit_usd']==20 and after['requests']==before['requests']
    assert CampaignBudget.effective_spend(after)==pytest.approx(1.15)
    assert after['max_requests']==151 and after['cooldown_until']==before['cooldown_until']
    with pytest.raises(AssertionError): authorize_twenty_resume(budget,3)


def test_action_only_prompt():
    prompt=prompt_for('task','observation',['look'],[],'skill',4,'')
    assert '<think>' not in prompt and '1 to 4' in prompt
    assert 'no reasoning' in prompt.lower() and '<think>' not in SYSTEM


def test_resume_preserves_charges_and_is_one_time(tmp_path):
    from experiments.alfworld_campaign import authorize_resume
    from dynamic_skill.campaign_budget import CampaignBudget
    budget=CampaignBudget(tmp_path/'b.json',deadline=100,clock=lambda:1)
    budget.settle(budget.reserve('unknown'),error=ConnectionError())
    old=budget.read()
    authorize_resume(budget,11)
    new=budget.read()
    assert new['requests']==old['requests'] and new['limit_usd']==old['limit_usd']
    assert new['max_unknown_requests']==2 and new['max_requests']==551
    assert new['cooldown_until']==old['cooldown_until']
    with pytest.raises(AssertionError): authorize_resume(budget,11)


@pytest.mark.skipif(os.environ.get('RUN_ALFWORLD_INTEGRATION')!='1',reason='requires ALFWorld data')
@pytest.mark.parametrize('mode',['react','fixed_chunk','dynamic_interrupt'])
def test_real_new_protocol_queue(tmp_path,mode):
    from skillopt.envs.alfworld.rollout import build_alfworld_env
    game=Path(os.environ['ALFWORLD_DATA'])/'json_2.1.1/valid_seen/look_at_obj_in_light-AlarmClock-None-DeskLamp-323/trial_T20190909_044715_250790/game.tw-pddl'
    actions=['go to desk 1','take alarmclock 2 from desk 1','use desklamp 1']
    plans=iter([[a] for a in actions] if mode=='react' else [actions])
    class Planner:
        def before_call(self): pass
        def __call__(self,prompt):
            return {'response':'<actions>'+json.dumps(next(plans))+'</actions>',
                    'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}}
    env=build_alfworld_env(env_num=1,eval_dataset='eval_in_distribution',seed=42,is_train=False,specific_gamefiles=[str(game)])
    try:
        row=run_episode(env,{'id':'test','gamefile':str(game)},mode,
            Settings(max_steps=8,action_only=True),tmp_path,Planner(),'')
    finally: env.close()
    assert row['success'] and row['environment_steps']==3
    assert row['model_calls']==(3 if mode=='react' else 1)
    assert row['multi_action_chunks']==(0 if mode=='react' else 1)
    assert row['invalid_actions']==0
