import time
import pytest
from dynamic_skill.campaign_budget import CampaignHalt
from experiments.bounded_research95 import StageBudget,auth_key
from experiments.unknown_independent_validation import choose_unknown


def test_unknown_selector_does_not_use_following_failure_or_episode_outcome():
    trace=[{'step':0,'done':False,'env_feedback':'On the shelf 1, you see nothing.','remaining_before_interrupt':['put box 1 in shelf 1'],'valid':True},
           {'step':1,'done':False,'env_feedback':'On the shelf 1, you see nothing.','remaining_before_interrupt':['put box 2 in shelf 1'],'valid':False}]
    selected,counts=choose_unknown(trace,{0:['look'],1:['look']})
    assert selected['step']==0 and counts['unknown']==2
    trace[0]['valid'],trace[1]['valid']=False,True
    assert choose_unknown(trace,{0:['look'],1:['look']})[0]['step']==0


def test_surface_false_and_supported_head_do_not_enter_unknown_stratum():
    trace=[{'step':0,'done':False,'env_feedback':'On the drawer 1, you see nothing.','remaining_before_interrupt':['open drawer 1']},
           {'step':1,'done':False,'env_feedback':'Room.','remaining_before_interrupt':['go to shelf 1']}]
    selected,counts=choose_unknown(trace,{0:['look'],1:['go to shelf 1']})
    assert selected is None and counts=={'true':1,'false':1,'unknown':0}


def budget_fixture(tmp_path,ceiling=2,unknown=1):
    root=tmp_path/'trial'
    b=StageBudget(root,path=tmp_path/'ledger.json');d=b.read()
    d.update(limit_usd=95,deadline_unix=time.time()+3600,max_requests=10,max_unknown_requests=5)
    d[auth_key(root)]={'start_request_count':0,'deadline_unix':time.time()+3600,'max_requests':10,
        'batch_spend_ceiling':ceiling,'max_new_unknown':unknown}
    b.write(d)
    return b


def test_unknown_request_keeps_charge_and_stops_at_local_cap(tmp_path):
    b=budget_fixture(tmp_path)
    index=b.reserve('trial/job')
    b.settle(index,error=TimeoutError())
    assert b.read()['requests'][index]['budget_charge_usd']==.5
    with pytest.raises(CampaignHalt,match='unknown_cost_stop'):
        b.reserve('trial/new-job')


def test_stage_cannot_spend_remaining_global_funds_outside_its_frozen_ceiling(tmp_path):
    b=budget_fixture(tmp_path,ceiling=.6,unknown=2)
    index=b.reserve('trial/job');b.settle(index,{'usage':{'prompt_tokens':100000,'completion_tokens':0}})
    # Default conservative input rate is 1 USD/M: charge .1, next .5 fits exactly.
    second=b.reserve('trial/job2');b.settle(second,error=TimeoutError())
    with pytest.raises(CampaignHalt,match='budget_stop'):
        b.reserve('trial/job3')


def test_historical_scene_exclusion_has_no_runner_namespace_exception(tmp_path):
    import json
    from pathlib import Path
    from experiments.unknown_independent_validation import used_scene_names
    game=Path('/fake/valid_seen/pick_and_place_simple-Box-None-Shelf-302/trial/game.tw-pddl')
    previous=tmp_path/'config.json'
    previous.write_text(json.dumps({'tasks':[{'gamefile':str(game)}]}))
    assert game.parent.parent.name in used_scene_names([previous],[game])
