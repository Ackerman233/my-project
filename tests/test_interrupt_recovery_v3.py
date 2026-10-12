"""Synthetic counterexamples test code semantics, never model task accuracy."""
import json
from pathlib import Path
import pytest
from dynamic_skill.control_only_evolution import ControlExecutor,ControlPolicy,run_control_episode
from dynamic_skill.acquisition_check import AcquisitionMonitor,inventory_objects
from dynamic_skill.interrupter import DecisionType
from dynamic_skill.micro_loop_protocol import make_prompt
from experiments.puzzle_pilot import Settings

TAKE='take bowl 1 from desk 1'
PUT='move bowl 1 to countertop 1'
OK='You pick up the bowl 1 from the desk 1.'
WRONG='You pick up the mug 1 from the desk 1.'

def ctl(check=True,recovery=0):
    c=ControlExecutor(ControlPolicy(recovery_steps=recovery),state_check=check)
    c.start('On the desk 1, you see a bowl 1.','put a bowl on countertop')
    return c

def step(c,feedback):
    a=c.next_action();c.before_action(a,'current public observation',[a])
    return c.observe(a,feedback)

def test_mid_invalid_discards_but_tail_does_not_count():
    c=ctl();c.load_chunk([TAKE,PUT]);e=step(c,'Nothing happens.')
    assert e.discarded_actions==(PUT,) and c.interruption_count==c.anomaly_replan_count==1
    c=ctl();c.load_chunk([TAKE]);step(c,'Nothing happens.')
    assert c.interruption_count==0 and c.anomaly_replan_count==1

def test_synthetic_nonerror_contradiction_changes_execution():
    old,new=ctl(False),ctl(True)
    for c in (old,new):c.load_chunk([TAKE,PUT])
    assert step(old,WRONG).decision.kind is DecisionType.CONTINUE
    event=step(new,WRONG)
    assert event.decision.kind is DecisionType.REPLAN and event.discarded_actions==(PUT,)
    assert new.last_execution_audit['state_check']['status']=='contradicted'
    assert new.last_execution_audit['remaining_before_interrupt']==[PUT]
    assert new.last_execution_audit['original_action_queue']==[TAKE,PUT]

@pytest.mark.parametrize('feedback',['Done.','You pick up something.','You pick up the bowl 10 from the desk 1.'])
def test_unknown_and_exact_id(feedback):
    c=ctl();c.load_chunk([TAKE,PUT]);step(c,feedback)
    check=c.last_execution_audit['state_check']
    assert check['status']==('contradicted' if 'bowl 10' in feedback else 'unknown')
    assert check['task_completed'] is None and c.interruption_count==1

def test_normal_and_unrelated_tail_continue():
    c=ctl();c.load_chunk([TAKE,PUT]);assert step(c,OK).decision.kind is DecisionType.CONTINUE
    assert c.last_execution_audit['state_check']['status']=='confirmed'
    for tail in ['go to countertop 1','examine bowl 1','take bowl 1 from desk 1']:
        c=ctl();c.load_chunk([TAKE,tail]);assert step(c,'Unclear.').decision.kind is DecisionType.CONTINUE
    c=ctl();c.load_chunk(['take mug 1 from desk 1','move mug 1 to countertop 1'])
    step(c,'Unclear.');assert not c.last_execution_audit['state_check']['applicable']

def test_source_mismatch_does_not_block_when_holding_requirement_is_confirmed():
    c=ctl();c.load_chunk([TAKE,PUT])
    assert step(c,'You pick up the bowl 1 from the table 2.').decision.kind is DecisionType.CONTINUE
    check=c.last_execution_audit['state_check']
    assert check['status']=='contradicted' and check['holding_evidence_status']=='confirmed'
    assert not check['blocks_tail']  # Source mismatch doesn't invalidate possession needed by PUT.

def test_real_inventory_action_can_resolve_unknown_or_contradict():
    c=ctl();c.load_chunk([TAKE,'inventory',PUT]);step(c,'Unclear.')
    assert c.pending_actions[0]=='inventory' and c.interruption_count==0
    event=step(c,'You are not carrying anything.')
    assert event.discarded_actions==(PUT,)
    assert c.last_execution_audit['state_check']['status']=='contradicted'
    c=ctl();c.load_chunk([TAKE,'inventory',PUT]);step(c,'Unclear.')
    assert step(c,'You are carrying: a bowl 1.').decision.kind is DecisionType.CONTINUE

def test_inventory_parser_does_not_infer_absence_from_partial_list():
    assert inventory_objects('You are carrying: a bowl 1 and a mug 2.')==['bowl 1','mug 2']
    assert inventory_objects('You are carrying: a mug 1, ???.') is None

def test_counter_end_is_not_failure_resolution():
    c=ctl(recovery=2);c.load_chunk([TAKE,PUT]);step(c,'Nothing happens.')
    for feedback in ['You arrive at countertop 1.','You see a countertop 1.']:
        c.load_chunk(['look']);step(c,feedback)
    log=c.last_execution_audit
    assert log['recovery_counter_ended'] and not log['original_failure_verified_resolved']
    assert log['failure_resolution'][0]['resolution']=='unverified'
    c.load_chunk([TAKE]);step(c,OK)
    assert c.last_execution_audit['original_failure_verified_resolved']

def test_historical_pickup_not_reused_as_current():
    c=ctl();c.load_chunk([TAKE]);step(c,OK)
    c.load_chunk([TAKE,PUT]);step(c,'Unclear.')
    assert c.last_execution_audit['state_check']['status']=='unknown'
    assert c.last_execution_audit['state_check']['evidence']['age_steps']==0

class FakeEnv:
    tasks=['put a bowl on countertop'];get_admissible_commands=[[TAKE,PUT,'look']]
    def __init__(self):self.envs=self;self.actions=[]
    def reset(self,_):return {'anchor':['On the desk 1, you see a bowl 1.']},[{}]
    def step(self,actions):
        self.actions.extend(actions)
        return {'anchor':['Nothing happens.' if len(self.actions)==1 else 'A new room.']},[0],[False],[{}]

class FakePlanner:
    def before_call(self):pass
    def __call__(self,prompt):
        return {'response':'<thought>Check the object.</thought><actions>'+json.dumps([TAKE,PUT] if 'Observed anomaly:' not in prompt else ['look'])+'</actions>',
                'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}}

@pytest.mark.parametrize('enhance,recovery',[(False,0),(False,2),(True,0),(True,2)])
def test_independent_flags_and_prompt_logging(tmp_path,enhance,recovery):
    env=FakeEnv();row=run_control_episode(env,{'id':'synthetic','gamefile':'synthetic'},
        Settings(max_calls=2,max_steps=6),tmp_path,FakePlanner(),'frozen',ControlPolicy(recovery_steps=recovery),history_evidence=enhance)
    p=tmp_path/'predictions'/row['id'];calls=json.loads((p/'model_calls.json').read_text())
    log=json.loads((p/'conversation.json').read_text())
    first,second=calls
    assert not first['historical_evidence_injected']
    assert second['generic_anomaly_prompt_used']
    assert second['historical_evidence_injected']==enhance==second['recovery_prompt_used']
    assert second['single_step_recovery_active']==bool(recovery)
    assert second['effective_action_limit']==(1 if recovery else 4)
    assert ('Recovery evidence (' in second['prompt'])==enhance
    assert 'Return only the same actions JSON block' not in second['prompt']
    assert 'thought/actions' in second['prompt']
    assert log[0]['actual_discarded_actions']==[PUT]
    assert second['anomaly_received']==log[0]['interruption_reason']
    if enhance:
        assert 'current_state_guaranteed' in second['prompt'] and 'age_steps' in second['prompt']
    else:
        expected=make_prompt('frozen',second['effective_action_limit'],'Replan from the current observation.')('',
            task=env.tasks[0],initial='On the desk 1, you see a bowl 1.',observation='Nothing happens.',
            admissible=env.get_admissible_commands[0],trace=log[:1],anomaly=second['anomaly_received'])
        assert second['prompt']==expected  # Legacy prompt bytes preserved when off.

def test_mock_end_to_end_semantic_pause_reaches_next_prompt(tmp_path):
    class Env(FakeEnv):
        def step(self,actions):
            self.actions.extend(actions)
            return {'anchor':[WRONG if len(self.actions)==1 else 'You are not carrying anything.']},[0],[False],[{}]
    env=Env()
    row=run_control_episode(env,{'id':'synthetic','gamefile':'synthetic'},Settings(max_calls=2,max_steps=2),
        tmp_path,FakePlanner(),'frozen',ControlPolicy(),history_evidence=True,state_check=True)
    calls=json.loads((tmp_path/'predictions'/row['id']/'model_calls.json').read_text())
    assert len(env.actions)==2 and all(PUT not in a for a in env.actions)
    assert calls[1]['state_recovery_received']['last_check']['status']=='contradicted'
    assert row['false_interrupt_count'] is None and row['history_evidence_calls']==1
