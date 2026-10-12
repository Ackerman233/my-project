import pytest
from dataclasses import asdict
from dynamic_skill.control_only_evolution import *
from dynamic_skill.interrupter import DecisionType

def executor(policy):
    c=ControlExecutor(policy);c.start('room','put object');return c
def step(c,obs):return c.observe(c.next_action(),obs)

def test_prompt_changes_rejected():
    p={'task_skill_sha256':fingerprint('fixed'),'control':asdict(ControlPolicy()),'task_skill':'new'}
    with pytest.raises(ValueError):accept_proposal(p,ControlPolicy(),'fixed')
    p.pop('task_skill');p['control']['recovery_note']='new prose'
    with pytest.raises(ValueError):accept_proposal(p,ControlPolicy(),'fixed')

def test_noop_not_forced_and_skill_immutable():
    p={'task_skill_sha256':fingerprint('fixed'),'control':asdict(ControlPolicy())}
    result=accept_proposal(p,ControlPolicy(),'fixed');assert result['status']=='noop'
    p['control']['recovery_steps']=2
    result=accept_proposal(p,ControlPolicy(),'fixed')
    assert result['changed_fields']==['recovery_steps'] and result['task_skill']=='fixed'

@pytest.mark.parametrize('field,value',[('max_actions',True),('max_actions',5),('invalid_patience',0),('recovery_steps',99),('trigger','always')])
def test_bounds(field,value):
    with pytest.raises(ValueError):ControlPolicy(**{**asdict(ControlPolicy()),field:value})

def test_patience_changes_queue_discard():
    a=executor(ControlPolicy(invalid_patience=1));b=executor(ControlPolicy(invalid_patience=2))
    for c in [a,b]:c.load_chunk(['bad1','bad2','look'])
    assert step(a,'Nothing happens.').discarded_actions==('bad2','look')
    assert not step(b,'Nothing happens.').discarded_actions
    assert step(b,'Nothing happens.').discarded_actions==('look',)

def test_recovery_enforced_and_expires():
    c=executor(ControlPolicy(recovery_steps=2));c.load_chunk(['bad','look'])
    assert step(c,'Nothing happens.').decision.kind is DecisionType.REPLAN
    assert c.planning_limit==1
    with pytest.raises(ValueError):c.load_chunk(['a','b'])
    for obs in ['New location 1','New location 2']:
        c.load_chunk(['go']);step(c,obs)
    assert c.planning_limit==4

def test_single_coordinate_search():
    base=ControlPolicy();candidates=neighbors(base)
    assert len(candidates)==9 and candidates[0]==base
    assert all(sum(asdict(c)[k]!=v for k,v in asdict(base).items())==1 for c in candidates[1:])

def test_terminal_success_not_suppressed():
    c=executor(ControlPolicy(trigger='invalid'));c.load_chunk(['finish','look'])
    event=c.observe(c.next_action(),'done',done=True,reward=1)
    assert event.decision.kind is DecisionType.STOP_SUCCESS

@pytest.mark.parametrize('oversize',[False,True])
def test_adapter_enforces_recovery_protocol(tmp_path,oversize):
    from experiments.puzzle_pilot import Settings
    class Env:
        tasks=['put object'];get_admissible_commands=[['look']]
        def __init__(self):self.envs=self;self.count=0
        def reset(self,arg):return {'anchor':['room']},[{}]
        def step(self,actions):
            self.count+=1;done=self.count==2
            return {'anchor':['done' if done else 'Nothing happens.']},[int(done)],[done],[{'won':done}]
    class Planner:
        def __init__(self):self.prompts=[]
        def before_call(self):pass
        def __call__(self,prompt):
            self.prompts.append(prompt)
            actions=['bad','discard'] if len(self.prompts)==1 else (['look','extra'] if oversize else ['look'])
            return {'response':'<thought>Next step.</thought><actions>'+json.dumps(actions)+'</actions>',
                    'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}}
    env=Env();planner=Planner()
    row=run_control_episode(env,{'id':'mock','gamefile':'mock'},Settings(max_steps=5,max_calls=3),tmp_path,
                            planner,'FROZEN TASK RULES',ControlPolicy(recovery_steps=2))
    assert row['effective_action_limits']==[4,1]
    assert 'Return 1 to 1' in planner.prompts[1]
    assert env.count==(1 if oversize else 2)
    assert row['status']==('format_error' if oversize else 'success')
