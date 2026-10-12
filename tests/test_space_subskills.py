import pytest
from dynamic_skill.space_subskills import expand
CTX={'initial':'drawer 1, desk 1, desk 2','admissible':['go to drawer 1','go to desk 1']}
def test_expansion_and_provenance():
    a,e=expand(['@inspect drawer 1'],CTX,4)
    assert a==['go to drawer 1','open drawer 1'] and e['not_model_emitted_primitives']
    assert expand(['@scan desk 1|desk 2'],CTX,4)[0]==['go to desk 1','go to desk 2']
@pytest.mark.parametrize('a,k',[(['@inspect drawer 9'],4),(['@inspect drawer 1'],1),(['@scan desk 1|desk 1'],4),(['@inspect drawer 1','inventory'],2),(['@scan desk 1|drawer 1'],4)])
def test_reject_unsupported(a,k):
    with pytest.raises(ValueError):expand(a,CTX,k)
def test_primitive_fallback():
    assert expand(['inventory'],CTX,4)==(['inventory'],None)

def test_compositional_scan_and_surface_inspect():
    assert expand(['@scan desk 1','@scan desk 2'],CTX,4)[0]==['go to desk 1','go to desk 2']
    assert expand(['@inspect desk 1'],CTX,4)[0]==['go to desk 1']

def test_exact_missing_tag_normalization_only():
    from dynamic_skill.space_subskills import normalize_response
    from dynamic_skill.micro_loop_protocol import parse_response
    raw='<thought>Inspect.</thought>\n["@inspect drawer 1"]'
    text,changed=normalize_response(raw)
    assert changed and parse_response(text,4)[0]==['@inspect drawer 1']
    assert normalize_response('extra '+raw)==('extra '+raw,False)
    assert normalize_response(text)==(text,False)

def test_runner_keeps_raw_skill_and_executes_expansion(tmp_path):
    import json
    from dynamic_skill.control_only_evolution import run_control_episode,ControlPolicy
    from experiments.puzzle_pilot import Settings
    class Env:
        tasks=['put some egg on microwave.']
        def __init__(self):self.envs=self;self.get_admissible_commands=[['go to drawer 1']];self.actions=[]
        def reset(self,_):return {'anchor':['Room has drawer 1.']},[{}]
        def step(self,actions):
            self.actions+=actions;self.get_admissible_commands=[['open drawer 1']]
            return {'anchor':['You arrive at drawer 1.' if len(self.actions)==1 else 'You open drawer 1.']},[0],[False],[{}]
    class Planner:
        def before_call(self):pass
        def __call__(self,p):return {'response':'<thought>Inspect.</thought><actions>["@inspect drawer 1"]</actions>','usage':{'prompt_tokens':1,'completion_tokens':1}}
    env=Env();r=run_control_episode(env,{'id':'synthetic','gamefile':'synthetic'},Settings(max_calls=1,max_steps=2),tmp_path,Planner(),'skill',ControlPolicy(),macro_mode=True)
    c=json.loads((tmp_path/'predictions'/r['id']/'model_calls.json').read_text())[0]
    assert len(env.actions)==2 and r['subskill_calls']==1
    assert c['model_native_actions']==['@inspect drawer 1'] and len(c['actions'])==2
