import json
from dataclasses import asdict
import pytest
from experiments.equal_operator_attribution import ARMS,WORKFLOWS,run_episode,build_prompt,parse_output
from experiments.puzzle_pilot import Settings
from dynamic_skill.observable_search_operator import ObservableSearch
from test_paper_controls import PublicEnv

class Env(PublicEnv):
    def reset(self,arg):
        obs,info=super().reset(arg);obs['text']=['Official public room prompt.'];return obs,info
    def step(self,wrapped):
        obs,*rest=super().step(wrapped);obs['text']=['Official latest public prompt.'];return (obs,*rest)

@pytest.mark.parametrize('arm',list(ARMS))
def test_every_method_gets_same_actual_operator_primitives_and_budget(tmp_path,arm):
    spec={**ARMS[arm],'workflow':WORKFLOWS['W2_acquire'],'learned_value':False}
    class Planner:
        count=0
        def before_call(self):pass
        def __call__(self,prompt):
            self.count+=1
            actions=['go to desk 1','move mug 1 to desk 1'] if spec['max_actions']>1 else [('go to desk 1' if self.count==1 else 'move mug 1 to desk 1')]
            text=('<think>Deliver.</think><action>'+actions[0]+'</action>') if spec['mode']=='official' else '<thought>Deliver.</thought><actions>'+json.dumps(actions)+'</actions>'
            return {'response':text,'usage':{'prompt_tokens':0,'completion_tokens':0}}
    game=tmp_path/'game';game.write_text('fixture')
    cfg={'settings':asdict(Settings(max_calls=2,max_steps=4)),'task_skill':'Use public commands.',
         'original_skill':'Untouched upstream skill.','arms':{arm:spec}}
    result=run_episode(Env(),{'id':'fixture','gamefile':str(game),'replicate':1},arm,cfg,Planner(),tmp_path/'out')
    assert result['success'] and result['environment_steps']==4
    assert result['compiled_search_actions']==2 and result['invalid_actions']==0
    assert result['model_calls']==(2 if spec['max_actions']==1 else 1)
    trace=json.loads((tmp_path/'out/conversation.json').read_text())
    assert [r['action'] for r in trace[:2]]==['go to shelf 1','take mug 1 from shelf 1']
    calls=json.loads((tmp_path/'out/model_calls.json').read_text())
    assert 'Shared public operator state:' in calls[0]['prompt']


def test_original_checkpoint_is_preserved_and_shared_state_identical():
    search=ObservableSearch('put a mug in desk.');search.searched.add('shelf 1')
    cfg={'original_skill':'FULL CHECKPOINT TEXT','task_skill':'CURRENT SKILL'}
    context={'task':'put a mug in desk.','initial':'Room.','observation':'Now.', 'admissible':['go to desk 1'],'trace':[],'anomaly':None}
    prompts=[build_prompt(None,{'text':['UPSTREAM TEMPLATE']},cfg,s,1,context,search) for s in ARMS.values()]
    assert 'FULL CHECKPOINT TEXT' in prompts[1] and 'FULL CHECKPOINT TEXT' not in prompts[0]
    assert len({p.split('Shared public operator state:',1)[1] for p in prompts})==1


def test_macros_are_unavailable_to_every_method():
    for spec in ARMS.values():
        text='<think>Search.</think><action>@scan shelf</action>' if spec['mode']=='official' else '<thought>Search.</thought><actions>["@scan shelf"]</actions>'
        with pytest.raises(ValueError):parse_output(text,spec,4)
