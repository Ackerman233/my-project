import json
import os
from pathlib import Path
import pytest
from experiments.alfworld_campaign import prompt_for, run_episode, SYSTEM
from experiments.puzzle_pilot import Settings


def test_action_only_prompt():
    prompt=prompt_for('task','observation',['look'],[],'skill',4,'')
    assert '<think>' not in prompt and '1 to 4' in prompt
    assert 'no reasoning' in prompt.lower() and '<think>' not in SYSTEM


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
