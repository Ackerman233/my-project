from dataclasses import asdict
import re
import numpy as np
from dynamic_skill.observable_search_operator import ObservableSearch,task_target
from dynamic_skill.value_termination import utility,fit,predict,FEATURES
from experiments.paper_method_trial import run_episode
from experiments.puzzle_pilot import Settings


def test_task_grammar_handles_look_at_and_does_not_guess_unrelated_goals():
    assert task_target('look at bowl under the desklamp.')=='bowl'
    assert task_target('examine the alarmclock with the desklamp.')=='alarmclock'
    assert task_target('look at alarmclock under a lamp.')=='alarmclock'
    assert task_target('clean the bowl') is None


def test_surface_named_drawer_is_observed_without_guessing_open():
    s=ObservableSearch('put a mug in desk.')
    s.update('go to drawer 1','You arrive at drawer 1. On the drawer 1, you see a pen 1.', ['go to shelf 1'],0)
    action=s.action('On the drawer 1, you see a pen 1.',['go to shelf 1'])
    assert action['actions']==['go to shelf 1'] and 'drawer 1' in s.searched


def test_closed_container_is_not_marked_searched_before_actual_open_feedback():
    s=ObservableSearch('put a mug in desk.')
    s.update('go to cabinet 1','You arrive at cabinet 1. The cabinet 1 is closed.',['open cabinet 1','go to shelf 1'],0)
    assert 'cabinet 1' not in s.searched
    assert s.action('closed',['open cabinet 1','go to shelf 1'])['actions']==['open cabinet 1']
    s.update('open cabinet 1','You open the cabinet 1. In it, you see nothing.',['go to shelf 1'],1)
    assert 'cabinet 1' in s.searched


def test_target_acquisition_requires_fresh_exact_command_and_lamp_never_picked_up():
    s=ObservableSearch('look at bowl under the desklamp.',acquire=True)
    s.update('go to shelf 1','You arrive at shelf 1. On the shelf 1, you see a bowl 1, and a desklamp 1.',
        ['take bowl 1 from shelf 1'],0)
    assert s.action('visible',['take bowl 1 from shelf 1'])['actions']==['take bowl 1 from shelf 1']
    s.update('take bowl 1 from shelf 1','You pick up the bowl 1 from the shelf 1.',['use desklamp 1'],1)
    assert s.action('holding',['use desklamp 1']) is None


def test_cached_object_location_is_reverified_and_inventory_can_revoke_holding():
    s=ObservableSearch('put a mug in desk.',acquire=True)
    s.update('go to shelf 1','You arrive at shelf 1. On the shelf 1, you see a mug 1.',['go to desk 1'],0)
    s.update('go to desk 1','You arrive at desk 1. On the desk 1, you see nothing.',['go to shelf 1'],1)
    action=s.action('empty',['go to shelf 1'])
    assert action['actions']==['go to shelf 1'] and action['current_object_guaranteed'] is False
    s.update('go to shelf 1','You arrive at shelf 1. On the shelf 1, you see nothing.',['go to desk 1'],2)
    assert 'mug 1' not in s.objects
    s.update('take mug 1 from shelf 1','You pick up the mug 1 from the shelf 1.',['inventory'],3)
    s.update('inventory','You are not carrying anything.',['go to desk 1'],4)
    assert s.held is None


def test_branch_objective_charges_call_cost_once():
    assert abs(utility({'success':True,'model_calls':2,'suffix_steps':5})-.91)<1e-9
    rows=[{'features':[1]+[0]*(len(FEATURES)-1),'utilities':[.9,.7],'group':str(i)} for i in range(4)]
    model=fit(rows);prediction=predict(model,rows[0]['features'])
    assert np.isfinite(model['weights']).all()
    assert not prediction['terminate'] and prediction['advantage']<0


class PublicEnv:
    tasks=['put a mug in desk.']
    def __init__(self):
        self.envs=self;self.commands=['go to shelf 1'];self.actions=[]
    @property
    def get_admissible_commands(self):return [self.commands]
    def reset(self,_):return {'anchor':['Room with shelf 1 and desk 1.']},[{}]
    def step(self,wrapped):
        action=re.search(r'<action>(.*?)</action>',wrapped[0])[1]
        assert action in self.commands;self.actions.append(action);done=False
        if action=='go to shelf 1':
            self.commands=['take mug 1 from shelf 1','go to desk 1'];feedback='You arrive at shelf 1. On the shelf 1, you see a mug 1.'
        elif action=='take mug 1 from shelf 1':
            self.commands=['go to desk 1'];feedback='You pick up the mug 1 from the shelf 1.'
        elif action=='go to desk 1':
            self.commands=['move mug 1 to desk 1'];feedback='You arrive at desk 1. On the desk 1, you see nothing.'
        else:
            self.commands=[];feedback='You move the mug 1 to the desk 1.';done=True
        return {'anchor':[feedback]},[float(done)],[done],[{'won':done}]


def test_workflow_reports_program_actions_and_counts_same_environment_budget(tmp_path):
    game=tmp_path/'game';game.write_text('public fixture')
    class Planner:
        def before_call(self):pass
        def __call__(self,prompt):
            return {'response':'<thought>Deliver.</thought><actions>["go to desk 1", "move mug 1 to desk 1"]</actions>',
                'usage':{'prompt_tokens':0,'completion_tokens':0}}
    cfg={'settings':asdict(Settings(max_calls=1,max_steps=4)), 'task_skill':'Use current public commands.',
        'arms':{'workflow':{'workflow':{'search':True,'acquire':True,'priority':'surface_first','max_actions':4},'learned_value':False}}}
    result=run_episode(PublicEnv(),{'id':'fixture','gamefile':str(game)},'workflow',cfg,Planner(),tmp_path/'out')
    assert result['success'] and result['model_calls']==1 and result['environment_steps']==4
    assert result['compiled_search_actions']==2 and result['invalid_actions']==0
