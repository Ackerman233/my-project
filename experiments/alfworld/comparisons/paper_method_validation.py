"""Independent frozen test of the selected workflow and fitted termination value."""
import argparse
import copy
from pathlib import Path
from experiments.paper_method_trial import DEV,TEST,MODEL,DESIGN,WORKFLOWS,sources,advance,summary
from experiments.unknown_independent_validation import used_scene_names
from experiments.interrupt_recovery_ablation import read,digest
from experiments.bounded_research95 import freeze,authorize
from experiments.puzzle_pilot import save


def prepare(root=TEST,confirmation=False):
    dev=read(DEV/'summary.json');assert dev['complete']
    selected=read(DEV/'selected_workflow.json')
    if confirmation:
        first=read(TEST/'summary.json');assert first['complete']
    candidates=sorted(Path('/opt/data/private/data/alfworld/json_2.1.1/valid_unseen').glob('*/*/game.tw-pddl'))
    history=list(Path('results').rglob('*.json'))
    used=used_scene_names(history,candidates);tasks=[]
    for family in ('pick_and_place_simple','look_at_obj_in_light'):
        objects=set()
        for game in candidates:
            name=game.parent.parent.name
            if not name.startswith(family+'-') or name in used or name.split('-')[1] in objects:continue
            obj=name.split('-')[1];objects.add(obj)
            tasks.append({'id':('confirm' if confirmation else 'paper-test')+'-'+family+'-'+str(len(objects)),
                'gamefile':str(game),'game_sha256':digest(game),'benchmark':'alfworld','family':family,'object_type':obj})
            if len(objects)==2:break
        assert len(objects)==2
    old=read(DEV/'config.json');arms={'rule':{'workflow':WORKFLOWS['W0_rule'],'learned_value':False},
        'value':{'workflow':WORKFLOWS['W0_rule'],'learned_value':True},
        'workflow':{'workflow':selected['workflow'],'learned_value':False}}
    if confirmation:arms.pop('value')
    cfg=copy.deepcopy(old)
    cfg.update(protocol='paper-inspired-independent-test-v1',stage='confirmation' if confirmation else 'test',tasks=tasks,
        settings={**old['settings'],'max_calls':8,'max_steps':40},arms=arms,
        job_order=[[t['id'],a] for i,t in enumerate(tasks) for a in list(arms)[i%len(arms):]+list(arms)[:i%len(arms)]],
        limits={'batch_usd':6 if confirmation else 10,'max_requests':64 if confirmation else 96,'max_new_unknown':2,'seconds':7200},
        source_hashes={**sources(),'experiments/paper_method_validation.py':digest(Path(__file__))},
        input_hashes={**old['input_hashes'],str(DEV/'selected_workflow.json'):digest(DEV/'selected_workflow.json'),str(DEV/'summary.json'):digest(DEV/'summary.json')},
        gain_gate='Completion wins without losses against rule, or equal paired successes with fewer total model calls and no higher inference dollars. At least two clean paired tasks required. All tasks included, infrastructure-invalid pairs reported separately. Confirmation reported independently.',
        note='New physical scenes selected only by path, family and object diversity before any execution. Same model skill, primitive-step/model-call budgets. Workflow program primitives explicitly counted; no cross-environment generalization claim.')
    freeze(root,cfg)
    save(root/'history_exclusion.json',{'used_scene_names':sorted(used),'selected_scene_names':[Path(t['gamefile']).parent.parent.name for t in tasks],
        'historical_files':{str(p):digest(p) for p in history if 'source_snapshot' not in p.parts and p.name in ('config.json','initial.json','tasks.json')}})
    print('PREPARED',root,[(t['family'],t['object_type']) for t in tasks],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','authorize','advance','summary','prepare-confirm','authorize-confirm','advance-confirm','summary-confirm'])
    action=parser.parse_args().action;confirm='confirm' in action
    root=Path('results/paper_methods_unseen_confirmation_20261007') if confirm else TEST
    if action.startswith('prepare'):prepare(root,confirm)
    elif action.startswith('authorize'):authorize(root)
    elif action.startswith('advance'):advance(root)
    else:summary(root)
