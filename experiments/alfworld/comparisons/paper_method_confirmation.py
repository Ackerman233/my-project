"""Use every remaining never-used scene in the two ALFWorld unseen families."""
import argparse
import copy
from pathlib import Path
from experiments.paper_method_trial import TEST,DEV,sources,advance,WORKFLOWS
from experiments.unknown_independent_validation import used_scene_names
from experiments.interrupt_recovery_ablation import read,digest
from experiments.bounded_research95 import freeze,authorize
from experiments.puzzle_pilot import save
ROOT=Path('results/paper_methods_unseen_confirmation_20261007')

def prepare():
    result=read(TEST/'summary.json');assert result['complete']
    audit=read(TEST/'validation_audit.json');assert audit['gain_vs_rule']['workflow']['task_gain_demonstrated']
    candidates=sorted(Path('/opt/data/private/data/alfworld/json_2.1.1/valid_unseen').glob('*/*/game.tw-pddl'))
    history=list(Path('results').rglob('*.json'));used=used_scene_names(history,candidates)
    tasks=[]
    for game in candidates:
        name=game.parent.parent.name
        if name in used:continue
        family=next((f for f in ('pick_and_place_simple','look_at_obj_in_light') if name.startswith(f+'-')),None)
        if not family:continue
        # One version per never-used physical scene, no performance or object filtering.
        if any(Path(t['gamefile']).parent.parent.name==name for t in tasks):continue
        tasks.append({'id':'paper-confirm-'+str(len(tasks)+1),'family':family,'object_type':name.split('-')[1],
            'gamefile':str(game),'game_sha256':digest(game),'benchmark':'alfworld'})
    assert len(tasks)==4, 'Expected three place and one light never-used scenes after first test'
    cfg=copy.deepcopy(read(TEST/'config.json'));cfg.update(stage='confirmation',protocol='paper-workflow-exhausted-unseen-confirmation-v1',tasks=tasks,
        arms={k:v for k,v in cfg['arms'].items() if k!='value'},
        job_order=[[t['id'],a] for i,t in enumerate(tasks) for a in (['rule','workflow'] if i%2==0 else ['workflow','rule'])],
        limits={'batch_usd':6,'max_requests':64,'max_new_unknown':2,'seconds':7200},
        source_hashes={**sources(),'experiments/paper_method_confirmation.py':digest(Path(__file__))},
        input_hashes={**cfg['input_hashes'],str(TEST/'summary.json'):digest(TEST/'summary.json'),str(TEST/'validation_audit.json'):digest(TEST/'validation_audit.json')},
        note='All remaining never-used valid_unseen physical scenes in these two families, selected before confirmation execution. Pool exhausted: three place plus one light, so unbalanced family composition reported explicitly. Frozen selected workflow, same 8-call/40-primitive budget. No task replacement or test-driven method tuning.')
    freeze(ROOT,cfg);save(ROOT/'history_exclusion.json',{'used_scene_names':sorted(used),'selected_scene_names':[Path(t['gamefile']).parent.parent.name for t in tasks],
        'historical_files':{str(p):digest(p) for p in history if 'source_snapshot' not in p.parts and p.name in ('config.json','initial.json','tasks.json')}})
    print('PREPARED_CONFIRM',[(t['family'],t['object_type']) for t in tasks],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','authorize','advance']);a=p.parse_args().action
    if a=='prepare':prepare()
    elif a=='authorize':authorize(ROOT)
    else:advance(ROOT)
