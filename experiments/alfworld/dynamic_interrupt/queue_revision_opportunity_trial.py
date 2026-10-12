"""Separate public-opportunity stratum after zero exposure in path-only sampling."""
import argparse
import copy
import json
from pathlib import Path
import re
import shutil
from experiments import queue_revision_transfer_trial as base
from experiments.interrupt_recovery_ablation import read, digest, config_hash
from experiments.puzzle_pilot import save

ROOT = Path('results/queue_revision_opportunity_20261007')
AUTH = 'queue_revision_opportunity_20261007_authorization'


def configure():
    base.ROOT, base.AUTH = ROOT, AUTH


def prepare():
    configure()
    assert not (ROOT / 'config.json').exists(), 'Do not change frozen config'
    source = read(Path('results/queue_revision_transfer_20261007/config.json'))
    assert all(digest(Path(p)) == h for p,h in source['rule_hashes'].items())
    candidates = sorted(Path('/opt/data/private/data/alfworld/json_2.1.1/valid_seen').glob('*/*/game.tw-pddl'))
    used_scenes, used_layouts = set(), set()
    for p in Path('results').rglob('*.json'):
        if ROOT in p.parents or 'source_snapshot' in p.parts or p.name not in ('config.json','initial.json','tasks.json'):
            continue
        text = p.read_text()
        for game in candidates:
            name = game.parent.parent.name
            if str(game) in text:
                used_scenes.add(name)
                used_layouts.add(name.split('-')[-1])
    # Protocol registered in code before screening: at most 60 public probes,
    # two tasks per family, distinct selected layouts, no object-content ranking.
    screened, tasks, visited = [], [], set()
    for family in ('pick_and_place_simple','look_at_obj_in_light'):
        selected_objects = set()
        attempts = 0
        for game in candidates:
            name = game.parent.parent.name
            layout = name.split('-')[-1]
            if not name.startswith(family+'-') or name in used_scenes or layout in used_layouts or layout in visited:
                continue
            object_type = name.split('-')[1]
            if object_type in selected_objects:
                continue
            attempts += 1
            visited.add(layout)
            env = base.environment(str(game))
            try:
                obs,_ = env.reset({})
                initial = obs['anchor'][0]
                commands = list(env.envs.get_admissible_commands[0])
                action = next((a for a in commands if re.fullmatch(r'go to drawer [0-9]+',a)),None)
                row = {'gamefile': str(game), 'layout': layout, 'family':family,
                    'initial_sha256': config_hash({'observation':initial,'admissible':commands}),
                    'probe_action': action, 'surface':False}
                if action:
                    obs,_,dones,_=env.step(['<think></think><action>'+action+'</action>'])
                    admissible = list(env.envs.get_admissible_commands[0])
                    target = action[len('go to '):]
                    row.update(observation=obs['anchor'][0],admissible=admissible,
                        surface=bool(re.search(r'On the '+re.escape(target)+r', you see ',obs['anchor'][0]) and 'open '+target not in admissible and not dones[0]))
                screened.append(row)
            finally:
                env.close()
            if row['surface']:
                selected_objects.add(object_type)
                tasks.append({'id':'opportunity-'+family+'-'+str(len(selected_objects)),
                    'gamefile':str(game),'game_sha256':digest(game),'benchmark':'alfworld',
                    'family':family,'object_type':object_type,'layout':layout})
            if len(selected_objects)==2 or attempts>=30:
                break
    save(ROOT/'public_screen.json', {'paid_calls':0,'max_probes_per_family':30,
        'selection':'first public drawer surface, excluding all historically configured scene layouts; contents ignored',
        'screened':screened,'tasks':tasks})
    cfg=copy.deepcopy(source)
    cfg.update(protocol='frozen-queue-revision-public-opportunity-stratum-v1',tasks=tasks,
        public_screen_sha256=digest(ROOT/'public_screen.json'),
        collection_settings={**source['collection_settings'],'max_calls':6,'max_steps':24},
        limits={'batch_usd':6,'max_requests':len(tasks)*21,'seconds':7200},
        note='Adaptive follow-up stratum after zero falsified heads in path-only six-task sampling. Candidate tasks screened only by public drawer-surface feedback; object contents and suffix outcomes ignored. All scene layout IDs excluded if historically configured; selected layouts distinct. Frozen rules, prompt, model, suffix budgets unchanged. Not an estimate of natural exposure or broad benchmark gain.')
    paths=sorted(set(cfg['source_hashes'])|{'experiments/queue_revision_opportunity_trial.py'})
    cfg['source_hashes']={p:digest(Path(p)) for p in paths}
    save(ROOT/'config.json',cfg)
    for p in paths:
        dst=ROOT/'source_snapshot'/p
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,dst)
    print('PREPARED_OPPORTUNITY',len(tasks),'public_probes',len(screened),'config',config_hash(cfg),flush=True)


def advance():
    configure()
    cfg=read(ROOT/'config.json')
    assert digest(ROOT/'public_screen.json')==cfg['public_screen_sha256']
    base.advance()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','authorize','advance','summary'])
    args=parser.parse_args()
    configure()
    {'prepare':prepare,'authorize':base.authorize,'advance':advance,'summary':base.summary}[args.action]()
