"""Auditable continuation of a batch stopped during training, before optimizer/selection.

Only clean, already completed training trajectories are reused. An interrupted
training task is explicitly rescheduled once in the new frozen batch, retaining
its failed attempt and charge. Held-out assignments are inherited only if the
parent had clean exposure audit and never entered any evaluation stage.
"""
import hashlib
import json
from pathlib import Path
import shutil

from experiments.common.puzzle_pilot import save
from .engine import CLEAN
from .schema import Genome


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_parent(cfg):
    root = Path(cfg['resume_parent']).resolve()
    path = root / 'config.json'
    if digest(path) != cfg['resume_parent_config_sha256']:
        raise ValueError('Resume parent config changed')
    parent = json.loads(path.read_text())
    if not parent.get('exposure_audit', {}).get('independent_evaluation_claim_allowed'):
        raise ValueError('Cannot inherit previously exposed evaluation')
    if json.loads((root/'run/status.json').read_text())['status'] != 'stopped':
        raise ValueError('Resume is only for stopped training batches')
    if (root/'run/test').exists() or any(p.exists() for g in (root/'run/generations').glob('*')
                                      for p in (g/'selection',g/'confirmation',g/'candidates')):
        raise ValueError('Parent already entered optimizer/selection/evaluation')
    for key in ('splits','settings','initial','generations','candidates_per_generation',
                'candidate_scopes','branch_feedback','optimizer_max_tokens','optimizer'):
        a,b = cfg[key],parent[key]
        if key == 'settings' and cfg.get('resume_transport_timeout_change'):
            a={k:v for k,v in a.items() if k!='request_timeout'}
            b={k:v for k,v in b.items() if k!='request_timeout'}
        if json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True):
            raise ValueError('Resume must preserve registered algorithm/data: ' + key)
    return root


def cache_manifest(cfg):
    root = validate_training_source(cfg) if cfg.get('training_source') else validate_parent(cfg)
    parent = Genome.from_dict(cfg['initial'])
    tasks = {t['id']:t for t in cfg['splits']['train']}
    entries = {}
    for path in (root/'run/generations/000/execution_feedback/predictions').glob('*/result.json'):
        row = json.loads(path.read_text())
        if row['status'] not in CLEAN:
            continue
        if row['task_id'] not in tasks or row['genome_hash'] != parent.hash or row['task_id'] in entries:
            raise ValueError('Training cache task/genome mismatch')
        initial = json.loads((path.parent/'initial.json').read_text())
        if Path(initial['gamefile']).resolve() != Path(tasks[row['task_id']]['gamefile']).resolve():
            raise ValueError('Training cache scene mismatch')
        entries[row['task_id']] = {'result_path':str(path.resolve()),
            'file_hashes':{str(p.resolve()):digest(p) for p in sorted(path.parent.glob('*.json'))}}
    return entries


def reuse_training(cache, task, genome, directory):
    if task['id'] not in cache or directory.name != 'execution_feedback':
        return None
    entry=cache[task['id']]
    for path,sha in entry['file_hashes'].items():
        if digest(path) != sha:
            raise ValueError('Training cache changed since freezing')
    source=Path(entry['result_path']);row=json.loads(source.read_text())
    if row['genome_hash'] != genome.hash or row['status'] not in CLEAN:
        raise ValueError('Incompatible training reuse')
    target=directory/'predictions'/row['id']
    shutil.copytree(source.parent,target)
    row['reuse_source']=str(source)
    row['new_paid_requests']=0
    save(target/'result.json',row)
    save(target/'reuse.json',{'source':str(source),'source_sha256':entry['file_hashes'][str(source)],
                            'historical_cost_usd':row['cost_usd'],'new_cost_usd':0})
    print('REUSE completed training',task['id'],row['status'],flush=True)
    return row


def branch_cache_manifest(cfg):
    root=validate_training_source(cfg) if cfg.get('training_source') else validate_parent(cfg)
    base=root/'run/generations/000/branch_feedback'
    if not base.exists():return {}
    cases={c['id']:c for c in json.loads((base/'cases.json').read_text())}
    entries={}
    for path in base.glob('*/*/result.json'):
        row=json.loads(path.read_text())
        if row['status'] not in CLEAN:continue
        key=row['case_id']+'/'+row['arm']
        case=cases[row['case_id']]
        entries[key]={'result_path':str(path.resolve()),
            'case_signature':{k:case[k] for k in ('id','step','tail','kind')},
            'file_hashes':{str(p.resolve()):digest(p) for p in sorted(path.parent.glob('*.json'))}}
    return entries


def reuse_branch(cache,case,arm,out):
    entry=cache.get(case['id']+'/'+arm)
    if entry is None:return None
    if entry['case_signature'] != {k:case[k] for k in ('id','step','tail','kind')}:
        raise ValueError('Branch cache checkpoint definition changed')
    for p,sha in entry['file_hashes'].items():
        if digest(p)!=sha:raise ValueError('Branch cache changed')
    source=Path(entry['result_path']);row=json.loads(source.read_text())
    if row['status'] not in CLEAN:raise ValueError('Cannot reuse infrastructure-failed branch')
    shutil.copytree(source.parent,out)
    row.update(reuse_source=str(source),new_paid_requests=0)
    save(out/'result.json',row)
    save(out/'reuse.json',{'source':str(source),'historical_cost_usd':row['cost_usd'],'new_cost_usd':0})
    print('REUSE completed branch',case['id'],arm,row['status'],flush=True)
    return row


def reservation_ancestors(cfg):
    roots=[]
    while cfg.get('resume_parent'):
        root=validate_parent(cfg)
        if root in roots:raise ValueError('Cyclic resume ancestry')
        roots.append(root)
        cfg=json.loads((root/'config.json').read_text())
    return roots


def validate_training_source(cfg):
    if cfg.get('resume_parent'):raise ValueError('Use training-only import OR continuation')
    root=Path(cfg['training_source']).resolve()
    if digest(root/'config.json') != cfg['training_source_config_sha256']:
        raise ValueError('Training source config changed')
    source=json.loads((root/'config.json').read_text())
    if json.dumps(source['initial'],sort_keys=True)!=json.dumps(cfg['initial'],sort_keys=True):
        raise ValueError('Training source genome differs')
    if source['splits']['train'] != cfg['splits']['train']:
        raise ValueError('Training source tasks differ')
    for key in ('model','base_url','max_calls','max_steps','max_tokens','temperature'):
        if source['settings'][key]!=cfg['settings'][key]:raise ValueError('Training source execution setting differs')
    return root


def unused_evaluation_reservation(cfg):
    root=Path(cfg['evaluation_reservation_source']).resolve()
    if digest(root/'config.json')!=cfg['evaluation_reservation_config_sha256']:
        raise ValueError('Reserved evaluation config changed')
    source=json.loads((root/'config.json').read_text())
    if not source['exposure_audit']['independent_evaluation_claim_allowed']:
        raise ValueError('Evaluation source was previously exposed')
    for key in ('selection','confirmation','test'):
        if source['splits'][key]!=cfg['splits'][key]:raise ValueError('Reserved evaluation tasks differ')
    run=root/'run'
    forbidden=['generations/*/selection','generations/*/confirmation','generations/*/candidates/*/selection',
               'test/initial','test/skill_only','test/policy_only','test/evolved']
    if any(list(run.glob(pattern)) for pattern in forbidden):
        raise ValueError('Reserved evaluation was already attempted')
    if not (run/'test/skipped.json').exists():raise ValueError('Require explicit untouched test record')
    return root
