"""Read-only audit of a frozen joint evolution batch; emits no model requests."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common.bounded_campaign import batch_accounting
from experiments.alfworld.joint_evolution.schema import Genome


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def summarize(rows):
    return {'episodes':len(rows), 'reused_episodes':sum(bool(r.get('reuse_source')) for r in rows),
        'historical_reused_cost_usd':sum(r.get('cost_usd',0) for r in rows if r.get('reuse_source')), 'successes':sum(r['success'] for r in rows),
        'statuses':dict(Counter(r['status'] for r in rows)),
        **{k:sum(r.get(k,0) for r in rows) for k in ('model_calls','environment_steps','invalid_actions','cost_usd')},
        'actual_tail_interruptions':sum(r['actual_tail_interruptions'] for r in rows) if rows and all('actual_tail_interruptions' in r for r in rows) else None}


def audit(root):
    run=root/'run';cfg=read(root/'config.json');initial=Genome.from_dict(cfg['initial'])
    grouped={}
    for p in run.rglob('results.json'):
        grouped[str(p.parent.relative_to(run))]=summarize(read(p))
    for folder in run.rglob('predictions'):
        group=str(folder.parent.relative_to(run))
        if group not in grouped:
            rows=[read(p) for p in folder.glob('*/result.json')]
            grouped[group]={**summarize(rows), 'incomplete_group':True}
    branches=[read(p) for p in run.glob('generations/*/branch_feedback/*/comparison.json')]
    candidates=[]
    for p in run.glob('generations/*/candidates/*'):
        application=read(p/'optimizer/apply.json',{})
        candidates.append({'directory':str(p.relative_to(run)), 'optimizer':read(p/'optimizer/status.json'),
            'scope':application.get('component_scope'), 'changed_policy_fields':application.get('changed_policy_fields'),
            'policy_evidence_ids':application.get('policy_evidence_ids'),
            'training_input_ids':application.get('training_input_ids'),
            'model_citation_ids':application.get('model_citation_ids'),
            'citation_mode':application.get('citation_mode'), 'screen':read(p/'screen.json')})
    final=read(run/'best/genome.json')
    result={'status':read(run/'status.json'), 'accounting':batch_accounting(root), 'groups':grouped,
        'scheduled_split_sizes':{k:len(v[0]) if k=='confirmation' else len(v) for k,v in cfg['splits'].items()},
        'branch_feedback':{'pairs':len(branches),'valid_pairs':sum(b['valid_pair'] for b in branches),
            'labels':dict(Counter(b['label'] for b in branches)), 'kinds':dict(Counter(b['kind'] for b in branches)),
            'arms':{a:summarize([b['arms'][a] for b in branches]) for a in ('continue_tail','interrupt_replan')}},
        'candidates':candidates,'history':read(run/'history.json',[]), 'final_comparison':read(run/'test/comparison.json'),
        'accepted_policy_changed':bool(final and Genome.from_dict(final).policy!=initial.policy),
        'transport_retries':len(list((root/'transport').glob('*.retry.json'))),
        'exposure_audit':cfg.get('exposure_audit'), 'no_new_paid_requests_by_audit':True}
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    args=parser.parse_args();root=args.root.resolve();result=audit(root)
    (root/'validation_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
