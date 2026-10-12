"""One post-connectivity-recovery continuation under the existing shared guard."""
import json
from pathlib import Path
from experiments import resume_brief_protocol_pair as base
from experiments.cold_start_gate import gate
from experiments.puzzle_pilot import save

ROOT=Path('results/brief_protocol_recovered_20260928')
AUTH='brief_protocol_recovered_authorization'
PRIOR=Path('results/brief_protocol_resume_20260928')


def plan_jobs(config, previous):
    later=json.loads((PRIOR/'results.json').read_text())
    attempted={(r['task_id'],a) for source in (previous,later) for a,rows in source.items() for r in rows}
    missing=[(tid,a,'unattempted') for tid,a in config['run_order'] if (tid,a) not in attempted]
    target=('brief-sudoku-2','brief_reason','infra_retry_once')
    original=next(r for r in previous['brief_reason'] if r['task_id']==target[0])
    assert original['status']=='api_error'
    assert not any(r['task_id']==target[0] for r in later['brief_reason'])
    assert len(missing)==9
    return missing+[target]


def run():
    base.ROOT=ROOT; base.AUTH=AUTH; base.plan_jobs=plan_jobs
    base.run()


def audit():
    import experiments.audit_brief_protocol_pair as checker
    checker.ROOT=ROOT; checker.AUTH=AUTH; checker.main()
    read=lambda p:json.loads(p.read_text())
    cfg=read(base.ORIGINAL/'config.json'); newcfg=read(ROOT/'config.json')
    for key in ('settings','tasks','base_skills','base_hashes','tasks_sha256','run_order'):
        assert cfg[key]==newcfg[key]
    roots=[base.ORIGINAL,PRIOR,ROOT]
    first={a:{} for a in cfg['settings']}; adjusted={a:{} for a in cfg['settings']}
    retries=[]; paid=0; cost=0
    for root in roots:
        a=read(root/'audit.json'); paid+=a['paid_requests']; cost+=a['incremental_conservative_usd']
        for arm,rows in read(root/'results.json').items():
            for row in rows:
                tid=row['task_id']
                if tid in adjusted[arm]:
                    assert tid=='brief-sudoku-2' and arm=='brief_reason'
                    assert adjusted[arm][tid]['status']=='api_error' and row['attempt_kind']=='infra_retry_once'
                    retries.append({'task_id':tid,'arm':arm,'retry_status':row['status']})
                else:first[arm][tid]=row
                adjusted[arm][tid]=row
    arms={}
    for arm in adjusted:
        rows=list(adjusted[arm].values())
        arms[arm]={'raw_first_attempt_gate':gate(list(first[arm].values()),cfg['tasks']),
                   'retry_adjusted_gate':gate(rows,cfg['tasks']),
                   'rows':[{k:r[k] for k in ('task_id','status','success','model_calls','environment_steps','invalid_actions')} for r in rows]}
    out={'arms':arms,'explicit_retries':retries,'all_attempts_requests':paid,
         'all_attempts_guarded_cost_usd':cost,'cumulative_guarded_usd':read(ROOT/'audit.json')['cumulative_guarded_usd'],
         'note':'Infrastructure aborts retained; raw and one authorized retry-adjusted results separated.'}
    save(ROOT/'combined.json',out); print(json.dumps(out,indent=2))

if __name__=='__main__':audit()
