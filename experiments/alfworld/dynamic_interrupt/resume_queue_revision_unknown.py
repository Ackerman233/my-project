"""Audited remaining jobs only after one timeout; never retry the failed arm."""
from pathlib import Path
import fcntl
import json
import os
import shutil
import time
from experiments import queue_revision_unknown_trial as trial
from experiments import queue_revision_transfer_trial as base
from experiments.interrupt_recovery_ablation import read,digest,config_hash
from experiments.puzzle_pilot import ApiPlanner,Settings,save
from dynamic_skill.campaign_budget import BudgetPlanner

AUTH='queue_revision_unknown_remaining_20261007_authorization'


def run():
    trial.configure()
    root=trial.ROOT
    cfg,frozen=read(root/'config.json'),read(root/'cases.json')
    with (base.CAMPAIGN/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        base.AUTH=AUTH
        budget=base.TransferBudget(base.CAMPAIGN/'budget.json',deadline=0)
        data=budget.read()
        assert not data['policy_violation'] and not any(r['status']=='pending' for r in data['requests'])
        original=data[trial.AUTH]
        prior=data['requests'][original['start_request_count']:]
        failures=[r for r in prior if r['status']!='returned_usage' or r.get('error_type')]
        assert len(failures)==1 and failures[0]['status']=='unknown_usage' and failures[0]['error_type']=='APITimeoutError'
        assert failures[0]['budget_charge_usd']==0.5 and data['consecutive_errors']==1
        assert time.time()>=data['cooldown_until'] and data['limit_usd']==85
        assert all(digest(Path(p))==sha for p,sha in cfg['source_hashes'].items())
        assert all(digest(Path(p))==sha for p,sha in cfg['source_configs'].items())
        remaining=[]
        for case_id,arm in frozen['job_order']:
            out=root/'online'/case_id/arm
            if out.exists():
                assert (out/'result.json').exists()
                assert read(out/'result.json')['status'] in ('success','api_error')
                continue
            remaining.append([case_id,arm])
        assert len(remaining)==2 and AUTH not in data
        manifest={'original_config_sha256':config_hash(cfg),'cases_sha256':digest(root/'cases.json'),
            'remaining_jobs':remaining,'failed_request_id_preserved':failures[0]['id'],
            'resume_source_sha256':digest(Path(__file__)),'no_retry_failed_arm':True,
            'note':'Complete only unattempted jobs after auditing one timeout. Keep unknown reservation and original batch/cumulative ceilings. No model/queue/prompt/selection changes.'}
        save(root/'remaining_manifest.json',manifest)
        shutil.copyfile(Path(__file__),root/'resume_source_snapshot.py')
        auth={'config_sha256':config_hash(cfg),'cases_sha256':digest(root/'cases.json'),
            'start_request_count':len(data['requests']),
            'max_requests':min(original['max_requests'],len(data['requests'])+len(remaining)*cfg['settings']['max_calls']),
            'deadline_unix':min(original['deadline_unix'],time.time()+900),
            'batch_spend_ceiling':original['batch_spend_ceiling'],
            'manifest_sha256':digest(root/'remaining_manifest.json'),
            'user_authorization':'Complete user-requested experiment autonomously; audited unattempted jobs only, no retry, no ceiling increase'}
        data[AUTH]=auth
        data.update(max_requests=auth['max_requests'],deadline_unix=auth['deadline_unix'])
        budget.write(data)
        os.environ.setdefault('TARGET_AZURE_OPENAI_API_KEY',Path('/opt/data/private/api-key').read_text().strip())
        for case_id,arm in remaining:
            assert digest(root/'remaining_manifest.json')==auth['manifest_sha256']
            assert all(digest(Path(p))==sha for p,sha in cfg['source_hashes'].items())
            case=next(c for c in frozen['cases'] if c['id']==case_id)
            out=root/'online'/case_id/arm
            env=base.environment(read(Path(case['source_trace']).parent/'initial.json')['gamefile'])
            api=ApiPlanner(Settings(**cfg['settings']),system_prompt=base.SYSTEM_SKILL)
            before=budget.effective_spend(budget.read())
            try:
                result=base.run_suffix(env,case,arm,cfg,BudgetPlanner(api,budget,str(out)),out)
            finally:
                api.close();env.close()
            result['cost_usd']=budget.effective_spend(budget.read())-before
            save(out/'result.json',result)
            base.summary()
            if result['status'] in base.HALTS:
                break
        save(root/'budget_after_remaining.json',budget.read())


if __name__=='__main__':
    run()
