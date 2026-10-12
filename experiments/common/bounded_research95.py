"""Serial, frozen research batches under explicit cumulative USD95 authorization."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import time
from dynamic_skill.campaign_budget import CampaignBudget, CampaignHalt
from experiments.alfworld_campaign import CAMPAIGN
from experiments.interrupt_recovery_ablation import read,digest,config_hash
from experiments.puzzle_pilot import save


def auth_key(root):
    return Path(root).name+'_authorization'


def freeze(root,cfg):
    root=Path(root)
    assert not (root/'config.json').exists(),'Do not overwrite frozen config'
    save(root/'config.json',cfg)
    for path,sha in cfg['source_hashes'].items():
        assert digest(Path(path))==sha
        dst=root/'source_snapshot'/path
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dst)


def check_frozen(root,cfg):
    assert all(digest(Path(p))==sha and digest(Path(root)/'source_snapshot'/p)==sha for p,sha in cfg['source_hashes'].items())
    assert all(digest(Path(p))==sha for p,sha in cfg.get('input_hashes',{}).items())


class StageBudget(CampaignBudget):
    def __init__(self,root,*,path=None):
        self.root=Path(root)
        super().__init__(path or CAMPAIGN/'budget.json',deadline=0)
    def reserve(self,label,reservation_usd=None):
        data=self.read();auth=data[auth_key(self.root)]
        reserve=data['reservation_usd'] if reservation_usd is None else reservation_usd
        if data['limit_usd']!=95:
            raise CampaignHalt('budget_authorization_changed')
        if time.time()>=auth['deadline_unix']:
            raise CampaignHalt('deadline_stop')
        if len(data['requests'])>=auth['max_requests']:
            raise CampaignHalt('request_cap_stop')
        if self.effective_spend(data)+reserve>auth['batch_spend_ceiling']:
            raise CampaignHalt('budget_stop')
        if data['consecutive_errors']>=2:
            raise CampaignHalt('api_error_guard')
        recent=data['requests'][auth['start_request_count']:]
        if sum(r['status'] in ('pending','unknown_usage') for r in recent)>=auth['max_new_unknown']:
            raise CampaignHalt('unknown_cost_stop')
        return super().reserve(label,reservation_usd)


def authorize(root):
    root=Path(root);cfg=read(root/'config.json')
    with (CAMPAIGN/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        budget=StageBudget(root);data=budget.read()
        assert data['limit_usd']==95 and not data['policy_violation']
        assert not any(r['status']=='pending' for r in data['requests'])
        assert time.time()>=data['cooldown_until'] and data['consecutive_errors']<2
        assert auth_key(root) not in data
        check_frozen(root,cfg)
        save(root/'budget_before_authorization.json',data)
        auth={'config_sha256':config_hash(cfg),'start_request_count':len(data['requests']),
            'max_requests':len(data['requests'])+cfg['limits']['max_requests'],
            'deadline_unix':time.time()+cfg['limits'].get('seconds',7200),
            'batch_spend_ceiling':min(95,budget.effective_spend(data)+cfg['limits']['batch_usd']),
            'max_new_unknown':cfg['limits'].get('max_new_unknown',2),
            'initial_unknown':sum(r['status']=='unknown_usage' for r in data['requests']),
            'prior_unknown_cap':data['max_unknown_requests'],
            'user_authorization':'User explicitly authorized independent verification, then multiple paper methods based on idea Markdown if no gain, with cumulative USD95 ceiling. No automatic retry of failed jobs.'}
        data[auth_key(root)]=auth
        data.update(max_requests=auth['max_requests'],deadline_unix=auth['deadline_unix'],
            max_unknown_requests=auth['initial_unknown']+auth['max_new_unknown'])
        budget.write(data)
        print('AUTHORIZED',root.name,json.dumps(auth),flush=True)


def open_batch(root):
    cfg=read(Path(root)/'config.json');budget=StageBudget(root);data=budget.read()
    auth=data[auth_key(root)]
    assert auth['config_sha256']==config_hash(cfg)
    assert data['limit_usd']==95 and not data['policy_violation']
    assert not any(r['status']=='pending' for r in data['requests'])
    check_frozen(root,cfg)
    failures=[{k:r.get(k) for k in ('id','label','status','error_type','budget_charge_usd')}
        for r in data['requests'][auth['start_request_count']:] if r['status']!='returned_usage' or r.get('error_type')]
    if failures:
        save(Path(root)/'infrastructure_audit.json',{'failures':failures,'no_retry_failed_jobs':True,
            'conservative_reservations_preserved':True,'note':'Only untouched jobs may continue; consecutive-error, unknown-reserve, cooldown and dollar guards remain active.'})
    os.environ.setdefault('TARGET_AZURE_OPENAI_API_KEY',Path('/opt/data/private/api-key').read_text().strip())
    return cfg,budget


def batch_accounting(root):
    root=Path(root);budget=StageBudget(root);data=budget.read()
    auth=data[auth_key(root)]
    rows=[r for r in data['requests'][auth['start_request_count']:] if r['label'].startswith(str(root)+'/')]
    return {'requests':len(rows),'returned_usage':sum(r['status']=='returned_usage' for r in rows),
        'unknown_usage':sum(r['status']=='unknown_usage' for r in rows),
        'cost_usd':sum(r['budget_charge_usd'] for r in rows),'cumulative_requests':len(data['requests']),
        'cumulative_usd':budget.effective_spend(data),'limit_usd':data['limit_usd'],
        'pending':sum(r['status']=='pending' for r in data['requests'])}


def label_cost(root,label):
    budget=StageBudget(root);data=budget.read();auth=data[auth_key(root)]
    return sum(r['budget_charge_usd'] for r in data['requests'][auth['start_request_count']:] if r['label']==label)
