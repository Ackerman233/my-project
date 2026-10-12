"""Explicit user restart; frozen scientific config, separate results and authorization."""
import fcntl
import json
from pathlib import Path
import sys
import time
from datetime import datetime, timezone
from experiments import controller_research as base
from dynamic_skill.campaign_budget import CampaignBudget
from experiments.puzzle_pilot import save

ROOT=Path('results/controller_resume_20260929')
ORIGINAL=Path('results/controller_research_20260928')
AUTH='controller_resume_authorization_20260929'
DEADLINE=datetime(2026,9,29,3,4,32,tzinfo=timezone.utc).timestamp()


def configure():
    base.ROOT=ROOT; base.AUTH=AUTH; base.DEADLINE=DEADLINE


def prepare():
    cfg=json.loads((ORIGINAL/'config.json').read_text()); base.check_config(cfg)
    prior=list(ORIGINAL.glob('*/*/*/result.json'))
    assert len(prior)==1
    old=json.loads(prior[0].read_text())
    assert old['status']=='api_error' and old['environment_steps']==0
    with (base.LEDGER.parent/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        b=CampaignBudget(base.LEDGER,deadline=0); d=b.read()
        if AUTH in d or ROOT.exists(): raise RuntimeError('restart already prepared')
        if any(r['status']=='pending' for r in d['requests']): raise RuntimeError('pending requests')
        if d['policy_violation'] or time.time()<d['cooldown_until']: raise RuntimeError('closed guard')
        if time.time()>=DEADLINE: raise RuntimeError('deadline expired')
        unknown=sum(r['status']=='unknown_usage' for r in d['requests'])
        auth={'config_sha':cfg['hash'],'initial_unknown':unknown,'at_unix':time.time(),
              'initial_requests':len(d['requests']),'initial_effective_usd':b.effective_spend(d),
              'old_deadline':d['deadline_unix'],'old_max_requests':d['max_requests'],
              'old_unknown_cap':d['max_unknown_requests'],'deadline_unix':DEADLINE,
              'reason':'User explicitly requested restart experiments after 3 successful diagnostics; original finite protocol only, cumulative $30.',
              'explicit_retry':{'task_id':old['task_id'],'policy':'react','split':'dev','prior_result':str(prior[0])}}
        d[AUTH]=auth; d['deadline_unix']=DEADLINE; d['max_requests']=len(d['requests'])+384
        d['max_unknown_requests']=unknown+1
        assert d['limit_usd']==30
        save(ROOT/'config.json',cfg); save(ROOT/'restart.json',auth); b.write(d)
        print(json.dumps({'prepared':True,'effective_usd':b.effective_spend(d),'deadline_unix':DEADLINE,'config_hash':cfg['hash']}))


def run():
    configure(); base.advance()


def audit():
    configure()
    from experiments import audit_controller_research as checker
    checker.ROOT=ROOT; checker.AUTH=AUTH
    checker.main()


if __name__=='__main__':
    configure()
    {'prepare':prepare,'evaluate':base.evaluate,'audit':audit}[sys.argv[1]]()
