"""Frozen cross-model comparison; no selection or tuning on these test tasks."""
import copy
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlparse
import os
sys.path.insert(0,str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))
from experiments import controller_research as base
from experiments.puzzle_pilot import Settings,save
from dynamic_skill.campaign_budget import CampaignBudget

ROOT=Path('results/qwen_model_pair_20260929')
MODELS=['qwen3.7-plus','qwen3.6-35b-a3b']
POLICIES=['react','fixed','interrupt','memory_adaptive3']
DEADLINE=1790658900.0  # 2026-09-29 05:15 UTC; finite new comparison authorization

def configure(model):
    base.ROOT=ROOT/model
    base.AUTH='qwen_pair_20260929_'+model
    base.DEADLINE=DEADLINE
    cfg=json.loads((base.ROOT/'config.json').read_text())
    base.settings=lambda:Settings(**cfg['settings'])
    base.check_config(cfg)
    return cfg

def prepare():
    original=json.loads(Path('results/controller_tw_20260929/config.json').read_text())
    assert not ROOT.exists()
    assert time.time()<DEADLINE
    with (base.LEDGER.parent/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        b=CampaignBudget(base.LEDGER,deadline=0); d=b.read()
        assert not any(r['status']=='pending' for r in d['requests'])
        assert not d['policy_violation'] and time.time()>=d['cooldown_until']
        unknown=sum(r['status']=='unknown_usage' for r in d['requests'])
        for model in MODELS:
            cfg=copy.deepcopy(original); cfg.pop('hash')
            cfg.update(evaluation_only=True,fixed_policies=POLICIES,
                       source_config_sha=original['hash'],deadline_unix=DEADLINE,max_new_requests=256)
            cfg['settings']['model']=model
            cfg['splits']['dev']=[]; cfg['splits']['validation']=[]
            cfg['hash']=base.sha(cfg)
            save(ROOT/model/'config.json',cfg)
            d['qwen_pair_20260929_'+model]={'config_sha':cfg['hash'],'initial_unknown':unknown,
                'at_unix':time.time(),'previous_requests':len(d['requests']),
                'previous_effective_usd':b.effective_spend(d),'old_deadline':d['deadline_unix'],
                'old_max_requests':d['max_requests'],'old_limit_usd':d['limit_usd'],
                'old_max_unknown':d['max_unknown_requests'],
                'reason':'User requests exact two Qwen models; frozen 4 tasks x 4 policies each, cumulative $30.'}
        d.update(limit_usd=30.0,deadline_unix=DEADLINE,max_requests=len(d['requests'])+256,
                 max_unknown_requests=unknown+1)
        b.write(d)
        save(ROOT/'manifest.json',{'models':MODELS,'policies':POLICIES,'episodes':32,
             'max_new_requests':256,'deadline':DEADLINE,'initial_requests':len(d['requests']),
             'initial_effective_usd':b.effective_spend(d),'initial_unknown':unknown,
             'enable_thinking':False,'source':original['hash'],
             'note':'Candidate frozen from GPT study, not adopted. Cross-model test only; no new validation or optimization.'})
        print(json.dumps({'prepared':True,'effective_usd':b.effective_spend(d),'episodes':32}))

def advance():
    proxy=urlparse(os.environ.get('HTTPS_PROXY',''))
    assert proxy.hostname=='127.0.0.1' and proxy.port==17890
    assert time.time()<DEADLINE
    manifest=json.loads((ROOT/'manifest.json').read_text())
    d=json.loads(base.LEDGER.read_text())
    assert not d['policy_violation']
    assert sum(r['status']=='unknown_usage' for r in d['requests'])==manifest['initial_unknown']
    for policy in POLICIES:
        for index in range(4):
            for model in MODELS:
                cfg=configure(model)
                out=base.ROOT/'test'/policy/cfg['splits']['test'][index]['id']
                if out.exists():
                    assert (out/'result.json').exists(),'incomplete episode requires audit'
                    row=json.loads((out/'result.json').read_text())
                else:
                    base.run(['--split','test','--policy',policy,'--index',str(index)])
                    row=json.loads((out/'result.json').read_text())
                    if row['status'] not in ['success','format_error','model_call_limit','step_limit']:
                        raise RuntimeError('Paid run halted: '+row['status'])
                    return
                assert row['status'] in ['success','format_error','model_call_limit','step_limit']
    save(ROOT/'complete.json',{'status':'complete','episodes':32,'time':time.time()})
    print(json.dumps({'status':'complete','episodes':32}))

if __name__=='__main__':
    assert sys.argv[1:]==['prepare']
    prepare()
