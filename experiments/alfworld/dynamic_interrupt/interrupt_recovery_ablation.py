"""Prepare A/B/C/D offline. Paid advancement additionally requires NEW ledger authorization.

This module never grants itself authorization or extends the shared budget/time caps.
"""
import argparse,fcntl,hashlib,json,time
from pathlib import Path
from dataclasses import asdict
from experiments.alfworld_campaign import CAMPAIGN
from experiments.puzzle_pilot import Settings,save
from dynamic_skill.control_only_evolution import ControlPolicy,run_control_episode
from dynamic_skill.micro_loop_protocol import SYSTEM
from dynamic_skill.campaign_budget import CampaignBudget,BudgetPlanner

ROOT=Path('results/interrupt_recovery_v3')
AUTH='interrupt_recovery_v3_authorization'
ARMS={'A':{'dynamic':False,'history_evidence':False,'state_check':False},
      'B':{'dynamic':True,'history_evidence':False,'state_check':False},
      'C':{'dynamic':True,'history_evidence':True,'state_check':False},
      'D':{'dynamic':True,'history_evidence':True,'state_check':True}}
SOURCES=['dynamic_skill/acquisition_check.py','dynamic_skill/control_only_evolution.py',
         'dynamic_skill/alfworld_recovery.py','dynamic_skill/skill_executor.py',
         'dynamic_skill/checker.py','dynamic_skill/interrupter.py','dynamic_skill/micro_loop_protocol.py',
         'experiments/alfworld_campaign.py','experiments/puzzle_pilot.py',
         'dynamic_skill/campaign_budget.py','experiments/interrupt_recovery_ablation.py']
def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def config_hash(cfg):return hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()

def prepare():
    src=read(Path('results/micro_evolution_qwen36_20260929/control_parameter_trial/config.json'))
    # Reuse the two prepared but unattempted validation tasks; do not search by outcome.
    cfg={'protocol':'interrupt-recovery-v3','arms':ARMS,'policy':asdict(ControlPolicy(recovery_steps=0)),
         'settings':asdict(Settings(model='qwen3.6-35b-a3b',max_calls=12,max_steps=50,
            max_tokens=384,temperature=0,request_timeout=45,episode_seconds=210,action_only=False)),
         'task_skill':src['task_skill'],'task_skill_sha256':hashlib.sha256(src['task_skill'].encode()).hexdigest(),
         'tasks':src['validation'],'system_prompt':SYSTEM,'enable_thinking':False,
         'budget_ledger':str(CAMPAIGN/'budget.json'),'paid_authorized':False,
         'source_hashes':{p:digest(Path(p)) for p in SOURCES},
         'job_order':[[i,arm] for i in range(len(src['validation'])) for arm in 'ABCD'],
         'metrics':['success','model_calls','prompt_tokens','completion_tokens','environment_steps',
                    'interrupt_count','anomaly_replans','history_evidence_calls','semantic_pauses',
                    'unverified_recoveries','verified_recoveries','false_interrupt_count'],
         'notes':'Offline preparation only. Fixed recovery_steps=0 in ALL arms; no simultaneous tuning. '
                 'No mechanism exposure means inconclusive, not a gain. False interruption needs labeled review.'}
    for task in cfg['tasks']:assert digest(Path(task['gamefile']))==task['game_sha256']
    if (ROOT/'config.json').exists():
        old=read(ROOT/'config.json')
        if old!=cfg:
            assert not (ROOT/'online').exists(),'Never modify a started experiment'
            assert {k:v for k,v in old.items() if k!='source_hashes'}=={k:v for k,v in cfg.items() if k!='source_hashes'},'Protocol change requires new experiment'
            save(ROOT/('config_history_'+config_hash(old)+'.json'),old)
            save(ROOT/'config.json',cfg)  # Only our offline unrun source pin may refresh; retain old version.
    else:save(ROOT/'config.json',cfg)
    print(json.dumps({'prepared':True,'paid_calls':0,'tasks':len(cfg['tasks']),'arms':4,'config_sha256':config_hash(cfg)}))

def summarize():
    rows=[read(p) for p in ROOT.glob('online/*/predictions/*/result.json')]
    review_path=ROOT/'interrupt_review.json'
    review=read(review_path) if review_path.exists() else {}
    for p in ROOT.glob('online/*/predictions/*/conversation.json'):
        for step in read(p):
            if step.get('interruption_reason'):
                key=str(p.parent.relative_to(ROOT))+':'+str(step['step'])
                review.setdefault(key,{'arm':p.parent.parent.parent.name,'step':step['step'],
                    'actual_cut':bool(step['actual_discarded_actions']),'reason':step['interruption_reason'],
                    'check':step['state_check'],'label':'unknown','evidence':None,
                    'instructions':'Label false_interrupt only with observable/counterfactual evidence that the blocked tail was safe; otherwise unknown. Never use the next episode success alone.'})
    save(review_path,review)
    report={'paid_experiments_run':bool(rows),'arms':{}}
    for arm in ARMS:
        group=[r for r in rows if r['arm']==arm]
        report['arms'][arm]={'attempted':len(group),'successes':sum(r['success'] for r in group),
            'success_rate':sum(r['success'] for r in group)/len(group) if group else None,
            'statuses':{s:sum(r['status']==s for r in group) for s in {r['status'] for r in group}},
            **{k:sum(r.get(k,0) for r in group) for k in ['model_calls','prompt_tokens','completion_tokens',
                'environment_steps','interrupt_count','anomaly_replans','history_evidence_calls','semantic_pauses',
                'state_check_exposures','single_step_recovery_calls','recovery_counter_end_events','cost_usd','latency_total_seconds']},
            'verified_recoveries':sum(f['resolution']=='verified_resolved' for r in group for f in r['failure_resolution']),
            'unverified_recoveries':sum(f['resolution']=='unverified' for r in group for f in r['failure_resolution']),
            'false_interrupt_count':None,'false_interrupt_note':'Unlabeled, requires observable/counterfactual adjudication; never inferred as zero.'}
        labels=[x for x in review.values() if x['arm']==arm and x['actual_cut']]
        judged=[x for x in labels if x['label'] in ('false_interrupt','justified') and x.get('evidence')]
        report['arms'][arm].update(adjudicated_cuts=len(judged),unadjudicated_cuts=len(labels)-len(judged),
            false_interrupt_count=sum(x['label']=='false_interrupt' for x in judged) if judged else None)
    save(ROOT/'summary.json',report);print(json.dumps(report))

def advance():
    cfg=read(ROOT/'config.json')
    with (CAMPAIGN/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        b=CampaignBudget(CAMPAIGN/'budget.json',deadline=0);d=b.read()
        auth=d.get(AUTH)
        if not auth or auth.get('config_sha256')!=config_hash(cfg):
            raise RuntimeError('Paid experiment NOT authorized: offline-only task; require a new explicit authorization bound to this config.')
        assert time.time()<auth['deadline_unix'] and len(d['requests'])<auth['max_requests']
        assert not any(r['status']=='pending' for r in d['requests'])
        assert not d['policy_violation'] and time.time()>=d['cooldown_until']
        assert d['limit_usd']<=45  # No budget expansion from a prepared config.
        for path,sha in cfg['source_hashes'].items():assert digest(Path(path))==sha
        for i,arm in cfg['job_order']:
            task=cfg['tasks'][i];mode='dynamic_interrupt' if ARMS[arm]['dynamic'] else 'fixed_chunk'
            out=ROOT/'online'/arm
            episode=out/'predictions'/(cfg['settings']['model']+'__'+task['id']+'__'+mode)
            if episode.exists():
                assert (episode/'result.json').exists(),'incomplete prior attempt: audit required'
                assert read(episode/'result.json')['status'] not in ['api_error','unknown_cost_stop','thinking_policy_stop']
                continue
            assert digest(Path(task['gamefile']))==task['game_sha256']
            from skillopt.envs.alfworld.rollout import build_alfworld_env
            from experiments.puzzle_pilot import ApiPlanner
            s=Settings(**cfg['settings']);env=build_alfworld_env(env_num=1,eval_dataset='eval_in_distribution',seed=42,is_train=False,specific_gamefiles=[task['gamefile']])
            api=ApiPlanner(s,system_prompt=SYSTEM);before=b.effective_spend(d)
            try:
                row=run_control_episode(env,task,s,out,BudgetPlanner(api,b,str(episode)),cfg['task_skill'],ControlPolicy(**cfg['policy']),**cfg['arms'][arm])
            finally:api.close();env.close()
            trace=read(episode/'conversation.json') if (episode/'conversation.json').exists() else []
            row.update(arm=arm,cost_usd=b.effective_spend(b.read())-before,
                       semantic_pauses=sum(r.get('semantic_pause',False) for r in trace))
            save(episode/'result.json',row);summarize();return
        summarize()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--summary',action='store_true');a=p.parse_args()
    if a.prepare:prepare()
    if a.summary:summarize()
