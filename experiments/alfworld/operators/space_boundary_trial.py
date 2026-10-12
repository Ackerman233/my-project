"""Bounded inference-only development diagnostic. No training or test-set claims."""
import argparse,fcntl,json,time
from dataclasses import asdict
from pathlib import Path
from experiments.interrupt_recovery_ablation import SOURCES,read,digest,config_hash
from experiments.alfworld_campaign import CAMPAIGN
from experiments.puzzle_pilot import Settings,ApiPlanner,save
from dynamic_skill.control_only_evolution import ControlPolicy,run_control_episode
from dynamic_skill.campaign_budget import CampaignBudget,BudgetPlanner,CampaignHalt

ROOT=Path('results/space_boundary_20261003');AUTH='space_boundary_20261003_authorization'
ARMS={'E_old':dict(boundary_guidance=False,discovery_interrupt=False),
      'F_prompt':dict(boundary_guidance=True,discovery_interrupt=False),
      'G_discovery':dict(boundary_guidance=True,discovery_interrupt=True)}

def prepare():
    old=read(Path('results/interrupt_recovery_v3/config.json'))
    cfg={'protocol':'space-inspired-inference-v1','model':'qwen3.6-35b-a3b',
         'settings':asdict(Settings(model='qwen3.6-35b-a3b',max_calls=20,max_steps=50,max_tokens=384,
                                   temperature=0,request_timeout=45,episode_seconds=210,action_only=False)),
         'arms':ARMS,'tasks':old['tasks'],'task_skill':old['task_skill'],
         'policy':asdict(ControlPolicy()),'enable_thinking':False,
         'common':dict(dynamic=True,history_evidence=True,state_check=True),
         'source_hashes':{p:digest(Path(p)) for p in SOURCES+['dynamic_skill/space_boundary.py','experiments/space_boundary_trial.py']},
         'job_order':[[i,a] for i in range(2) for a in ARMS],
         'note':'Previously observed development tasks, NOT independent validation. No SPACE RL/distillation. 20 calls in ALL arms, do not compare directly to prior 12-call success rates.'}
    if (ROOT/'config.json').exists():assert read(ROOT/'config.json')==cfg
    else:save(ROOT/'config.json',cfg)
    print('PREPARED',config_hash(cfg))

class TrialBudget(CampaignBudget):
    def reserve(self,label,reservation_usd=None):
        d=self.read();a=d[AUTH];reserve=d['reservation_usd'] if reservation_usd is None else reservation_usd
        if time.time()>=a['deadline_unix']:raise CampaignHalt('deadline_stop')
        if len(d['requests'])>=a['max_requests']:raise CampaignHalt('request_cap_stop')
        if self.effective_spend(d)+reserve>a['batch_spend_ceiling']:raise CampaignHalt('budget_stop')
        return super().reserve(label,reservation_usd)

def authorize():
    # Invocation only after explicit 2026-10-03 user authorization to raise cumulative ceiling to USD85.
    cfg=read(ROOT/'config.json')
    with (CAMPAIGN/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        b=TrialBudget(CAMPAIGN/'budget.json',deadline=0);d=b.read()
        assert AUTH not in d and not d['policy_violation'] and time.time()>=d['cooldown_until']
        assert not any(r['status']=='pending' for r in d['requests'])
        assert all(digest(Path(p))==h for p,h in cfg['source_hashes'].items())
        save(ROOT/'budget_before_authorization.json',d)
        a={'config_sha256':config_hash(cfg),'deadline_unix':time.time()+3600,
           'max_requests':len(d['requests'])+120,'start_request_count':len(d['requests']),
           'batch_spend_ceiling':min(85,b.effective_spend(d)+12),
           'user_authorization':'2026-10-03 cumulative USD85; first bounded 6-episode inference diagnostic'}
        d.update(limit_usd=85,deadline_unix=a['deadline_unix'],max_requests=a['max_requests']);d[AUTH]=a;b.write(d)
        print('AUTHORIZED',json.dumps(a))

def summary():
    rows=[read(p) for p in ROOT.glob('online/*/predictions/*/result.json')]
    result={}
    for arm in ARMS:
        group=[r for r in rows if r['arm']==arm]
        calls=sum(r['model_calls'] for r in group)
        result[arm]={'attempted':len(group),'successes':sum(r['success'] for r in group),
            'statuses':[r['status'] for r in group],
            **{k:sum(r.get(k,0) for r in group) for k in ['model_calls','environment_steps','planned_actions','multi_action_chunks',
              'interrupt_count','anomaly_replans','invalid_actions','history_evidence_calls','state_check_exposures','semantic_pauses','discovery_pauses','subskill_calls','cost_usd']},
            'mean_planned_chunk':sum(r['planned_actions'] for r in group)/calls if calls else None,
            'multi_action_rate':sum(r['multi_action_chunks'] for r in group)/calls if calls else None}
    save(ROOT/'summary.json',result);print(json.dumps(result))

def advance():
    cfg=read(ROOT/'config.json')
    with (CAMPAIGN/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        b=TrialBudget(CAMPAIGN/'budget.json',deadline=0);d=b.read();a=d[AUTH]
        assert config_hash(cfg)==a['config_sha256'] and d['limit_usd']==85
        assert time.time()<a['deadline_unix'] and not d['policy_violation'] and time.time()>=d['cooldown_until']
        assert not any(r['status']!='returned_usage' or r.get('error_type') for r in d['requests'][a['start_request_count']:])
        assert all(digest(Path(p))==h for p,h in cfg['source_hashes'].items())
        for i,arm in cfg['job_order']:
            task=cfg['tasks'][i];out=ROOT/'online'/arm
            ep=out/'predictions'/(cfg['model']+'__'+task['id']+'__dynamic_interrupt')
            if ep.exists():
                assert (ep/'result.json').exists(),'Incomplete episode: audit before any continuation'
                assert read(ep/'result.json')['status'] in ('success','model_call_limit','step_limit','wall_time_limit','environment_failure','interrupt_budget_exhausted','format_error')
                continue
            assert digest(Path(task['gamefile']))==task['game_sha256']
            from skillopt.envs.alfworld.rollout import build_alfworld_env
            from dynamic_skill.micro_loop_protocol import SYSTEM
            if cfg['arms'][arm].get('macro_mode'):
                from dynamic_skill.space_subskills import SYSTEM_SKILL
                SYSTEM=SYSTEM_SKILL
            s=Settings(**cfg['settings']);api=ApiPlanner(s,system_prompt=SYSTEM)
            env=build_alfworld_env(env_num=1,eval_dataset='eval_in_distribution',seed=42,is_train=False,specific_gamefiles=[task['gamefile']])
            before=b.effective_spend(d)
            try:row=run_control_episode(env,task,s,out,BudgetPlanner(api,b,str(ep)),cfg['task_skill'],ControlPolicy(**cfg['policy']),**cfg['common'],**cfg['arms'][arm])
            finally:api.close();env.close()
            row.update(arm=arm,cost_usd=b.effective_spend(b.read())-before)
            save(ep/'result.json',row);summary();return
        summary()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--authorize',action='store_true');p.add_argument('--summary',action='store_true');a=p.parse_args()
    if a.prepare:prepare()
    if a.authorize:authorize()
    if a.summary:summary()
