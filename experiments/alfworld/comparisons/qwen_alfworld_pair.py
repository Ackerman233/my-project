"""两个 Qwen 模型的有界 ALFWorld 对照；复用官方环境，不修改上游源码。"""
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from urllib.parse import urlparse
sys.path.insert(0,str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))
from experiments import alfworld_campaign as base
from experiments.puzzle_pilot import Settings,ApiPlanner,save
from dynamic_skill.campaign_budget import CampaignBudget,BudgetPlanner

OUT=Path('results/qwen_alfworld_pair_20260929')
AUTH='qwen_alfworld_pair_20260929_authorization'
MODELS=['qwen3.7-plus','qwen3.6-35b-a3b']
def digest(c):return hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()

def prepare():
    from skillopt.envs.alfworld.rollout import build_alfworld_env
    with (base.CAMPAIGN/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        b=CampaignBudget(base.CAMPAIGN/'budget.json',deadline=0);d=b.read()
        assert not OUT.exists() and AUTH not in d
        assert not any(r['status']=='pending' for r in d['requests'])
        assert not d['policy_violation'] and time.time()>=d['cooldown_until']
        tasks=base.tasks(); smoke=[]
        for t in tasks:
            env=build_alfworld_env(env_num=1,eval_dataset='eval_in_distribution',seed=42,is_train=False,specific_gamefiles=[t['gamefile']])
            try:
                obs,infos=env.reset({})
                smoke.append({'task_id':t['id'],'observation':obs['anchor'][0],'admissible_count':len(env.envs.get_admissible_commands[0])})
            finally:env.close()
        cfg={'protocol':'alfworld-actiononly-v3-qwen-pair','tasks':tasks,'skill':base.skill_text(),
             'system_prompt':base.SYSTEM,'models':MODELS,'modes':list(base.MODES),
             'settings':asdict(Settings(max_steps=50,max_calls=20,max_tokens=256,temperature=0,
                  request_timeout=45,episode_seconds=210,action_only=True)),
             'run_order':[[t['id'],mode,model] for mode in base.MODES for t in tasks for model in MODELS],
             'deadline_unix':time.time()+3600,'max_new_requests':240,'enable_thinking':False,'seed':42,
             '说明':'沿用已有 ALFWorld action-only 协议；本轮上限20调用，与原50调用实验不能直接混算。无训练、无调参、无自动重试。'}
        cfg['hash']=digest(cfg)
        unknown=sum(r['status']=='unknown_usage' for r in d['requests'])
        auth={'at_unix':time.time(),'initial_unknown':unknown,'config_sha':cfg['hash'],
              'initial_requests':len(d['requests']),'initial_effective_usd':b.effective_spend(d),
              'old_deadline':d['deadline_unix'],'old_max_requests':d['max_requests'],
              'old_max_unknown':d['max_unknown_requests'],'old_limit':d['limit_usd'],
              'reason':'用户要求两个Qwen模型测试ALFWorld；累计预算仍不超过30美元。'}
        save(OUT/'config.json',cfg);save(OUT/'smoke.json',smoke);save(OUT/'authorization.json',auth)
        d[AUTH]=auth;d.update(limit_usd=30.0,deadline_unix=cfg['deadline_unix'],
          max_requests=len(d['requests'])+240,max_unknown_requests=unknown+1)
        b.write(d)
        print(json.dumps({'prepared':True,'episodes':12,'effective_usd':b.effective_spend(d)}))

def authorize_remaining(second=False):
    with (base.CAMPAIGN/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        b=CampaignBudget(base.CAMPAIGN/'budget.json',deadline=0);d=b.read()
        key=AUTH+('_resume2' if second else '_resume')
        assert key not in d
        assert not d['policy_violation'] and time.time()>=d['cooldown_until']
        assert not any(r['status']=='pending' for r in d['requests'])
        cfg=json.loads((OUT/'config.json').read_text())
        assert digest({k:v for k,v in cfg.items() if k!='hash'})==cfg['hash']
        skipped=[];remaining=[]
        for tid,mode,model in cfg['run_order']:
            p=OUT/'predictions'/(model+'__'+tid+'__'+mode)
            if p.exists():
                assert (p/'result.json').exists()
                skipped.append(p.name)
            else:remaining.append(p.name)
        assert len(remaining)==(5 if second else 7)
        auth={'config_sha':cfg['hash'],'initial_unknown':sum(r['status']=='unknown_usage' for r in d['requests']),
              'deadline_unix':time.time()+3600,'initial_requests':len(d['requests']),
              'initial_effective_usd':b.effective_spend(d),'skip_existing':skipped,'remaining':remaining,
              'old_deadline':d['deadline_unix'],'old_max_requests':d['max_requests'],
              'old_max_unknown':d['max_unknown_requests'],
              'reason':'用户明确要求续跑剩余组合；不重试原超时组、不重置历史费用。'}
        d[key]=auth
        d.update(deadline_unix=auth['deadline_unix'],max_requests=len(d['requests'])+len(remaining)*20,
                 max_unknown_requests=auth['initial_unknown']+1)
        b.write(d);save(OUT/('resume2_authorization.json' if second else 'resume_authorization.json'),auth)
        print(json.dumps({'remaining':len(remaining),'effective_usd':b.effective_spend(d)}))

def advance():
    from skillopt.envs.alfworld.rollout import build_alfworld_env
    proxy=urlparse(os.environ.get('HTTPS_PROXY',''));assert proxy.hostname=='127.0.0.1' and proxy.port==17890
    with (base.CAMPAIGN/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        b=CampaignBudget(base.CAMPAIGN/'budget.json',deadline=0);d=b.read()
        cfg=json.loads((OUT/'config.json').read_text());assert digest({k:v for k,v in cfg.items() if k!='hash'})==cfg['hash']
        auth=d.get(AUTH+'_resume2',d.get(AUTH+'_resume',d[AUTH]));assert auth['config_sha']==cfg['hash'] and d['limit_usd']==30
        assert not d['policy_violation'] and time.time()<auth.get('deadline_unix',cfg['deadline_unix'])
        assert not any(r['status']=='pending' for r in d['requests'])
        assert sum(r['status']=='unknown_usage' for r in d['requests'])==auth['initial_unknown']
        okay={'success','format_error','step_limit','model_call_limit','wall_time_limit','environment_failure','interrupt_budget_exhausted'}
        for tid,mode,model in cfg['run_order']:
            directory=OUT/'predictions'/(model+'__'+tid+'__'+mode)
            if directory.exists():
                assert (directory/'result.json').exists(),'未完成的轨迹需要人工核对'
                assert directory.name in auth.get('skip_existing',[]) or json.loads((directory/'result.json').read_text())['status'] in okay
                continue
            task=next(t for t in cfg['tasks'] if t['id']==tid)
            assert hashlib.sha256(Path(task['gamefile']).read_bytes()).hexdigest()==task['game_sha256']
            s=Settings(**cfg['settings']);s.model=model
            env=build_alfworld_env(env_num=1,eval_dataset='eval_in_distribution',seed=42,is_train=False,specific_gamefiles=[task['gamefile']])
            api=ApiPlanner(s,system_prompt=cfg['system_prompt']);before=b.effective_spend(d)
            try:
                row=base.run_episode(env,task,mode,s,OUT,BudgetPlanner(api,b,str(directory)),cfg['skill'])
            finally:api.close();env.close()
            d=b.read();row.update(cost_usd=b.effective_spend(d)-before,config_sha=cfg['hash'])
            save(directory/'result.json',row)
            save(OUT/'budget_snapshot.json',{'effective_usd':b.effective_spend(d),'limit_usd':d['limit_usd'],
                 'requests':len(d['requests']),'pending':sum(r['status']=='pending' for r in d['requests'])})
            if row['status'] not in okay or sum(r['status']=='unknown_usage' for r in d['requests'])>auth['initial_unknown']:
                save(OUT/'run_status.json',{'status':'已停止','原因':row['status'],'task':directory.name})
                raise RuntimeError('停止新增付费请求：'+row['status'])
            return
        save(OUT/'run_status.json',{'status':'已完成','episodes':12})
        print(json.dumps({'status':'complete','episodes':12}))

if __name__=='__main__':
    {'prepare':prepare,'authorize_remaining':authorize_remaining,'authorize_remaining2':lambda:authorize_remaining(True)}[sys.argv[1]]()
