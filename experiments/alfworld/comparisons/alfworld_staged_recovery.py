"""分阶段、冻结任务、独立验证门槛；所有推理经同一持久预算。"""
from dataclasses import asdict
import fcntl,hashlib,json,os,time
from pathlib import Path
from urllib.parse import urlparse
from experiments import alfworld_campaign as base
from experiments.puzzle_pilot import ApiPlanner,Settings,save
from dynamic_skill.campaign_budget import CampaignBudget,BudgetPlanner
from dynamic_skill.alfworld_recovery import recovery_prompt

ROOT=Path('results/alfworld_staged_recovery_20260929')
AUTH='alfworld_staged_recovery_45_authorization'
TYPES=['pick_and_place_simple','pick_two_obj_and_place','look_at_obj_in_light',
       'pick_clean_then_place_in_recep','pick_heat_then_place_in_recep','pick_cool_then_place_in_recep']
SAFE={'success','model_call_limit','step_limit','format_error','wall_time_limit',
      'environment_failure','interrupt_budget_exhausted'}
def sha(v):return hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()
def result_path(split,policy,task,attempt=0):
    output=ROOT/split/policy/('first' if attempt==0 else 'retry_1')
    mode=policy if policy!='recovery' else 'dynamic_interrupt'
    return output/'predictions'/('qwen3.7-plus__'+task['id']+'__'+mode)
def read(p):return json.loads(p.read_text())

def prepare():
    with (base.CAMPAIGN/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        b=CampaignBudget(base.CAMPAIGN/'budget.json',deadline=0);d=b.read()
        assert not ROOT.exists() and AUTH not in d
        assert not d['policy_violation'] and not any(r['status']=='pending' for r in d['requests'])
        assert time.time()>=d['cooldown_until']
        old=read(Path('results/qwen_alfworld_pair_20260929/config.json'))
        excluded={Path(t['gamefile']).parent.parent.name for t in old['tasks']}
        data=Path(os.environ['ALFWORLD_DATA'])/'json_2.1.1/valid_seen'
        splits={k:[] for k in ['dev','validation','test']};families=set()
        for kind in TYPES:
            paths=sorted(data.glob(kind+'-*/trial_*/game.tw-pddl'),key=lambda p:sha(str(p.relative_to(data))))
            chosen=[]
            for p in paths:
                family=p.parent.parent.name
                if family in families or family in excluded:continue
                families.add(family);chosen.append(p)
                if len(chosen)==3:break
            assert len(chosen)==3
            for split,p in zip(splits,chosen):
                splits[split].append({'id':split+'-'+kind,'gamefile':str(p.resolve()),'task_type':kind,
                   'game_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'benchmark':'alfworld'})
        cfg={'protocol':'alfworld-recovery-evidence-v1','splits':splits,
             'settings':asdict(Settings(model='qwen3.7-plus',max_steps=50,max_calls=20,max_tokens=256,
                temperature=0,request_timeout=45,episode_seconds=210,action_only=True)),
             'skill':base.skill_text(),'system_prompt':base.SYSTEM,'enable_thinking':False,
             'recovery_sha':hashlib.sha256(Path('dynamic_skill/alfworld_recovery.py').read_bytes()).hexdigest(),
             'gate':'独立验证6题全部有效；恢复策略至少多成功1题，且不能让原成功题失败；共同成功题总调用不超过原策略110%。至少一次实际启用恢复提示。',
             'development':'开发6题原始中断+恢复各一遍；不据验证集调参。',
             'test':'仅验证通过时执行6题四策略：ReAct、固定chunk、原始中断、恢复中断。',
             'retry':'每组超时后至少等60秒，遵守共享冷却；最多单独重试一次；本轮最多3个未知用量。',
             'seed':42,'task_selection':'确定性哈希排序，每类每split一题；不同任务族；排除旧两题，无答案筛选。'}
        cfg['hash']=sha(cfg)
        a={'config_sha':cfg['hash'],'initial_requests':len(d['requests']),
           'initial_effective_usd':b.effective_spend(d),'initial_unknown':sum(r['status']=='unknown_usage' for r in d['requests']),
           'old_limit':d['limit_usd'],'old_deadline':d['deadline_unix'],'old_max_requests':d['max_requests'],
           'old_max_unknown':d['max_unknown_requests'],'deadline':time.time()+7200,'max_new_requests':1020,
           'reason':'用户授权分阶段代码改动及实验，累计提高到45美元；保留全部历史账本。'}
        save(ROOT/'config.json',cfg);save(ROOT/'authorization.json',a)
        d[AUTH]=a;d.update(limit_usd=45.0,deadline_unix=a['deadline'],max_requests=len(d['requests'])+1020,
                           max_unknown_requests=a['initial_unknown']+3)
        b.write(d);print(json.dumps({'status':'prepared','tasks':18,'effective_usd':b.effective_spend(d),'limit':45}))

def selected(split,policy,task):
    first=result_path(split,policy,task)/'result.json'
    if not first.exists():return None
    row=read(first)
    retry=result_path(split,policy,task,1)/'result.json'
    if row['status']=='api_error' and retry.exists():row=read(retry)
    return row

def summary(cfg):
    report={}
    for split,tasks in cfg['splits'].items():
        report[split]={}
        for policy in ['react','fixed_chunk','dynamic_interrupt','recovery']:
            rows=[r for t in tasks if (r:=selected(split,policy,t)) is not None]
            if rows:
                report[split][policy]={'n':len(rows),'successes':sum(r['success'] for r in rows),
                    'calls':sum(r['model_calls'] for r in rows),'invalid':sum(r['invalid_actions'] for r in rows),
                    'complete':len(rows)==6 and all(r['status'] in SAFE for r in rows),'rows':rows}
    save(ROOT/'evaluation.json',report);return report

def gate(cfg):
    report=summary(cfg)['validation'];x=report.get('dynamic_interrupt',{});y=report.get('recovery',{})
    adopted=False;reason='验证未完成或含基础设施中止'
    if x.get('complete') and y.get('complete'):
        pairs=list(zip(x['rows'],y['rows']))
        no_regression=all(not a['success'] or b['success'] for a,b in pairs)
        common=[(a,b) for a,b in pairs if a['success'] and b['success']]
        call_ok=not common or sum(b['model_calls'] for a,b in common)<=1.1*sum(a['model_calls'] for a,b in common)
        used=sum(r.get('recovery_calls',0) for r in y['rows'])>0
        adopted=y['successes']>=x['successes']+1 and no_regression and call_ok and used
        reason='预注册门槛通过' if adopted else '未通过成功率、逐题无退步、共同成功题成本或恢复启用门槛'
    g={'adopted':adopted,'reason':reason,'config_sha':cfg['hash'],'validation':report}
    save(ROOT/'validation_gate.json',g);return adopted

def next_job(cfg):
    for split in ['dev','validation','test']:
        if split=='test' and not gate(cfg):return None
        policies=['dynamic_interrupt','recovery'] if split!='test' else ['react','fixed_chunk','dynamic_interrupt','recovery']
        jobs=([(t,p) for p in policies for t in cfg['splits'][split]] if split=='dev'
              else [(t,p) for t in cfg['splits'][split] for p in policies])
        for task,policy in jobs:
            for attempt in range(2):
                p=result_path(split,policy,task,attempt)
                if not p.exists():return split,policy,task,attempt
                assert (p/'result.json').exists(),'不完整回合禁止重复调用，需要审计'
                row=read(p/'result.json')
                if row['status']=='api_error':
                    calls=read(p/'model_calls.json')
                    if calls[-1].get('error_type')=='APITimeoutError':
                        if attempt==0:continue
                        break
                    raise RuntimeError('非超时接口异常，停止新增调用')
                if row['status'] not in SAFE:raise RuntimeError('预算、时间或正确性保护停止：'+row['status'])
                break
    return None

def advance():
    from skillopt.envs.alfworld.rollout import build_alfworld_env
    proxy=urlparse(os.environ.get('HTTPS_PROXY',''));assert proxy.hostname=='127.0.0.1' and proxy.port==17890
    with (base.CAMPAIGN/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        b=CampaignBudget(base.CAMPAIGN/'budget.json',deadline=0);d=b.read();a=d[AUTH]
        cfg=read(ROOT/'config.json');assert sha({k:v for k,v in cfg.items() if k!='hash'})==a['config_sha']==cfg['hash']
        assert hashlib.sha256(Path('dynamic_skill/alfworld_recovery.py').read_bytes()).hexdigest()==cfg['recovery_sha']
        assert not any(r['status']=='pending' for r in d['requests'])
        reason=None
        if d['policy_violation']:reason='原生思考保护'
        elif time.time()>=a['deadline']:reason='批次期限'
        elif b.effective_spend(d)+d['reservation_usd']>45:reason='45美元预算保护'
        elif sum(r['status']=='unknown_usage' for r in d['requests'])-a['initial_unknown']>=3:reason='本轮未知用量次数上限'
        if reason:
            save(ROOT/'run_status.json',{'status':'stopped','reason':reason});print(json.dumps({'status':'stopped','reason':reason}));return
        wait_until=max(d['cooldown_until'],a.get('retry_not_before',0))
        if time.time()<wait_until:print(json.dumps({'status':'waiting','seconds':int(wait_until-time.time())+1}));return
        job=next_job(cfg)
        if job is None:
            adopted=gate(cfg);summary(cfg)
            save(ROOT/'run_status.json',{'status':'finished','adopted':adopted,'reason':'验证通过并完成测试' if adopted else '验证未通过，不扩大测试'})
            print(json.dumps({'status':'finished','adopted':adopted}));return
        split,policy,task,attempt=job
        assert hashlib.sha256(Path(task['gamefile']).read_bytes()).hexdigest()==task['game_sha256']
        s=Settings(**cfg['settings']);mode=policy if policy!='recovery' else 'dynamic_interrupt'
        p=result_path(split,policy,task,attempt);output=p.parent.parent
        env=build_alfworld_env(env_num=1,eval_dataset='eval_in_distribution',seed=42,is_train=False,specific_gamefiles=[task['gamefile']])
        api=ApiPlanner(s,system_prompt=cfg['system_prompt']);before=b.effective_spend(d)
        try:
            row=base.run_episode(env,task,mode,s,output,BudgetPlanner(api,b,str(p)),cfg['skill'],
                                 recovery_prompt if policy=='recovery' else None)
        finally:api.close();env.close()
        calls=read(p/'model_calls.json') if (p/'model_calls.json').exists() else []
        d=b.read();row.update(split=split,policy=policy,attempt=attempt,config_sha=cfg['hash'],
               cost_usd=b.effective_spend(d)-before,recovery_calls=sum(c['recovery_prompt_used'] for c in calls))
        save(p/'result.json',row)
        if row['status']=='api_error':d[AUTH]['retry_not_before']=time.time()+60;b.write(d)
        save(ROOT/'budget_snapshot.json',{'effective_usd':b.effective_spend(d),'limit_usd':d['limit_usd'],
            'requests':len(d['requests']),'pending':sum(r['status']=='pending' for r in d['requests'])})
        summary(cfg)
        print(json.dumps({'status':'episode','split':split,'policy':policy,'task':task['id'],'attempt':attempt,
              'outcome':row['status'],'calls':row['model_calls'],'effective_usd':b.effective_spend(d)}))
