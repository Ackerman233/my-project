"""预算内参数试验：开发筛选→新题验证→条件触发跨模型验证。"""
import fcntl,json,time,hashlib,os
from pathlib import Path
from dataclasses import asdict
from urllib.parse import urlparse
from experiments import alfworld_campaign as base
from experiments.puzzle_pilot import Settings,ApiPlanner,save
from dynamic_skill.campaign_budget import CampaignBudget,BudgetPlanner
from dynamic_skill.control_only_evolution import ControlPolicy,run_control_episode
from dynamic_skill.micro_loop_protocol import SYSTEM

ROOT=Path('results/micro_evolution_qwen36_20260929/control_parameter_trial')
AUTH='control_parameter_trial_45'
POLICIES={'baseline':ControlPolicy(),'patience2':ControlPolicy(invalid_patience=2),'recovery2':ControlPolicy(recovery_steps=2)}
SAFE={'success','format_error','model_call_limit','step_limit','interrupt_budget_exhausted','wall_time_limit','environment_failure'}
def read(p):return json.loads(p.read_text())
def sha(v):return hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()
def prepare():
 with (base.CAMPAIGN/'run.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);b=CampaignBudget(base.CAMPAIGN/'budget.json',deadline=0);d=b.read()
  assert not ROOT.exists() and AUTH not in d
  assert not d['policy_violation'] and not any(r['status']=='pending' for r in d['requests'])
  assert time.time()>=d['cooldown_until'] and d['limit_usd']==45
  old=read(Path('results/qwen_alfworld_pair_20260929/config.json'))
  dev=old['tasks'][1].copy();dev['id']='dev-recovery-bowl'
  prior=read(Path('results/alfworld_staged_recovery_20260929/config.json'))
  excluded={Path(t['gamefile']).parent.parent.name for ts in prior['splits'].values() for t in ts}
  excluded.update(Path(t['gamefile']).parent.parent.name for t in old['tasks'])
  validation=[];data=Path(os.environ['ALFWORLD_DATA'])/'json_2.1.1/valid_seen'
  for kind in ['pick_and_place_simple','look_at_obj_in_light']:
   paths=sorted(data.glob(kind+'-*/trial_*/game.tw-pddl'),key=lambda p:sha('control-trial-v1/'+str(p.relative_to(data))))
   p=next(p for p in paths if p.parent.parent.name not in excluded)
   validation.append({'id':'new-'+kind,'gamefile':str(p),'game_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'benchmark':'alfworld'})
  skill=read(Path('results/micro_evolution_qwen36_20260929/control_only_v2/manifest.json'))['task_skill']
  cfg={'task_skill':skill,'dev':[dev],'validation':validation,'policies':{k:asdict(v) for k,v in POLICIES.items()},
       'settings':asdict(Settings(model='qwen3.6-35b-a3b',max_calls=12,max_steps=50,max_tokens=384,request_timeout=45,episode_seconds=210,temperature=0,action_only=False)),
       'gate':'开发需有成功；候选按成功优先、仅成功时调用少优先选择。独立验证不得损失原成功题，成功数提高或共同成功题调用严格降低；参数必须实际触发。',
       'native_enable_thinking':False,'说明':'仅3配置、1开发题、2新验证题；12调用，与上一轮20调用不混算。qwen3.7只在验证门槛通过且预算允许时运行。'}
  cfg['hash']=sha(cfg);a={'config_sha':cfg['hash'],'deadline':time.time()+3600,'initial_requests':len(d['requests']),
    'initial_effective_usd':b.effective_spend(d),'initial_unknown':sum(r['status']=='unknown_usage' for r in d['requests']),
    'old_deadline':d['deadline_unix'],'old_max_requests':d['max_requests'],'old_max_unknown':d['max_unknown_requests'],
    'reason':'用户授权参数比较；有效时自动qwen3.7，同一45美元总预算。'}
  save(ROOT/'config.json',cfg);save(ROOT/'authorization.json',a);d[AUTH]=a
  d.update(deadline_unix=a['deadline'],max_requests=len(d['requests'])+156,max_unknown_requests=a['initial_unknown']+2);b.write(d)
  print(json.dumps({'status':'prepared','remaining_usd':45-b.effective_spend(d)}))
def path(stage,policy,task,model):return ROOT/stage/policy/'predictions'/(model+'__'+task['id']+'__dynamic_interrupt')
def result(stage,policy,task,model):
 p=path(stage,policy,task,model)/'result.json';return read(p) if p.exists() else None
def advance():
 from skillopt.envs.alfworld.rollout import build_alfworld_env
 proxy=urlparse(os.environ.get('HTTPS_PROXY',''));assert proxy.hostname=='127.0.0.1' and proxy.port==17890
 with (base.CAMPAIGN/'run.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);b=CampaignBudget(base.CAMPAIGN/'budget.json',deadline=0);d=b.read();a=d[AUTH]
  cfg=read(ROOT/'config.json');assert sha({k:v for k,v in cfg.items() if k!='hash'})==a['config_sha']==cfg['hash']
  assert not any(r['status']=='pending' for r in d['requests'])
  if d['policy_violation'] or time.time()>=a['deadline'] or b.effective_spend(d)+.5>45 or sum(r['status']=='unknown_usage' for r in d['requests'])-a['initial_unknown']>=2:
   save(ROOT/'status.json',{'status':'stopped','原因':'预算、时间或正确性保护'});print(json.dumps({'status':'stopped'}));return
  wait=max(d['cooldown_until'],a.get('not_before',0))-time.time()
  if wait>0:print(json.dumps({'status':'waiting','seconds':int(wait)+1}));return
  model='qwen3.6-35b-a3b';jobs=[('dev',p,t,model) for t in cfg['dev'] for p in POLICIES]
  if all(result(*j) for j in jobs):
   rows={p:result('dev',p,cfg['dev'][0],model) for p in POLICIES}
   successful=[p for p in ['patience2','recovery2'] if rows[p]['success']]
   selected=min(successful,key=lambda p:(rows[p]['model_calls'],p)) if successful else None
   if not selected:
    save(ROOT/'selection.json',{'selected':None,'原因':'候选开发题均未成功；不使用提前失败的低调用作为收益'})
    save(ROOT/'status.json',{'status':'finished','原因':'开发筛选未通过，不运行qwen3.7'});print(json.dumps({'status':'finished'}));return
   save(ROOT/'selection.json',{'selected':selected})
   jobs += [('validation',p,t,model) for t in cfg['validation'] for p in ['baseline',selected]]
   if all(result(*j) for j in jobs):
    pairs=[(result('validation','baseline',t,model),result('validation',selected,t,model)) for t in cfg['validation']]
    valid=all(x['status'] in SAFE and y['status'] in SAFE for x,y in pairs)
    before=sum(x['success'] for x,y in pairs);after=sum(y['success'] for x,y in pairs)
    common=[(x,y) for x,y in pairs if x['success'] and y['success']]
    triggered=any(y['anomaly_replans']>0 for x,y in pairs) if selected=='recovery2' else any(y['invalid_actions']>0 for x,y in pairs)
    passed=valid and triggered and all(not x['success'] or y['success'] for x,y in pairs) and (after>before or (after==before and common and sum(y['model_calls'] for x,y in common)<sum(x['model_calls'] for x,y in common)))
    save(ROOT/'gate.json',{'passed':bool(passed),'before':before,'after':after,'selected':selected,'triggered':triggered})
    if not passed:
     save(ROOT/'status.json',{'status':'finished','原因':'独立验证未通过，不运行qwen3.7'});print(json.dumps({'status':'finished'}));return
    jobs += [('transfer37',p,t,'qwen3.7-plus') for t in cfg['validation'] for p in ['baseline',selected]]
  for stage,policy,task,model in jobs:
   out=path(stage,policy,task,model)
   if out.exists():
    assert (out/'result.json').exists(),'incomplete result requires audit'
    r=read(out/'result.json')
    if r['status'] not in SAFE and r['status']!='api_error':raise RuntimeError('unsafe prior result')
    continue
   assert hashlib.sha256(Path(task['gamefile']).read_bytes()).hexdigest()==task['game_sha256']
   s=Settings(**cfg['settings']);s.model=model
   env=build_alfworld_env(env_num=1,eval_dataset='eval_in_distribution',seed=42,is_train=False,specific_gamefiles=[task['gamefile']])
   api=ApiPlanner(s,system_prompt=SYSTEM);before=b.effective_spend(d)
   try:r=run_control_episode(env,task,s,out.parent.parent,BudgetPlanner(api,b,str(out)),cfg['task_skill'],ControlPolicy(**cfg['policies'][policy]))
   finally:api.close();env.close()
   d=b.read();r.update(stage=stage,policy_id=policy,cost_usd=b.effective_spend(d)-before);save(out/'result.json',r)
   if r['status']=='api_error':d[AUTH]['not_before']=time.time()+60;b.write(d)
   save(ROOT/'budget_snapshot.json',{'effective_usd':b.effective_spend(d),'requests':len(d['requests']),'limit_usd':45})
   print(json.dumps({'status':'episode','stage':stage,'policy':policy,'outcome':r['status'],'calls':r['model_calls']}));return
  save(ROOT/'status.json',{'status':'finished','原因':'条件跨模型对照完成'});print(json.dumps({'status':'finished'}))
