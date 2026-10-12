"""一轮真实 SkillOpt 核心更新与有界动作技能更新，共享轨迹及独立选择集。"""
import copy,fcntl,hashlib,json,os,subprocess,time
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlparse
from experiments import alfworld_campaign as base
from experiments.puzzle_pilot import ApiPlanner,Settings,save
from dynamic_skill.campaign_budget import CampaignBudget,BudgetPlanner,CampaignHalt
from dynamic_skill.skill_executor import ChunkExecutor
from dynamic_skill.micro_loop_protocol import SYSTEM,INITIAL_CONTROL,controller_block,parse_controller,parse_response,SkillChecker,make_prompt

ROOT=Path('results/micro_evolution_qwen36_20260929')
AUTH='micro_evolution_qwen36_45'
MODEL='qwen3.6-35b-a3b'
SAFE={'success','step_limit','model_call_limit','format_error','interrupt_budget_exhausted','environment_failure'}
class Pause(BaseException):pass
def read(p):return json.loads(p.read_text())
def sha(v):return hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()
def initial_skill():
    s=base.skill_text()
    return s+'\n\n## Controlled output protocol\nUse a brief thought followed by the requested concrete action array; the current system output contract overrides any other formatting instruction.\n'
def ours_initial(s):
    return s+'\n## Executable chunk and interruption policy\n'+controller_block(INITIAL_CONTROL)+'\n'

def prepare():
    with (base.CAMPAIGN/'run.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        b=CampaignBudget(base.CAMPAIGN/'budget.json',deadline=0);d=b.read()
        assert not ROOT.exists() and AUTH not in d
        assert not d['policy_violation'] and not any(r['status']=='pending' for r in d['requests'])
        assert time.time()>=d['cooldown_until']
        previous=read(Path('results/alfworld_staged_recovery_20260929/config.json'))
        kinds={'look_at_obj_in_light','pick_and_place_simple'}
        splits={k:[t for t in tasks if t['task_type'] in kinds] for k,tasks in previous['splits'].items()}
        assert all(len(v)==2 for v in splits.values())
        cfg={'model':MODEL,'splits':splits,'system':SYSTEM,'initial_skill':initial_skill(),
             'settings':asdict(Settings(model=MODEL,max_steps=50,max_calls=20,max_tokens=384,
                request_timeout=45,episode_seconds=210,temperature=0,action_only=False)),
             'native_thinking':False,'optimizer_tokens':2048,'optimizer_max_calls_per_arm':4,
             'upstream_commit':subprocess.check_output(['git','-C','skillopt','rev-parse','HEAD'],text=True).strip(),
             'selection':'分别与自身初始策略配对；成功数不得降低，成功数相同则共同成功题调用必须严格降低；无共同成功题不接受。选择后才打开测试集。',
             'baseline_label':'SkillOpt官方核心流程受控适配版：官方ALFWorld分析提示+Reflect/Aggregate/Rank/Apply，一轮更新，关闭meta/slow-update；非默认完整训练复现。',
             'comparison':'初始ReAct；所选SkillOpt单步；所选动作技能固定chunk；同一所选动作技能动态中断。',
             'fairness':'两条进化分支共享4条开发轨迹及相同最多4次优化调用上限；单步/分块机制及允许学习的参数不同；记录实际优化成本。',
             'source_hashes':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['dynamic_skill/micro_loop_protocol.py','experiments/alfworld_campaign.py']}}
        cfg['hash']=sha(cfg)
        a={'config_sha':cfg['hash'],'initial_requests':len(d['requests']),'initial_effective_usd':b.effective_spend(d),
           'initial_unknown':sum(r['status']=='unknown_usage' for r in d['requests']),
           'old_limit':d['limit_usd'],'old_deadline':d['deadline_unix'],'old_max_requests':d['max_requests'],
           'old_max_unknown':d['max_unknown_requests'],'deadline':time.time()+7200,
           'reason':'用户授权qwen3.6模型、显式简短思考、一个四组进化闭环；总额45美元，超时等待有界重试。'}
        save(ROOT/'config.json',cfg);save(ROOT/'authorization.json',a)
        save(ROOT/'initial_skills.json',{'skillopt':cfg['initial_skill'],'ours':ours_initial(cfg['initial_skill'])})
        d[AUTH]=a;d.update(limit_usd=45.0,deadline_unix=a['deadline'],max_requests=len(d['requests'])+500,
                         max_unknown_requests=a['initial_unknown']+5)
        b.write(d);print(json.dumps({'status':'prepared','effective_usd':b.effective_spend(d),'tasks':6}))

def directory(split,arm,task,attempt=0):
    return ROOT/split/arm/('first' if attempt==0 else 'retry_1')/'predictions'/(MODEL+'__'+task['id']+'__'+('react' if arm in ['react','skillopt'] else 'dynamic_interrupt' if 'dynamic' in arm else 'fixed_chunk'))
def get_result(split,arm,task):
    p=directory(split,arm,task)/'result.json'
    if not p.exists():return None
    row=read(p);retry=directory(split,arm,task,1)/'result.json'
    if row['status']=='api_error' and retry.exists():row=read(retry)
    return row

def episode(cfg,b,split,arm,task,skill):
    from skillopt.envs.alfworld.rollout import build_alfworld_env
    attempt=0;p=directory(split,arm,task)
    if p.exists():
        assert (p/'result.json').exists(),'incomplete episode needs audit'
        row=read(p/'result.json')
        if row['status'] in SAFE:return False
        if row['status']!='api_error':raise Pause('回合保护停止：'+row['status'])
        calls=read(p/'model_calls.json')
        if calls[-1].get('error_type')!='APITimeoutError':raise Pause('非超时接口异常')
        attempt=1;p=directory(split,arm,task,1)
        if p.exists():
            assert (p/'result.json').exists()
            if read(p/'result.json')['status'] not in SAFE:raise Pause('单题超时重试上限')
            return False
    control=parse_controller(skill) if arm not in ['react','skillopt'] else INITIAL_CONTROL
    s=Settings(**cfg['settings']);s.chunk_size=control['max_actions']
    mode='react' if arm in ['react','skillopt'] else 'dynamic_interrupt' if 'dynamic' in arm else 'fixed_chunk'
    k=1 if mode=='react' else s.chunk_size
    executor=ChunkExecutor(chunk_size=k,dynamic=mode=='dynamic_interrupt',checker=SkillChecker(control['trigger']))
    assert hashlib.sha256(Path(task['gamefile']).read_bytes()).hexdigest()==task['game_sha256']
    env=build_alfworld_env(env_num=1,eval_dataset='eval_in_distribution',seed=42,is_train=False,specific_gamefiles=[task['gamefile']])
    api=ApiPlanner(s,system_prompt=SYSTEM);before=b.effective_spend(b.read())
    try:
        row=base.run_episode(env,task,mode,s,p.parent.parent,BudgetPlanner(api,b,str(p)),skill,
            prompt_transform=make_prompt(skill,k,control['recovery_note']),response_parser=parse_response,
            system_prompt=SYSTEM,executor=executor)
    finally:api.close();env.close()
    d=b.read();row.update(split=split,arm=arm,attempt=attempt,config_sha=cfg['hash'],
        skill_sha=hashlib.sha256(skill.encode()).hexdigest(),control=control,cost_usd=b.effective_spend(d)-before)
    save(p/'result.json',row)
    if row['status']=='api_error':d[AUTH]['retry_not_before']=time.time()+60;b.write(d)
    snapshot(b)
    print(json.dumps({'status':'episode','split':split,'arm':arm,'task':task['id'],'outcome':row['status'],'calls':row['model_calls']}))
    return True

def snapshot(b):
    d=b.read();save(ROOT/'budget_snapshot.json',{'effective_usd':b.effective_spend(d),'limit_usd':d['limit_usd'],
       'requests':len(d['requests']),'pending':sum(r['status']=='pending' for r in d['requests'])})

def evolve(cfg,b,arm):
    # 官方函数原样调用，只替换其模型传输层以接入预算、关闭原生思考和调用缓存。
    import skillopt.gradient.reflect as reflect
    import skillopt.gradient.aggregate as aggregate
    import skillopt.optimizer.clip as clip
    from skillopt.optimizer.skill import apply_patch_with_report
    from skillopt.prompts import load_prompt
    out=ROOT/'evolution'/arm
    if (out/'candidate.json').exists():return False
    items=[]
    for policy in ['react','ours_fixed']:
        for task in cfg['splits']['dev']:
            row=get_result('dev',policy,task);assert row and row['status'] in SAFE
            src=directory('dev',policy,task,row['attempt'])
            ident=policy+'__'+task['id'];dest=ROOT/'shared_development'/'predictions'/ident
            trace=read(src/'conversation.json') if (src/'conversation.json').exists() else []
            if not trace:continue
            save(dest/'conversation.json',trace)
            initial=read(src/'initial.json')
            items.append({'id':ident,'task_description':initial['task'],'task_type':task['task_type'],
                 'hard':int(row['success']),'fail_reason':'' if row['success'] else row['status'],
                 'n_turns':row['environment_steps'],'target_system_prompt':SYSTEM,
                 'target_user_prompt':read(src/'model_calls.json')[0]['prompt']})
    assert items,'no executed development data'
    initial=cfg['initial_skill'] if arm=='skillopt' else ours_initial(cfg['initial_skill'])
    count=0
    def transport(*,system,user,**unused):
        nonlocal count
        count+=1
        if count>4:raise Pause('优化调用上限')
        signature=sha({'system':system,'user':user})
        for attempt in range(2):
            p=out/f'call_{count:02d}_attempt_{attempt}.json'
            if p.exists():
                rec=read(p);assert rec['prompt_sha']==signature
                if rec['status']=='ok':return rec['response'],rec.get('usage')
                if rec.get('error_type')=='APITimeoutError' and attempt==0:continue
                raise Pause('优化器已有失败记录，不能隐式覆盖')
            s=Settings(**cfg['settings']);s.max_tokens=cfg['optimizer_tokens']
            api=ApiPlanner(s,system_prompt=system);planner=BudgetPlanner(api,b,str(p))
            d=b.read()
            reserve=max(.5,((len(system.encode())+len(user.encode())+2048)*d['input_usd_per_million']+s.max_tokens*d['output_usd_per_million'])/1e6)
            try:
                planner.before_call(reserve)
                rec={'status':'pending','prompt_sha':signature,'system':system,'prompt':user,'enable_thinking':False}
                save(p,rec);start=time.monotonic()
                try:rec.update(planner(user));rec['status']='ok'
                except Exception as e:rec.update(status='api_error',error_type=type(e).__name__)
                rec['latency_seconds']=time.monotonic()-start;save(p,rec)
                d=b.read();snapshot(b)
                if rec['status']!='ok':
                    d[AUTH]['retry_not_before']=time.time()+60;b.write(d)
                    raise Pause('wait_optimizer_timeout' if rec['error_type']=='APITimeoutError' else '优化接口错误')
                if d['policy_violation'] or d['requests'][-1]['status']!='returned_usage':raise Pause('优化用量或原生思考保护')
                if rec.get('finish_reason')=='length':raise Pause('优化输出截断，不应用不完整修改')
                return rec['response'],rec.get('usage')
            finally:api.close()
    originals=[m.chat_optimizer for m in [reflect,aggregate,clip]]
    for m in [reflect,aggregate,clip]:m.chat_optimizer=transport
    context=('Learn generalizable chunk boundaries from successes and interrupt/recovery rules from failures. '
             'You may edit task guidance and the controller_json block. Preserve exactly one controller_json block. '
             'Its only keys: max_actions integer 1..4; trigger "invalid" or "anomaly"; recovery_note string <=1200 characters. '
             'Do not add task IDs, object instance IDs or memorized solutions. Keep the output thought/actions protocol unchanged.') if arm=='ours' else ''
    try:
        patches={}
        for success in [False,True]:
            group=[r for r in items if bool(r['hard'])==success]
            if not group:continue
            fun=reflect.run_success_analyst_minibatch if success else reflect.run_error_analyst_minibatch
            system=load_prompt('analyst_success' if success else 'analyst_error',env='alfworld')
            patch=fun(initial,group,str(ROOT/'shared_development'/'predictions'),edit_budget=4,
                      system_prompt=system,step_buffer_context=context)
            if not patch:raise Pause('官方分析未生成可用修改')
            patches['success' if success else 'failure']=patch
        merged=aggregate.merge_patches(initial,[patches['failure']['patch']] if 'failure' in patches else [],
                 [patches['success']['patch']] if 'success' in patches else [],workers=1,verbose=False)
        ranked=clip.rank_and_select(initial,merged,max_edits=4)
        candidate,application=apply_patch_with_report(initial,ranked)
        if arm=='ours':parse_controller(candidate)
        save(out/'candidate.json',{'skill':candidate,'initial':initial,'patches':patches,'merged':merged,
            'ranked':ranked,'application':application,'changed':candidate!=initial,'optimizer_calls':count,
            'source_items':items,'config_sha':cfg['hash']})
    finally:
        for m,original in zip([reflect,aggregate,clip],originals):m.chat_optimizer=original
    print(json.dumps({'status':'evolved','arm':arm,'calls':count,'changed':candidate!=initial}));return True

def select(cfg):
    decisions={}
    for arm,basearm,candarm in [('skillopt','react','skillopt'),('ours','ours_initial_dynamic','ours_dynamic')]:
        initial=cfg['initial_skill'] if arm=='skillopt' else ours_initial(cfg['initial_skill'])
        candidate=read(ROOT/'evolution'/arm/'candidate.json')['skill']
        pairs=[(get_result('validation',basearm,t),get_result('validation',candarm,t)) for t in cfg['splits']['validation']]
        assert all(a and b and a['status'] in SAFE and b['status'] in SAFE for a,b in pairs)
        before=sum(a['success'] for a,b in pairs);after=sum(b['success'] for a,b in pairs)
        common=[(a,b) for a,b in pairs if a['success'] and b['success']]
        adopted=after>before or (after==before and bool(common) and sum(b['model_calls'] for a,b in common)<sum(a['model_calls'] for a,b in common))
        decisions[arm]={'adopted':adopted,'skill':candidate if adopted else initial,
            'baseline_successes':before,'candidate_successes':after,'config_sha':cfg['hash']}
    save(ROOT/'selection.json',decisions);return decisions

def advance():
    proxy=urlparse(os.environ.get('HTTPS_PROXY',''));assert proxy.hostname=='127.0.0.1' and proxy.port==17890
    with (base.CAMPAIGN/'run.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        b=CampaignBudget(base.CAMPAIGN/'budget.json',deadline=0);d=b.read();a=d[AUTH];cfg=read(ROOT/'config.json')
        assert sha({k:v for k,v in cfg.items() if k!='hash'})==a['config_sha']==cfg['hash']
        for p,h in cfg['source_hashes'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
        assert not any(r['status']=='pending' for r in d['requests'])
        if d['policy_violation'] or time.time()>=a['deadline'] or b.effective_spend(d)+.5>45 or sum(r['status']=='unknown_usage' for r in d['requests'])-a['initial_unknown']>=5:
            save(ROOT/'status.json',{'status':'stopped','reason':'预算、时间、思考或未知用量保护'});print(json.dumps({'status':'stopped'}));return
        wait=max(d['cooldown_until'],a.get('retry_not_before',0))-time.time()
        if wait>0:print(json.dumps({'status':'waiting','seconds':int(wait)+1}));return
        try:
            for t in cfg['splits']['dev']:
                for arm in ['react','ours_fixed']:
                    skill=cfg['initial_skill'] if arm=='react' else ours_initial(cfg['initial_skill'])
                    if episode(cfg,b,'dev',arm,t,skill):return
            for arm in ['skillopt','ours']:
                if evolve(cfg,b,arm):return
            for t in cfg['splits']['validation']:
                for arm in ['react','skillopt','ours_initial_dynamic','ours_dynamic']:
                    skill=(cfg['initial_skill'] if arm=='react' else ours_initial(cfg['initial_skill']) if arm=='ours_initial_dynamic'
                           else read(ROOT/'evolution'/('skillopt' if arm=='skillopt' else 'ours')/'candidate.json')['skill'])
                    if episode(cfg,b,'validation',arm,t,skill):return
            chosen=read(ROOT/'selection.json') if (ROOT/'selection.json').exists() else select(cfg)
            for t in cfg['splits']['test']:
                for arm in ['react','skillopt','ours_fixed','ours_dynamic']:
                    skill=cfg['initial_skill'] if arm=='react' else chosen['skillopt' if arm=='skillopt' else 'ours']['skill']
                    if episode(cfg,b,'test',arm,t,skill):return
            snapshot(b);save(ROOT/'status.json',{'status':'complete','selection':{k:v['adopted'] for k,v in chosen.items()}})
            print(json.dumps({'status':'finished'}))
        except (Pause,CampaignHalt) as e:
            snapshot(b)
            if str(e)=='wait_optimizer_timeout':print(json.dumps({'status':'waiting','seconds':60}));return
            save(ROOT/'status.json',{'status':'stopped','reason':str(e)});print(json.dumps({'status':'stopped','reason':str(e)}))
