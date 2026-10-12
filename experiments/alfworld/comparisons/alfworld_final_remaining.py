"""最后一组有限续跑：超时等待，重试单独保存，历史费用不清零。"""
import fcntl,json,time,os
from urllib.parse import urlparse
from experiments import qwen_alfworld_pair as pair
from experiments.puzzle_pilot import Settings,ApiPlanner,save
from dynamic_skill.campaign_budget import CampaignBudget,BudgetPlanner

AUTH='alfworld_final_remaining_wait_retry_authorization'
def run():
    from skillopt.envs.alfworld.rollout import build_alfworld_env
    proxy=urlparse(os.environ.get('HTTPS_PROXY',''));assert proxy.hostname=='127.0.0.1' and proxy.port==17890
    with (pair.base.CAMPAIGN/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        b=CampaignBudget(pair.base.CAMPAIGN/'budget.json',deadline=0);d=b.read()
        cfg=json.loads((pair.OUT/'config.json').read_text())
        assert pair.digest({k:v for k,v in cfg.items() if k!='hash'})==cfg['hash']
        assert not any(r['status']=='pending' for r in d['requests']) and not d['policy_violation']
        tid,mode,model=cfg['run_order'][-1]
        assert (tid,mode,model)==('val_0001','dynamic_interrupt','qwen3.6-35b-a3b')
        name=model+'__'+tid+'__'+mode
        if AUTH not in d:
            assert not (pair.OUT/'predictions'/name).exists()
            a={'deadline':time.time()+3600,'config_sha':cfg['hash'],'initial_requests':len(d['requests']),
               'initial_effective_usd':b.effective_spend(d),'initial_unknown':sum(r['status']=='unknown_usage' for r in d['requests']),
               'old_deadline':d['deadline_unix'],'old_max_requests':d['max_requests'],
               'old_max_unknown':d['max_unknown_requests'],'max_attempts':3,
               '说明':'用户授权超时等待继续；只运行最后一组，最多两次独立重试，总额30美元。'}
            d[AUTH]=a;d.update(deadline_unix=a['deadline'],max_requests=len(d['requests'])+60,
                               max_unknown_requests=a['initial_unknown']+3)
            b.write(d);save(pair.OUT/'final_remaining_authorization.json',a)
        a=d[AUTH];assert a['config_sha']==cfg['hash'] and d['limit_usd']==30
        if time.time()>=a['deadline']:print(json.dumps({'status':'deadline_stop'}));return
        for attempt in range(3):
            output=pair.OUT if attempt==0 else pair.OUT/f'final_remaining_retry_{attempt}'
            path=output/'predictions'/name
            if path.exists():
                assert (path/'result.json').exists(),'未完成请求须审计，不能重复发送'
                row=json.loads((path/'result.json').read_text())
                if row['status']!='api_error':
                    print(json.dumps({'status':'finished','outcome':row['status'],'attempt':attempt}));return
                calls=json.loads((path/'model_calls.json').read_text())
                if calls[-1].get('error_type')!='APITimeoutError':
                    print(json.dumps({'status':'non_timeout_error_stop'}));return
                continue
            previous=pair.OUT if attempt==1 else pair.OUT/f'final_remaining_retry_{attempt-1}'
            wait_until=max(d['cooldown_until'],a.get('retry_not_before',0))
            if time.time()<wait_until:
                print(json.dumps({'status':'waiting','seconds':int(wait_until-time.time())+1}));return
            task=next(t for t in cfg['tasks'] if t['id']==tid)
            if attempt:save(output/'config.json',{**cfg,'retry_of':str(pair.OUT/'predictions'/name),'attempt':attempt})
            s=Settings(**cfg['settings']);s.model=model
            env=build_alfworld_env(env_num=1,eval_dataset='eval_in_distribution',seed=42,is_train=False,specific_gamefiles=[task['gamefile']])
            api=ApiPlanner(s,system_prompt=cfg['system_prompt']);before=b.effective_spend(d)
            try:row=pair.base.run_episode(env,task,mode,s,output,BudgetPlanner(api,b,str(path)),cfg['skill'])
            finally:api.close();env.close()
            d=b.read();row.update(cost_usd=b.effective_spend(d)-before,config_sha=cfg['hash'],attempt=attempt)
            save(path/'result.json',row)
            d[AUTH]['retry_not_before']=time.time()+60;b.write(d)
            print(json.dumps({'status':'attempt_recorded','outcome':row['status'],'attempt':attempt}));return
        print(json.dumps({'status':'retry_limit_stop'}))
