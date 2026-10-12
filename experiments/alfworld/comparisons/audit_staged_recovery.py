"""离线重放实际环境轨迹，检查提示与队列，不发模型请求。"""
from experiments import alfworld_staged_recovery as exp
from experiments import alfworld_campaign as base
from experiments.puzzle_pilot import save
from dynamic_skill.skill_executor import ChunkExecutor
from dynamic_skill.alfworld_recovery import recovery_prompt

def run():
    from skillopt.envs.alfworld.rollout import build_alfworld_env
    cfg=exp.read(exp.ROOT/'config.json')
    ledger=exp.read(base.CAMPAIGN/'budget.json');steps=0;prompts=0;episodes=0
    for split,tasks in cfg['splits'].items():
        for task in tasks:
            paths=[p for p in (exp.ROOT/split).glob('*/*/predictions/*/result.json') if exp.read(p)['task_id']==task['id']]
            if not paths:continue
            env=build_alfworld_env(env_num=1,eval_dataset='eval_in_distribution',seed=42,is_train=False,specific_gamefiles=[task['gamefile']])
            try:
                for p in paths:
                    row=exp.read(p);out=p.parent
                    calls=exp.read(out/'model_calls.json') if (out/'model_calls.json').exists() else []
                    trace=exp.read(out/'conversation.json') if (out/'conversation.json').exists() else []
                    paid=[r for r in ledger['requests'] if r['label']==str(out)]
                    assert len(calls)==len(paid)==row['model_calls']
                    assert all(r['status']!='pending' for r in paid)
                    obs,info=env.reset({});initial=obs['anchor'][0]
                    assert initial==exp.read(out/'initial.json')['observation']
                    k=1 if row['mode']=='react' else cfg['settings']['chunk_size']
                    ctl=ChunkExecutor(chunk_size=k,dynamic=row['mode']=='dynamic_interrupt');ctl.start(initial,task=env.tasks[0])
                    for i,step in enumerate(trace):
                        if step['model_called']:
                            call=calls[step['model_call_id']-1]
                            admissible=list(env.envs.get_admissible_commands[0])
                            prompt=base.prompt_for(env.tasks[0],obs['anchor'][0],admissible,trace[:i],cfg['skill'],min(k,50-i),ctl.last_replan_reason)
                            if row['policy']=='recovery':
                                prompt=recovery_prompt(prompt,task=env.tasks[0],initial=initial,observation=obs['anchor'][0],admissible=admissible,trace=trace[:i],anomaly=ctl.last_replan_reason)
                            assert prompt==call['prompt'];ctl.load_chunk(call['actions']);prompts+=1
                        action=ctl.next_action();assert action==step['action']
                        assert list(env.envs.get_admissible_commands[0])==step['admissible_before']
                        obs,reward,done,info=env.step(['<think></think><action>'+action+'</action>'])
                        assert obs['anchor'][0]==step['env_feedback'] and float(reward[0])==step['reward']
                        event=ctl.observe(action,obs['anchor'][0],reward=float(reward[0]),done=bool(done[0]),info=info[0])
                        assert event.decision.kind.value==step['controller_decision'] and list(event.discarded_actions)==step['discarded_actions']
                        steps+=1
                    assert len(trace)==row['environment_steps']
                    assert row['success']==bool(trace and trace[-1]['done'] and (trace[-1]['won'] or trace[-1]['reward']>0))
                    episodes+=1
            finally:env.close()
    report={'通过':True,'真实回合数':episodes,'动作回放数':steps,'已执行调用提示核对数':prompts,
            '说明':'离线回放不是额外模型成绩；付费调用计数与共享账本匹配。'}
    save(exp.ROOT/'audit.json',report);print(report)
