"""不发模型请求；核对四组协议、动作队列、技能参数、提示和账本。"""
from experiments import micro_evolution_loop as e
from dynamic_skill.micro_loop_protocol import SYSTEM,parse_controller,INITIAL_CONTROL,make_prompt,parse_response,SkillChecker
from dynamic_skill.skill_executor import ChunkExecutor

def run():
    from skillopt.envs.alfworld.rollout import build_alfworld_env
    cfg=e.read(e.ROOT/'config.json');ledger=e.read(e.base.CAMPAIGN/'budget.json')
    chosen=e.read(e.ROOT/'selection.json') if (e.ROOT/'selection.json').exists() else {}
    episodes=steps=prompts=missing=0
    for split,tasks in cfg['splits'].items():
        for task in tasks:
            paths=[p for p in (e.ROOT/split).glob('*/*/predictions/*/result.json') if e.read(p)['task_id']==task['id']]
            if not paths:continue
            env=build_alfworld_env(env_num=1,eval_dataset='eval_in_distribution',seed=42,is_train=False,specific_gamefiles=[task['gamefile']])
            try:
                for p in paths:
                    row=e.read(p);arm=row['arm'];out=p.parent
                    if arm=='react':skill=cfg['initial_skill']
                    elif split=='dev' or arm=='ours_initial_dynamic':skill=e.ours_initial(cfg['initial_skill'])
                    elif split=='validation':skill=e.read(e.ROOT/'evolution'/('skillopt' if arm=='skillopt' else 'ours')/'candidate.json')['skill']
                    else:skill=chosen['skillopt' if arm=='skillopt' else 'ours']['skill']
                    assert e.hashlib.sha256(skill.encode()).hexdigest()==row['skill_sha']
                    calls=e.read(out/'model_calls.json') if (out/'model_calls.json').exists() else []
                    trace=e.read(out/'conversation.json') if (out/'conversation.json').exists() else []
                    paid=[r for r in ledger['requests'] if r['label']==str(out)]
                    assert len(calls)==len(paid)==row['model_calls']
                    for call in calls:
                        assert call['system_prompt']==SYSTEM and not call.get('reasoning_content')
                        details=(call.get('usage') or {}).get('completion_tokens_details') or {}
                        assert not details.get('reasoning_tokens');missing+=details.get('reasoning_tokens') is None
                        if call['status']=='ok':
                            actions,thought=parse_response(call['response'],call['action_limit'])
                            assert actions==call['actions'] and thought==call['reasoning']
                    obs,info=env.reset({});initial=obs['anchor'][0]
                    control=parse_controller(skill) if arm not in ['react','skillopt'] else INITIAL_CONTROL
                    k=1 if row['mode']=='react' else control['max_actions']
                    ctl=ChunkExecutor(chunk_size=k,dynamic=row['mode']=='dynamic_interrupt',checker=SkillChecker(control['trigger']))
                    ctl.start(initial,task=env.tasks[0]);prompt=make_prompt(skill,k,control['recovery_note'])
                    for i,step in enumerate(trace):
                        if step['model_called']:
                            call=calls[step['model_call_id']-1]
                            expected=prompt('',task=env.tasks[0],initial=initial,observation=obs['anchor'][0],admissible=list(env.envs.get_admissible_commands[0]),trace=trace[:i],anomaly=ctl.last_replan_reason)
                            assert expected==call['prompt'];ctl.load_chunk(call['actions']);prompts+=1
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
    result={'通过':True,'回合数':episodes,'动作回放数':steps,'已执行提示核对数':prompts,
            '思考token字段缺失或空值的调用数':missing,'说明':'无原生思考内容报告；缺失字段不能证明服务端没有隐藏思考。回放不计新模型成绩。'}
    e.save(e.ROOT/'audit.json',result);print(result)
