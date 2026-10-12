"""免费环境回放核对，不作为新增模型成绩。"""
from experiments.control_parameter_trial import ROOT,read,base
from experiments.puzzle_pilot import save
from dynamic_skill.control_only_evolution import ControlPolicy,ControlExecutor,fingerprint
from dynamic_skill.micro_loop_protocol import SYSTEM,make_prompt,parse_response

def run():
 from skillopt.envs.alfworld.rollout import build_alfworld_env
 cfg=read(ROOT/'config.json');ledger=read(base.CAMPAIGN/'budget.json')
 episodes=steps=prompts=calls_total=0;details=[]
 for p in ROOT.glob('*/*/predictions/*/result.json'):
  row=read(p);out=p.parent;initial=read(out/'initial.json')
  calls=read(out/'model_calls.json');trace=read(out/'conversation.json')
  policy=ControlPolicy(**cfg['policies'][row['policy_id']])
  assert row['task_skill_sha256']==fingerprint(cfg['task_skill'])
  paid=[r for r in ledger['requests'] if r['label']==str(out)]
  assert len(calls)==len(paid)==row['model_calls']
  for c in calls:
   assert c['system_prompt']==SYSTEM and not c.get('reasoning_content')
   assert not ((c.get('usage') or {}).get('completion_tokens_details') or {}).get('reasoning_tokens')
   if c['status']=='ok':
    actions,thought=parse_response(c['response'],c['effective_action_limit'])
    assert actions==c['actions'] and thought==c['reasoning']
  env=build_alfworld_env(env_num=1,eval_dataset='eval_in_distribution',seed=42,is_train=False,specific_gamefiles=[initial['gamefile']])
  single_recovery=delayed_invalid=0
  try:
   obs,_=env.reset({});assert obs['anchor'][0]==initial['observation']
   ctl=ControlExecutor(policy);ctl.start(initial['observation'],task=env.tasks[0])
   for i,step in enumerate(trace):
    if step['model_called']:
     c=calls[step['model_call_id']-1];limit=min(ctl.planning_limit,cfg['settings']['max_steps']-i)
     assert c['effective_action_limit']==limit
     single_recovery+=bool(ctl.recovery_remaining)
     prompt=make_prompt(cfg['task_skill'],limit,'Replan from the current observation.')
     assert c['prompt']==prompt('',task=env.tasks[0],initial=initial['observation'],observation=obs['anchor'][0],admissible=list(env.envs.get_admissible_commands[0]),trace=trace[:i],anomaly=ctl.last_replan_reason)
     ctl.load_chunk(c['actions']);prompts+=1
    action=ctl.next_action();assert action==step['action']
    obs,reward,done,info=env.step(['<think></think><action>'+action+'</action>'])
    assert obs['anchor'][0]==step['env_feedback']
    event=ctl.observe(action,obs['anchor'][0],reward=float(reward[0]),done=bool(done[0]),info=info[0])
    assert event.decision.kind.value==step['controller_decision'] and list(event.discarded_actions)==step['discarded_actions']
    delayed_invalid+=bool(ctl.checker.invalid_streak==1 and policy.invalid_patience==2 and not event.check.should_interrupt)
    steps+=1
   assert len(trace)==row['environment_steps']
   assert row['success']==bool(trace and trace[-1]['done'] and (trace[-1]['won'] or trace[-1]['reward']>0))
  finally:env.close()
  episodes+=1;calls_total+=len(calls)
  details.append({'阶段':row['stage'],'参数':row['policy_id'],'单步恢复规划次数':single_recovery,'首次非法反馈延迟中断次数':delayed_invalid})
 result={'通过':True,'回合':episodes,'真实模型调用':calls_total,'免费回放动作':steps,'核对提示':prompts,'机制触发':details,'说明':'逐动作回放与原轨迹一致；回放不是新模型成绩。无已报告原生思考token，不推断服务端隐藏行为。'}
 save(ROOT/'audit.json',result);print(result)

if __name__=='__main__':run()
