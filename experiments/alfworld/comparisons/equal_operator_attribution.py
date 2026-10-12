"""Matched operator attribution: original inference and controlled queue ablations."""
import argparse
import re
from experiments.paper_method_trial import *
from dynamic_skill.micro_loop_protocol import SYSTEM
from skillopt.envs.alfworld.rollout import _build_skill_prompt,_extract_action,_extract_think
ROOT=Path('results/equal_operator_attribution_20261007')
ORIGINAL_SYSTEM='You are an expert agent operating in the ALFRED Embodied Environment.'
ARMS={
 'react':{'method':'upstream_no_skill_react','mode':'official','max_actions':1},
 'skillopt':{'method':'upstream_gpt55_checkpoint_inference','mode':'official','max_actions':1},
 'matched_single':{'method':'same_prompt_single','mode':'dynamic','max_actions':1},
 'matched_fixed':{'method':'same_prompt_fixed_queue','mode':'fixed','max_actions':4},
 'matched_dynamic':{'method':'same_prompt_dynamic_queue','mode':'dynamic','max_actions':4}}


def system_for(spec):
    return ORIGINAL_SYSTEM if spec['mode']=='official' else SYSTEM


def build_prompt(env,obs,cfg,spec,limit,context,search):
    if spec['mode']=='official':
        skill=cfg['original_skill'] if spec['method']=='upstream_gpt55_checkpoint_inference' else ''
        prompt=_build_skill_prompt(skill)+'\n'+obs['text'][0]
        prompt+='\nUse a brief decision summary of at most two sentences inside <think>, followed by exactly one primitive <action>. No programs or macros.'
    else:
        prompt=make_prompt(cfg['task_skill'],limit,'Replan from the current observation.')('',**context)
        if context['anomaly']:prompt=recovery_prompt(prompt,**context,output_protocol='thought_actions')
        prompt+='\nOnly concrete primitive commands, no @macros.'
    # All methods get exactly the same operator state information and scheduling rule.
    prompt+='\nShared public operator state: '+json.dumps(search.public_state(),sort_keys=True)
    prompt+='\nThe common public search/pickup operator runs at a planning boundary when it has a supported primitive; each primitive costs one environment action. Final delivery/tool operations remain yours.'
    return prompt


def parse_output(text,spec,limit):
    if spec['mode']=='official':
        action=_extract_action(text)
        if action is None or len(re.findall(r'<action>',text))!=1:raise ValueError('Expected exactly one upstream action tag')
        actions=[action];reason=_extract_think(text) or '';normalized=False
    else:
        text,normalized=normalize_response(text);actions,reason=parse_response(text,limit)
    if any(a.startswith('@') for a in actions):raise ValueError('Macros are unavailable equally in every arm')
    return actions,reason,normalized


def run_episode(env,task,arm,cfg,planner,out):
    out.mkdir(parents=True,exist_ok=False)
    spec=cfg['arms'][arm];workflow=spec['workflow'];settings=Settings(**cfg['settings'])
    policy=ControlPolicy(max_actions=spec['max_actions'])
    ctl=ControlExecutor(policy,dynamic=spec['mode']=='dynamic',state_check=spec['mode']=='dynamic')
    search=ObservableSearch('',acquire=workflow['acquire'],priority=workflow['priority'])
    obs,_=env.reset({});initial=obs['anchor'][0];task_text=env.tasks[0];observation=initial
    ctl.start(initial,task_text);search.task=task_text
    search=ObservableSearch(task_text,acquire=workflow['acquire'],priority=workflow['priority'])
    initial_record={'observation':initial,'task':task_text,'admissible':list(env.envs.get_admissible_commands[0]),
        'gamefile':task['gamefile'],'game_sha256':digest(Path(task['gamefile']))}
    save(out/'initial.json',initial_record);calls=[];trace=[];save(out/'model_calls.json',calls)
    status='step_limit';started=time.perf_counter();origin='model';operator=None
    for step in range(settings.max_steps):
        if time.perf_counter()-started>=settings.episode_seconds:status='wall_time_limit';break
        admissible=list(env.envs.get_admissible_commands[0])
        if ctl.needs_plan:
            operator=search.action(observation,admissible) if workflow['search'] else None
            if operator is not None:
                assert all(a in admissible for a in operator['actions'])
                ctl.load_chunk(operator['actions']);origin='compiled_public_search'
            else:
                if len(calls)>=settings.max_calls:status='model_call_limit';break
                limit=min(policy.max_actions,settings.max_steps-step)
                context={'task':task_text,'initial':initial,'observation':observation,'admissible':admissible,
                    'trace':trace,'anomaly':ctl.last_replan_reason}
                prompt=build_prompt(env,obs,cfg,spec,limit,context,search)
                # Identical remaining-resource description in every arm.
                prompt+=f'\nBudget remaining: {settings.max_calls-len(calls)} model calls and {settings.max_steps-step} environment actions.'
                try:planner.before_call()
                except CampaignHalt as halt:status=halt.status;break
                call={'call_id':len(calls)+1,'step':step,'prompt':prompt,'system_prompt':system_for(spec),
                    'status':'pending','actions':[],'action_limit':limit};calls.append(call);save(out/'model_calls.json',calls)
                try:
                    call.update(planner(prompt))
                    usage=call.get('usage')
                    if not isinstance(usage,dict) or not all(isinstance(usage.get(k),int) and usage[k]>=0 for k in ('prompt_tokens','completion_tokens')):
                        status=call['status']='unknown_cost_stop';break
                    details=usage.get('completion_tokens_details') or {}
                    if call.get('reasoning_content') or details.get('reasoning_tokens',0):status=call['status']='thinking_policy_stop';break
                    primitive,reason,normalized=parse_output(call['response'],spec,limit);actions=primitive;provenance=[]
                    ctl.load_chunk(primitive);call.update(status='ok',actions=primitive,model_native_actions=actions,
                        reasoning=reason,protocol_normalized=normalized,subskill_expansion=provenance)
                    origin='model';operator=None
                except ValueError as exc:status=call['status']='format_error';call['parse_error']=str(exc);break
                except Exception as exc:status=call['status']='api_error';call['error_type']=type(exc).__name__;break
                finally:save(out/'model_calls.json',calls)
                print(task['id'],arm,'call',len(calls),'step',step,call['status'],flush=True)
        action=ctl.next_action();ctl.before_action(action,observation,admissible)
        obs,rewards,dones,infos=env.step(['<think></think><action>'+action+'</action>']);observation=obs['anchor'][0]
        after=list(env.envs.get_admissible_commands[0]);ctl.after_action(after)
        event=ctl.observe(action,observation,reward=float(rewards[0]),done=bool(dones[0]),info=infos[0])
        search.update(action,observation,after,step)
        row={'step':step,'action':action,'admissible_before':admissible,'valid':action in admissible,
            'env_feedback':observation,'reward':float(rewards[0]),'done':bool(dones[0]),'won':bool(infos[0].get('won',False)),
            'model_call_id':len(calls),'action_origin':origin,'operator':operator,
            'reasoning':calls[-1].get('reasoning','') if calls and origin=='model' else '',
            'checker_status':event.check.status.value,**ctl.last_execution_audit}
        trace.append(row);save(out/'conversation.json',trace)
        if dones[0]:status='success' if infos[0].get('won',False) or rewards[0]>0 else 'environment_failure';break
        if event.decision.kind.value=='stop_failure':status='interrupt_budget_exhausted';break
    result={'task_id':task['id'],'arm':arm,'status':status,'success':status=='success','initial_sha256':config_hash(initial_record),
        'model_calls':len(calls),'environment_steps':len(trace),'invalid_actions':sum(not r['valid'] for r in trace),
        'format_errors':sum(c['status']=='format_error' for c in calls),'api_errors':sum(c['status']=='api_error' for c in calls),
        'compiled_search_actions':sum(r['action_origin']=='compiled_public_search' for r in trace),
        'learned_value_decisions':len(getattr(ctl,'value_decisions',[])),'learned_interrupts':getattr(ctl,'value_interrupts',0),
        'interrupt_count':ctl.interruption_count,'wall_seconds':time.perf_counter()-started,
        'search_public_state':search.public_state(),'method':spec['method'],'replicate':task['replicate'],
        **{key:sum((c.get('usage') or {}).get(key,0) or 0 for c in calls) for key in ('prompt_tokens','completion_tokens')}}
    result['utility']=float(result['success'])-.04*result['model_calls']-.002*result['environment_steps']
    save(out/'result.json',result)
    return result



def prepare():
    old=read(Path('results/paper_methods_unseen_confirmation_20261007/config.json'))
    tasks=[]
    for rep in (1,2):
        for t in old['tasks']:
            tasks.append({**t,'id':'attribution-r'+str(rep)+'-'+t['id'],'replicate':rep,'physical_task_id':t['id']})
    arms={k:{**v,'workflow':WORKFLOWS['W2_acquire'],'learned_value':False} for k,v in ARMS.items()}
    order=[]
    for i,t in enumerate(tasks):
        keys=list(arms);offset=i%len(keys);keys=keys[offset:]+keys[:offset]
        if t['replicate']==2:keys.reverse()
        order.extend([[t['id'],a] for a in keys])
    cfg={**old,'protocol':'equal-operator-attribution-v1','stage':'attribution','tasks':tasks,'arms':arms,'job_order':order,
        'original_skill':Path('skillopt/ckpt/alfworld/gpt5.5_skill.md').read_text(),
        'source_hashes':{**sources(),**{p:digest(Path(p)) for p in ['experiments/equal_operator_attribution.py','tests/test_equal_operator_attribution.py',
          'skillopt/ckpt/alfworld/gpt5.5_skill.md','skillopt/skillopt/envs/alfworld/rollout.py',
          'skillopt/skillopt/envs/alfworld/vendor/env_manager.py','skillopt/skillopt/envs/alfworld/prompts/rollout_no_history.md',
          'skillopt/skillopt/envs/alfworld/prompts/rollout_with_history.md']}},
        'input_hashes':{str(Path('results/paper_methods_unseen_confirmation_20261007/config.json')):digest(Path('results/paper_methods_unseen_confirmation_20261007/config.json'))},
        'limits':{'batch_usd':12,'max_requests':320,'max_new_unknown':2,'seconds':14400},
        'gain_gate':'Paired physical-task results reported by repeat. Increment requires success wins without losses or equal successes with fewer calls and no higher dollars. Mechanism claim additionally requires actual multi-action use / actual pending-tail interruption. No learned value enabled. Compare dynamic vs matched_single/fixed separately from original baselines.',
        'note':'All four previously exposed confirmation tasks, two repeats, no new independent generalization claim. Same frozen W2 operator and full shared public operator state for all arms; same model, temperature, max_tokens, eight calls and forty primitive steps. Original ReAct uses upstream environment text and no skill. Original SkillOpt inference uses upstream full released GPT5.5 checkpoint transferred to Qwen, not a newly trained checkpoint or original-paper end-to-end replication. Strict parse, no hidden retries/look fallback, bounded short summary appended equally. Macros disabled for every arm; original inference history remains upstream two steps, matched ablations use identical recent-eight prompt.'}
    freeze(ROOT,cfg);print('PREPARED',len(order),'jobs',flush=True)


def summarize():
    return summary(ROOT)


def advance_one():
    with (CAMPAIGN/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);cfg,budget=open_batch(ROOT)
        for task_id,arm in cfg['job_order']:
            out=ROOT/'online'/task_id/arm
            if out.exists():
                assert (out/'result.json').exists(),'Audit incomplete job; never retry silently'
                continue
            task=next(t for t in cfg['tasks'] if t['id']==task_id)
            assert digest(Path(task['gamefile']))==task['game_sha256']
            env=environment(task['gamefile']);api=ApiPlanner(Settings(**cfg['settings']),system_prompt=system_for(cfg['arms'][arm]))
            try:result=run_episode(env,task,arm,cfg,BudgetPlanner(api,budget,str(out)),out)
            finally:api.close();env.close()
            result['cost_usd']=label_cost(ROOT,str(out));save(out/'result.json',result);summarize();return
        summarize()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','authorize','advance','summary']);a=p.parse_args().action
    if a=='prepare':prepare()
    elif a=='authorize':authorize(ROOT)
    elif a=='advance':advance_one()
    else:summarize()
