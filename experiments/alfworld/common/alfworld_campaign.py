"""Serial, budgeted action-only ALFWorld replication; never edits SkillOpt."""
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time

from dynamic_skill.alfworld_rollout import parse_action_chunk
from dynamic_skill.campaign_budget import CampaignBudget, BudgetPlanner, CampaignHalt
from dynamic_skill.interrupter import DecisionType
from dynamic_skill.skill_executor import ChunkExecutor
from experiments.puzzle_pilot import ApiPlanner, Settings, save

CAMPAIGN = Path('results/gpt54_recovery_20260928')
OUT = Path('results/alfworld_actiononly_20260928')
MODELS = ('qwen3.5-35b-a3b', 'gpt-5.4')
MODES = ('react', 'fixed_chunk', 'dynamic_interrupt')
SYSTEM = ('You operate ALFWorld. Output ONLY one <actions>JSON string array</actions> block. '
          'Use concrete environment commands, not descriptions. Do not output reasoning, '
          'analysis, think tags, explanations or markdown. Respect the requested action limit.')


def skill_text():
    raw = Path('skillopt/skillopt/envs/alfworld/skills/initial.md').read_text()
    text = '\n'.join(line for line in raw.splitlines() if not line.startswith('**Output format**'))
    text = text.replace('Actions must be chosen from the admissible action list provided\nat each step.',
        'The first queued action must be currently admissible; later actions must be valid after preceding actions.')
    text = text.replace('Always pick an action from the admissible action list. Do not invent actions.',
        'Pick a currently admissible first action. Later queued actions must have predictable preconditions; never invent object IDs.')
    return text


def tasks():
    rows = [json.loads(s) for s in Path('results/react_inferera_metrics_2/results.jsonl').read_text().splitlines()]
    result = []
    for r in rows:
        relative = 'json_2.1.1/' + r['gamefile'].split('/json_2.1.1/', 1)[1]
        p = Path(os.environ['ALFWORLD_DATA'])/relative
        assert p.is_file()
        result.append({'id':r['id'].replace(':','_'), 'gamefile':str(p.resolve()),
            'game_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
            'task_description':r['task_description'], 'benchmark':'alfworld'})
    assert len(result)==2
    return result


def prompt_for(task, observation, admissible, trace, skill, limit, anomaly):
    history=[{'action':r['action'],'observation':r['env_feedback']} for r in trace[-2:]]
    return (skill + '\n\nTask: ' + task + '\nRecent executed history: ' + json.dumps(history)
        + '\nCurrent observation:\n' + observation
        + '\nCurrently admissible actions:\n' + json.dumps(admissible)
        + f'\nReturn 1 to {limit} concrete actions in order. '
        + ('Return exactly ONE action. ' if limit==1 else
           'Use multiple actions only when their objects and preconditions are known. ')
        + 'Stop before an action requiring new information; never invent object IDs. '
        + 'Queued actions execute without a model call between them. '
        + ('Previous anomaly: '+anomaly+'\n' if anomaly else '')
        + 'Output ONLY <actions>["action"]</actions> with a JSON string array. No reasoning or explanation.')


def run_episode(env, task, mode, settings, output, planner, skill, prompt_transform=None,
                response_parser=None, system_prompt=None, executor=None):
    k=1 if mode=='react' else settings.chunk_size
    ctl=executor or ChunkExecutor(chunk_size=k,dynamic=mode=='dynamic_interrupt')
    obs, infos=env.reset({})
    initial=obs['anchor'][0]
    ctl.start(initial,task=env.tasks[0])
    directory=output/'predictions'/(settings.model+'__'+task['id']+'__'+mode)
    directory.mkdir(parents=True,exist_ok=False)
    save(directory/'initial.json',{'observation':initial,'task':env.tasks[0],
        'admissible':list(env.envs.get_admissible_commands[0]),'gamefile':task['gamefile']})
    calls, trace=[],[]
    started=time.perf_counter()
    status='step_limit'
    for step in range(settings.max_steps):
        if time.perf_counter()-started >= settings.episode_seconds:
            status='wall_time_limit'; break
        called=ctl.needs_plan
        if called:
            if len(calls)>=settings.max_calls:
                status='model_call_limit'; break
            limit=min(k,settings.max_steps-step)
            prompt=prompt_for(env.tasks[0],obs['anchor'][0],list(env.envs.get_admissible_commands[0]),
                trace,skill,limit,ctl.last_replan_reason)
            if prompt_transform is not None:
                prompt=prompt_transform(prompt,task=env.tasks[0],initial=initial,
                    observation=obs['anchor'][0],admissible=list(env.envs.get_admissible_commands[0]),
                    trace=trace,anomaly=ctl.last_replan_reason)
            try:
                planner.before_call()
            except CampaignHalt as halt:
                status=halt.status; break
            record={'call_id':len(calls)+1,'step':step,'prompt':prompt,'system_prompt':system_prompt or SYSTEM,
                'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'status':'pending',
                'actions':[],'response':'','usage':None,'action_limit':limit,
                'generic_anomaly_prompt_used':bool(ctl.last_replan_reason),
                'historical_evidence_injected':bool(ctl.last_replan_reason and
                    'Recovery evidence (historical observations, not current-state guarantees):' in prompt),
                'recovery_prompt_used':bool(ctl.last_replan_reason and
                    'Recovery evidence (historical observations, not current-state guarantees):' in prompt)}
            record.update(getattr(ctl,'planning_context',{}))
            calls.append(record); save(directory/'model_calls.json',calls)
            before=time.perf_counter()
            try:
                record.update(planner(prompt)); record['status']='ok'
            except Exception as exc:
                record.update(status='api_error',error_type=type(exc).__name__)
            record['latency_seconds']=time.perf_counter()-before
            if record['status']=='ok':
                details=(record.get('usage') or {}).get('completion_tokens_details') or {}
                if record.get('reasoning_content') or (details.get('reasoning_tokens') or 0)>0:
                    record['status']='thinking_policy_stop'
                else:
                    try:
                        # Strict new protocol: no narrative/think block or legacy single-action fallback.
                        if response_parser is not None:
                            record['actions'],record['reasoning']=response_parser(record['response'],limit)
                            record.update(getattr(ctl,'response_audit',{}))
                        elif not re.fullmatch(r'\s*<actions>.*?</actions>\s*',record['response'],re.DOTALL):
                            raise ValueError('action-only output contract violated')
                        else:
                            record['actions']=parse_action_chunk(record['response'],limit)
                    except ValueError as exc:
                        record['status']='format_error'
                        record['parse_error']=str(exc)
            save(directory/'model_calls.json',calls)
            print(f"{settings.model} {task['id']} {mode} call={len(calls)} step={step} k={len(record['actions'])} {record['status']}",flush=True)
            if record['status']!='ok':
                status=record['status']; break
            if not isinstance(record.get('usage'),dict):
                status='unknown_cost_stop'; break
            ctl.load_chunk(record['actions'])
        action=ctl.next_action()
        admissible=list(env.envs.get_admissible_commands[0])
        if hasattr(ctl,'before_action'):
            ctl.before_action(action,obs['anchor'][0],admissible)
        valid=action.lower() in {a.lower() for a in admissible}
        # Synthetic empty wrapper satisfies upstream parser; it is not model-generated reasoning.
        obs,rewards,dones,infos=env.step(['<think></think><action>'+action+'</action>'])
        feedback=obs['anchor'][0]
        if hasattr(ctl,'after_action'):
            ctl.after_action(list(env.envs.get_admissible_commands[0]))
        event=ctl.observe(action,feedback,reward=float(rewards[0]),done=bool(dones[0]),info=infos[0])
        trace.append({'step':step,'action':action,'valid':valid,'admissible_before':admissible,
            'reasoning':calls[-1].get('reasoning','') if called else '',
            'env_feedback':feedback,'reward':float(rewards[0]),'done':bool(dones[0]),
            'won':bool(infos[0].get('won',False)), 'model_called':called,'model_call_id':len(calls),
            'chunk_id':ctl.chunk_id,'chunk_action_index':ctl.actions_in_chunk-1,
            'remaining_actions':list(ctl.pending_actions),'discarded_actions':list(event.discarded_actions),
            'checker_status':event.check.status.value,'controller_decision':event.decision.kind.value,
            'controller_reason':event.decision.reason,'interrupt_count':ctl.interruption_count})
        trace[-1].update(getattr(ctl,'last_execution_audit',{}))
        save(directory/'conversation.json',trace)
        if dones[0]:
            status='success' if infos[0].get('won',False) or rewards[0]>0 else 'environment_failure'; break
        if event.decision.kind is DecisionType.STOP_FAILURE:
            status='interrupt_budget_exhausted'; break
    usage={key:sum((c.get('usage') or {}).get(key,0) or 0 for c in calls)
           for key in ('prompt_tokens','completion_tokens','total_tokens')}
    result={'id':directory.name,'task_id':task['id'],'benchmark':'alfworld','model':settings.model,
        'mode':mode,'status':status,'success':status=='success','environment_steps':len(trace),
        'model_calls':len(calls),'model_errors':sum(c['status']=='api_error' for c in calls),
        'format_errors':sum(c['status']=='format_error' for c in calls),
        'invalid_actions':sum(not r['valid'] for r in trace),
        'interrupt_count':ctl.interruption_count,'anomaly_replans':ctl.anomaly_replan_count,
        'multi_action_chunks':sum(len(c['actions'])>1 for c in calls),
        'planned_actions':sum(len(c['actions']) for c in calls),
        'wall_seconds':time.perf_counter()-started,'usage_incomplete':any(c.get('usage') is None for c in calls),
        'latency_total_seconds':sum(c['latency_seconds'] for c in calls),**usage}
    save(directory/'result.json',result)
    print(f"FINISHED {directory.name}: {status}, {len(trace)} steps, {len(calls)} calls",flush=True)
    return result


def run_infra_retries():
    """One separately recorded attempt for each infrastructure-aborted episode."""
    from skillopt.envs.alfworld.rollout import build_alfworld_env
    retry_out=OUT/'infra_retry_1'
    with (CAMPAIGN/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        budget=CampaignBudget(CAMPAIGN/'budget.json',deadline=0)
        state=budget.read()
        assert state.get('alfworld_twenty_authorization') and state['limit_usd']==20
        assert not state.get('alfworld_infra_retry_authorization')
        assert not any(r['status']=='pending' for r in state['requests'])
        config=json.loads((OUT/'config.json').read_text())
        original=[json.loads(s) for s in (OUT/'results.jsonl').read_text().splitlines()]
        assert len(original)==12, 'finish unattempted jobs first'
        aborted={r['id'] for r in original if r['status'] in {'api_error','budget_stop'}}
        by_task={t['id']:t for t in config['tasks']}
        jobs=[(by_task[t],m,model) for t,m,model in config['run_order'] if model+'__'+t+'__'+m in aborted]
        retry_out.mkdir(exist_ok=False)
        state['alfworld_infra_retry_authorization']={'at_unix':time.time(),
            'reason':'Continue user-authorized $20 ALFWorld evaluation; one separate retry of infrastructure-aborted tasks only.',
            'previous_requests':len(state['requests']),'jobs':sorted(aborted)}
        state['max_requests']=len(state['requests'])+len(jobs)*50
        state['deadline_unix']=time.time()+3600
        budget.write(state)
        save(retry_out/'config.json',{**config,'retry_of':str(OUT),
            'run_order':[[t['id'],m,model] for t,m,model in jobs]})
        settings=Settings(**config['settings']); rows=[]
        try:
            for task,mode,model in jobs:
                settings.model=model
                env=build_alfworld_env(env_num=1,eval_dataset='eval_in_distribution',seed=42,
                    is_train=False,specific_gamefiles=[task['gamefile']])
                api=ApiPlanner(settings,system_prompt=SYSTEM)
                try:
                    planner=BudgetPlanner(api,budget,str(retry_out)+'/'+model+'/'+task['id']+'/'+mode)
                    result=run_episode(env,task,mode,settings,retry_out,planner,config['skill'])
                finally:
                    api.close(); env.close()
                rows.append(result)
                with (retry_out/'results.jsonl').open('a') as handle:
                    handle.write(json.dumps(result)+'\n')
                if result['status'] in {'api_error','budget_stop','unknown_cost_stop','thinking_policy_stop','deadline_stop','api_cooldown','request_cap_stop'}:
                    break
        finally:
            state=budget.read(); state['max_requests']=len(state['requests']); budget.write(state)
            save(retry_out/'budget_snapshot.json',state)
            save(retry_out/'run_status.json',{'status':'finished' if len(rows)==len(jobs) else 'stopped',
                'episodes':len(rows),'scheduled_episodes':len(jobs)})


def authorize(budget):
    data=budget.read()
    assert len(data['requests'])>=56 and not data.get('alfworld_authorization')
    assert not any(r['status']=='pending' for r in data['requests'])
    now=time.time()
    data['alfworld_authorization']={'at_unix':now,'old_deadline':data['deadline_unix'],
        'old_max_requests':data['max_requests'],'reason':'User requested new-protocol ALFWorld two-model three-mode evaluation; shared remaining $5 budget.',
        'previous_requests':len(data['requests']),
        'previous_charge':sum(r['budget_charge_usd'] for r in data['requests']),
        'additional_request_cap':600,'duration_seconds':3600}
    data['max_requests']=len(data['requests'])+600; data['deadline_unix']=now+3600
    budget.write(data)


def authorize_resume(budget, remaining):
    data=budget.read()
    assert not data.get('alfworld_resume_authorization')
    assert not any(r['status']=='pending' for r in data['requests'])
    now=time.time()
    data['alfworld_resume_authorization']={'at_unix':now,
        'reason':'User reports reseller usage about $0.40 and explicitly authorizes remaining ALFWorld jobs; retain existing $5 accounting.',
        'previous_requests':len(data['requests']), 'previous_max_unknown':data['max_unknown_requests'],
        'previous_deadline':data['deadline_unix'], 'remaining_episodes':remaining}
    data['max_unknown_requests']=sum(r['status']=='unknown_usage' for r in data['requests'])+1
    data['max_requests']=len(data['requests'])+remaining*50
    data['deadline_unix']=now+3600
    budget.write(data)


def authorize_billing_resume(budget, remaining):
    data=budget.read()
    assert not data.get('billing_checkpoint'), 'single-use user authorization'
    assert not any(r['status']=='pending' for r in data['requests'])
    now=time.time()
    data['billing_checkpoint']={'at_unix':now,'user_reported_total_usd':0.44,
        'source':'User explicitly confirmed current reseller cumulative usage $0.44 and requested continuation.',
        'request_count':len(data['requests']),
        'unsettled_reserve_usd':sum(r['budget_charge_usd'] for r in data['requests'] if r['status']=='unknown_usage'),
        'prior_raw_estimate_usd':sum(r['budget_charge_usd'] for r in data['requests']),
        'note':'User-reported, not independently verified. Retain old unknown reserves for delayed billing; all original requests preserved.'}
    data['max_unknown_requests']=sum(r['status']=='unknown_usage' for r in data['requests'])+1
    data['max_requests']=len(data['requests'])+remaining*50
    data['deadline_unix']=now+3600
    budget.write(data)


def authorize_twenty_resume(budget, remaining):
    data=budget.read()
    assert not data.get('alfworld_twenty_authorization'), 'single-use explicit authorization'
    assert not any(r['status']=='pending' for r in data['requests'])
    now=time.time()
    data.setdefault('billing_checkpoint_history',[]).append(data.get('billing_checkpoint'))
    data['alfworld_twenty_authorization']={'at_unix':now,'previous_limit':data['limit_usd'],
        'reason':'User explicitly raised total budget to $20 and reported cumulative reseller usage $0.65; continue remaining ALFWorld jobs.',
        'remaining_episodes':remaining,'previous_requests':len(data['requests'])}
    data['billing_checkpoint']={'at_unix':now,'user_reported_total_usd':0.65,
        'source':'Explicit user report, not independently verified provider billing.',
        'request_count':len(data['requests']),
        'unsettled_reserve_usd':sum(r['budget_charge_usd'] for r in data['requests'] if r['status']=='unknown_usage'),
        'prior_raw_estimate_usd':sum(r['budget_charge_usd'] for r in data['requests'])}
    data['limit_usd']=20.0
    data['max_unknown_requests']=sum(r['status']=='unknown_usage' for r in data['requests'])+1
    data['max_requests']=len(data['requests'])+remaining*50
    data['deadline_unix']=now+3600
    budget.write(data)


def run(resume=False, billing_resume=False, twenty_resume=False):
    from skillopt.envs.alfworld.rollout import build_alfworld_env
    with (CAMPAIGN/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert (CAMPAIGN/'budget.json').exists()
        budget=CampaignBudget(CAMPAIGN/'budget.json',deadline=0)
        task_list=tasks(); skill=skill_text()
        settings=Settings(max_steps=50,max_calls=50,max_tokens=512,request_timeout=45,
                          episode_seconds=600,action_only=True)
        jobs=[(t,m,model) for t in task_list for m in MODES for model in MODELS]
        random.Random(20260928).shuffle(jobs)
        rows=[]
        if resume or billing_resume or twenty_resume:
            config=json.loads((OUT/'config.json').read_text())
            assert skill==config['skill'] and task_list==config['tasks']
            assert [[t['id'],m,model] for t,m,model in jobs]==config['run_order']
            rows=[json.loads(s) for s in (OUT/'results.jsonl').read_text().splitlines()]
            finished={r['id'] for r in rows}
            assert len(finished)==len(rows)
            jobs=[(t,m,model) for t,m,model in jobs if model+'__'+t['id']+'__'+m not in finished]
            if twenty_resume:
                authorize_twenty_resume(budget,len(jobs))
                save(OUT/'twenty_authorization.json',budget.read()['alfworld_twenty_authorization'])
            elif billing_resume:
                authorize_billing_resume(budget,len(jobs))
                save(OUT/'billing_checkpoint.json',budget.read()['billing_checkpoint'])
            else:
                authorize_resume(budget,len(jobs))
                save(OUT/'resume_authorization.json',budget.read()['alfworld_resume_authorization'])
        else:
            OUT.mkdir(exist_ok=False); authorize(budget)
            save(OUT/'config.json',{'protocol':'alfworld-actiononly-v3','settings':asdict(settings),
            'tasks':task_list,'run_order':[[t['id'],m,model] for t,m,model in jobs],
            'skill':skill,'skill_sha256':hashlib.sha256(skill.encode()).hexdigest(),
            'system_prompt':SYSTEM,'budget_ledger':str(budget.path),'seed':42,
            'qwen_wire':{'enable_thinking':False},'gpt_wire':{'reasoning_effort':'none'},
            'workers':1,'sdk_retries':0})
        try:
            for task,mode,model in jobs:
                settings.model=model
                env=build_alfworld_env(env_num=1,eval_dataset='eval_in_distribution',seed=42,
                    is_train=False,specific_gamefiles=[task['gamefile']])
                api=ApiPlanner(settings,system_prompt=SYSTEM)
                try:
                    planner=BudgetPlanner(api,budget,str(OUT)+'/'+model+'/'+task['id']+'/'+mode)
                    result=run_episode(env,task,mode,settings,OUT,planner,skill)
                finally:
                    api.close(); env.close()
                rows.append(result)
                with (OUT/'results.jsonl').open('a') as h: h.write(json.dumps(result)+'\n')
                state=budget.read()
                if result['status'] not in {'success','format_error','step_limit','model_call_limit','wall_time_limit',
                                            'environment_failure','interrupt_budget_exhausted'} or state['policy_violation'] or state['requests'][-1]['status']!='returned_usage':
                    break
        finally:
            state=budget.read(); state['max_requests']=len(state['requests'])
            state['alfworld_authorization']['completed_episodes']=len(rows)
            budget.write(state); save(OUT/'budget_snapshot.json',state)
            save(OUT/'run_status.json',{'status':'finished' if len(rows)==12 else 'stopped',
                'episodes':len(rows),'scheduled_episodes':12})
