"""Two paper-inspired controls, fixed LLM skill, explicit structural-search cost."""
import argparse
import copy
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import time
from experiments.alfworld_campaign import CAMPAIGN
from experiments.interrupt_recovery_ablation import read,digest,config_hash
from experiments.puzzle_pilot import ApiPlanner,Settings,save
from experiments.bounded_research95 import freeze,authorize,open_batch,label_cost,batch_accounting
from experiments.unknown_independent_validation import environment,used_scene_names
from dynamic_skill.campaign_budget import BudgetPlanner,CampaignHalt
from dynamic_skill.control_only_evolution import ControlPolicy,ControlExecutor
from dynamic_skill.paper_control import ValueExecutor
from dynamic_skill.observable_search_operator import ObservableSearch
from dynamic_skill.value_termination import features,utility,fit
from dynamic_skill.micro_loop_protocol import make_prompt,parse_response
from dynamic_skill.space_subskills import INSTRUCTIONS,SYSTEM_SKILL,expand,normalize_response
from dynamic_skill.alfworld_recovery import recovery_prompt

DEV=Path('results/paper_workflow_tree_dev_20261007')
TEST=Path('results/paper_methods_unseen_test_20261007')
MODEL=Path('results/paper_value_model_20261007/model.json')
DESIGN=Path('results/paper_value_model_20261007/design.json')
WORKFLOWS={
 'W0_rule':{'search':False,'acquire':False,'priority':'surface_first','max_actions':4},
 'W1_frontier':{'search':True,'acquire':False,'priority':'surface_first','max_actions':4},
 'W2_acquire':{'search':True,'acquire':True,'priority':'surface_first','max_actions':4},
 'W3_balanced_boundary':{'search':True,'acquire':True,'priority':'balanced','max_actions':2}}
BAD={'format_error','api_error','thinking_policy_stop','unknown_cost_stop','budget_stop','deadline_stop','request_cap_stop','api_cooldown','api_error_guard'}


def train_value():
    assert not MODEL.exists()
    roots=[Path('results')/s for s in ('prefix_recovery_20261007','queue_revision_transfer_20261007','queue_revision_opportunity_20261007','queue_revision_unknown_20261007')]
    rows=[];inputs={}
    for root in roots:
        cfg=read(root/'config.json')
        frozen=read(root/'cases.json') if (root/'cases.json').exists() else cfg
        for case in frozen['cases']:
            paths=[root/'online'/case['id']/a/'result.json' for a in ('continue_tail','interrupt_replan')]
            if not all(p.exists() for p in paths):continue
            a,b=map(read,paths)
            if a['status'] in BAD or b['status'] in BAD:continue
            checkpoint=read(paths[0].parent/'checkpoint.json')
            assert a['checkpoint_sha256']==b['checkpoint_sha256']==config_hash(checkpoint)
            history=read(paths[0].parent/'prefix.json')
            vector=features(checkpoint['task'],checkpoint['observation'],checkpoint['admissible'],history,case['tail'])
            rows.append({'case_id':root.name+'/'+case['id'],'group':checkpoint['game_sha256'],
                'features':vector,'utilities':[utility(a),utility(b)],'source_root':str(root)})
            for p in [root/'config.json',*paths,paths[0].parent/'checkpoint.json',paths[0].parent/'prefix.json']:
                inputs[str(p)]=digest(p)
    model=fit(rows);model['input_hashes']=inputs
    save(MODEL,model)
    save(DESIGN,{'idea_md':'idea/基于价值学习与树搜索的技能自进化和中断机制设计.md',
        'idea_sha256':digest(Path('idea/基于价值学习与树搜索的技能自进化和中断机制设计.md')),
        'papers':{
          'deliberation_cost':{'url':'https://cdn.aaai.org/ojs/11831/11831-13-15359-1-2-20201228.pdf','adaptation':'Fit two branch utilities with API-call and primitive-action penalties from historical real prefix pairs. Not Option-Critic gradient training; LLM frozen.'},
          'aflow':{'url':'https://proceedings.iclr.cc/paper_files/paper/2025/file/5492ecbce4439401798dcd2c90be94cd-Paper-Conference.pdf','adaptation':'Search a small tree of executable public-observation frontier/acquisition/boundary workflows with real execution scoring. Human-defined operator library, exhaustive constrained tree, no LLM code optimizer or full MCTS.'},
          'lats_considered':{'url':'https://proceedings.mlr.press/v235/zhou24r.html','decision':'Deferred initially: branch rollout cost needs separate fair accounting; use paired real branches to fit value first.'}},
        'objective':'success - .04 * model_calls - .002 * environment_steps; dollars reported separately, not double-penalized',
        'tree_edges':[['W0_rule','W1_frontier'],['W1_frontier','W2_acquire'],['W2_acquire','W3_balanced_boundary']],
        'not_model_native_chunking':'Frontier routine chooses one admitted primitive per feedback step without LLM calls; search/acquisition primitives are labeled program-origin. Final task operations stay with LLM.'})
    print('TRAINED_VALUE',len(rows),'physical_groups',len({r['group'] for r in rows}),'ridge',model['ridge'],flush=True)


def sources():
    old=read(Path('results/unknown_independent_validation_v2_20261007/config.json'))
    paths=set(old['source_hashes'])|{'dynamic_skill/value_termination.py','dynamic_skill/observable_search_operator.py',
        'dynamic_skill/paper_control.py','experiments/paper_method_trial.py','tests/test_paper_controls.py'}
    return {p:digest(Path(p)) for p in sorted(paths)}


def prepare_dev():
    assert read(Path('results/unknown_independent_validation_v2_20261007/summary.json'))['complete']
    assert not read(Path('results/unknown_independent_validation_v2_20261007/summary.json'))['task_gain_demonstrated']
    if not MODEL.exists():train_value()
    old=read(Path('results/unknown_independent_validation_v2_20261007/config.json'))
    # Prespecified development failure stratum: all incomplete collector tasks, no outcome filtering of methods.
    tasks=[]
    for task in old['tasks']:
        result=read(Path('results/unknown_independent_validation_v2_20261007/collection/predictions')/(old['model']+'__'+task['id']+'__fixed_chunk')/'result.json')
        if not result['success'] and result['status']=='model_call_limit':tasks.append(task)
    assert len(tasks)==2
    cfg={'protocol':'paper-inspired-workflow-tree-development-v1','stage':'dev','model':old['model'],
        'task_skill':old['task_skill'],'tasks':tasks,'settings':{**old['settings'],'max_calls':6,'max_steps':40,'episode_seconds':300},
        'arms':{key:{'workflow':value,'learned_value':False} for key,value in WORKFLOWS.items()},
        'job_order':[[task['id'],a] for i,task in enumerate(tasks) for a in list(WORKFLOWS)[i:]+list(WORKFLOWS)[:i]],
        'source_hashes':sources(),'input_hashes':{str(MODEL):digest(MODEL),str(DESIGN):digest(DESIGN),**read(MODEL)['input_hashes']},
        'limits':{'batch_usd':5,'max_requests':48,'max_new_unknown':2,'seconds':7200},
        'select':'Highest mean utility across both dev tasks; no selection if protocol/infrastructure missing for any dev task. Ties prefer fewer model calls, then W0 baseline. Independent test determines acceptance.',
        'note':'Two already observed incomplete valid_unseen tasks used for algorithm development only. Four-node hand-constrained workflow tree, evaluated exhaustively. No claimed MCTS or original AFlow reproduction. All search costs counted in development ledger.'}
    freeze(DEV,cfg);print('PREPARED_DEV',flush=True)


def run_episode(env,task,arm,cfg,planner,out):
    out.mkdir(parents=True,exist_ok=False)
    spec=cfg['arms'][arm];workflow=spec['workflow'];settings=Settings(**cfg['settings'])
    policy=ControlPolicy(max_actions=workflow['max_actions'])
    ctl=ValueExecutor(policy,read(MODEL)) if spec['learned_value'] else ControlExecutor(policy,dynamic=True,state_check=True)
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
                prompt=make_prompt(cfg['task_skill'],limit,'Replan from the current observation.')('',**context)+INSTRUCTIONS
                if context['anomaly']:prompt=recovery_prompt(prompt,**context,output_protocol='thought_actions')
                # Identical remaining-resource description in every arm.
                prompt+=f'\nBudget remaining: {settings.max_calls-len(calls)} model calls and {settings.max_steps-step} environment actions.'
                try:planner.before_call()
                except CampaignHalt as halt:status=halt.status;break
                call={'call_id':len(calls)+1,'step':step,'prompt':prompt,'system_prompt':SYSTEM_SKILL,
                    'status':'pending','actions':[],'action_limit':limit};calls.append(call);save(out/'model_calls.json',calls)
                try:
                    call.update(planner(prompt))
                    usage=call.get('usage')
                    if not isinstance(usage,dict) or not all(isinstance(usage.get(k),int) and usage[k]>=0 for k in ('prompt_tokens','completion_tokens')):
                        status=call['status']='unknown_cost_stop';break
                    details=usage.get('completion_tokens_details') or {}
                    if call.get('reasoning_content') or details.get('reasoning_tokens',0):status=call['status']='thinking_policy_stop';break
                    text,normalized=normalize_response(call['response'])
                    actions,reason=parse_response(text,limit);primitive,provenance=expand(actions,context,limit)
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
        'search_public_state':search.public_state(),
        **{key:sum((c.get('usage') or {}).get(key,0) or 0 for c in calls) for key in ('prompt_tokens','completion_tokens')}}
    result['utility']=float(result['success'])-.04*result['model_calls']-.002*result['environment_steps']
    save(out/'result.json',result)
    return result


def summary(root):
    cfg=read(root/'config.json');rows=[read(p) for p in root.glob('online/*/*/result.json')]
    result={'attempted':len(rows),'scheduled':len(cfg['job_order']),'complete':len(rows)==len(cfg['job_order']),
        'arms':{a:{'attempted':sum(r['arm']==a for r in rows),'successes':sum(r['success'] for r in rows if r['arm']==a),
            **{k:sum(r.get(k,0) for r in rows if r['arm']==a) for k in ('model_calls','environment_steps','invalid_actions','cost_usd','compiled_search_actions','learned_value_decisions','learned_interrupts','format_errors','api_errors','utility')}} for a in cfg['arms']},
        'pairs':[{'task_id':t['id'],'arms':{r['arm']:r for r in rows if r['task_id']==t['id']}} for t in cfg['tasks']],
        'batch':batch_accounting(root)}
    save(root/'summary.json',result);print(json.dumps({k:v for k,v in result.items() if k!='pairs'}),flush=True)
    if cfg['stage']=='dev' and result['complete']:
        eligible=[a for a in cfg['arms'] if all(any(r['task_id']==t['id'] and r['arm']==a and r['status'] not in BAD for r in rows) for t in cfg['tasks'])]
        if 'W0_rule' not in eligible:
            selected='W0_rule';reason='Baseline invalid; no development gain can be assessed'
        else:
            selected=max(eligible,key=lambda a:(result['arms'][a]['utility'],-result['arms'][a]['model_calls'],a=='W0_rule'))
            reason='Maximum prespecified utility over all dev tasks, with complete protocol-valid evaluation'
        save(root/'selected_workflow.json',{'selected':selected,'workflow':cfg['arms'][selected]['workflow'],
            'reason':reason,'dev_summary_sha256':digest(root/'summary.json'),'research_cost_usd':result['batch']['cost_usd'],
            'not_accepted_on_test':True,'candidate_tree':read(DESIGN)['tree_edges']})
    return result


def advance(root):
    with (CAMPAIGN/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg,budget=open_batch(root)
        for task_id,arm in cfg['job_order']:
            out=root/'online'/task_id/arm
            if out.exists():
                assert (out/'result.json').exists(),'Incomplete prior job; audit required'
                continue
            task=next(t for t in cfg['tasks'] if t['id']==task_id)
            assert digest(Path(task['gamefile']))==task['game_sha256']
            env=environment(task['gamefile']);api=ApiPlanner(Settings(**cfg['settings']),system_prompt=SYSTEM_SKILL)
            try:result=run_episode(env,task,arm,cfg,BudgetPlanner(api,budget,str(out)),out)
            finally:api.close();env.close()
            result['cost_usd']=label_cost(root,str(out));save(out/'result.json',result);summary(root)
            return
        summary(root)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare-dev','authorize-dev','advance-dev','summary-dev'])
    args=parser.parse_args()
    {'prepare-dev':prepare_dev,'authorize-dev':lambda:authorize(DEV),'advance-dev':lambda:advance(DEV),'summary-dev':lambda:summary(DEV)}[args.action]()
