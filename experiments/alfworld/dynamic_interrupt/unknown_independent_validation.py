"""Prospective new-task validation of frozen UNKNOWN recovery under USD95."""
import argparse
import copy
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import time

from experiments import queue_revision_transfer_trial as prior
from experiments.bounded_research95 import freeze,authorize,open_batch,batch_accounting,label_cost,auth_key
from experiments.interrupt_recovery_ablation import read,digest,config_hash
from experiments.puzzle_pilot import ApiPlanner,Settings,save
from experiments.prefix_recovery_trial import environment
from dynamic_skill.campaign_budget import BudgetPlanner,CampaignHalt
from dynamic_skill.control_only_evolution import ControlPolicy,ControlExecutor,run_control_episode
from dynamic_skill.alfworld_recovery import recovery_prompt
from dynamic_skill.micro_loop_protocol import make_prompt,parse_response
from dynamic_skill.space_subskills import INSTRUCTIONS,SYSTEM_SKILL,expand,normalize_response
from dynamic_skill.queue_revision import revise_head,Truth
from dynamic_skill.queue_evidence import AlfworldEvidence

ROOT=Path('results/unknown_independent_validation_v2_20261007')
ARMS=prior.ARMS
replay_prefix=prior.replay_prefix
BAD={'format_error','api_error','thinking_policy_stop','unknown_cost_stop','budget_stop','deadline_stop','request_cap_stop','api_cooldown','api_error_guard'}


def choose_unknown(trace,evidence):
    """Choose solely from public pre-execution evidence, never following validity."""
    counts={v.value:0 for v in Truth}
    first=None
    for row in trace:
        tail=row.get('remaining_before_interrupt',[])
        if row['done'] or not tail:
            continue
        proof=AlfworldEvidence(row['env_feedback'],evidence[row['step']],complete=False).executable(tail[0])
        counts[proof.value.value]+=1
        if first is None and proof.value is Truth.UNKNOWN:
            first=row
    return first,counts


def environment(gamefile):
    import os
    os.environ.setdefault('ALFWORLD_DATA','/opt/data/private/data/alfworld')
    from skillopt.envs.alfworld.rollout import build_alfworld_env
    return build_alfworld_env(env_num=1,eval_dataset='eval_out_of_distribution',seed=42,is_train=False,specific_gamefiles=[gamefile])


def used_scene_names(paths,candidates):
    used=set()
    for path in paths:
        if 'source_snapshot' in path.parts or path.name not in ('config.json','initial.json','tasks.json'):
            continue
        text=path.read_text()
        used.update(game.parent.parent.name for game in candidates if str(game) in text)
    return used


def select_new_tasks():
    candidates=sorted(Path('/opt/data/private/data/alfworld/json_2.1.1/valid_unseen').glob('*/*/game.tw-pddl'))
    used=used_scene_names(Path('results').rglob('*.json'),candidates)
    tasks=[]
    for family in ('pick_and_place_simple','look_at_obj_in_light'):
        objects=set()
        for game in candidates:
            name=game.parent.parent.name
            if not name.startswith(family+'-') or name in used or name.split('-')[1] in objects:
                continue
            obj=name.split('-')[1];objects.add(obj)
            tasks.append({'id':'independent-unknown-'+family+'-'+str(len(objects)),
                'gamefile':str(game),'game_sha256':digest(game),'benchmark':'alfworld','family':family,'object_type':obj})
            if len(objects)==3:
                break
        assert len(objects)==3
    assert not {Path(t['gamefile']).parent.parent.name for t in tasks}&used
    return tasks


def prepare():
    old=read(Path('results/queue_revision_transfer_20261007/config.json'))
    tasks=select_new_tasks()
    sources=sorted(set(old['source_hashes'])|{'experiments/bounded_research95.py',
        'experiments/unknown_independent_validation.py','tests/test_independent_validation95.py'})
    cfg=copy.deepcopy(old)
    cfg.update(protocol='prospective-unknown-recovery-new-tasks-v2',tasks=tasks,benchmark_split='valid_unseen',
        collection_settings={**old['collection_settings'],'max_calls':8,'max_steps':32},
        settings={**old['settings'],'max_calls':4,'max_steps':12},
        limits={'batch_usd':7,'max_requests':120,'max_new_unknown':2,'seconds':7200},
        selector='First nonterminal UNKNOWN head of a nonempty actual model/expanded queue per task, evaluated against fresh public post-action commands. No outcome or next-action-validity selection. No fallback/control suffixes, no task replacement. All support/false/unknown exposures reported.',
        gain_gate={'min_protocol_valid_triplets':2,'primary':'More successes without losses vs continue, OR equal successes with fewer model calls and no higher aggregate suffix dollar charge. Unknown-head exposure absent or too small => not demonstrated. Illegal/step reduction is secondary, not task gain. Clear is an active recovery comparator, not an independent novel method.'},
        note='Prospective selection fixed before new collection. Frozen queue core and evidence adapters unchanged. UNKNOWN language corrected identically for clear/repair. Same two ALFWorld families, valid_unseen split rather than historical valid_seen; no different-environment claim.')
    cfg['source_hashes']={p:digest(Path(p)) for p in sources}
    freeze(ROOT,cfg)
    print('PREPARED',config_hash(cfg),[(t['id'],t['object_type']) for t in tasks],flush=True)


def freeze_cases(cfg):
    cases,exposures,omissions=[],[],[]
    for task in cfg['tasks']:
        directory=ROOT/'collection'/'predictions'/(cfg['model']+'__'+task['id']+'__fixed_chunk')
        trace=read(directory/'conversation.json') if (directory/'conversation.json').exists() else []
        if not trace:
            omissions.append({'task_id':task['id'],'reason':'no executed trace'})
            continue
        env=environment(task['gamefile'])
        evidence={}
        try:
            env.reset({})
            for row in trace:
                assert list(env.envs.get_admissible_commands[0])==row['admissible_before']
                obs,_,dones,_=env.step(['<think></think><action>'+row['action']+'</action>'])
                assert obs['anchor'][0]==row['env_feedback'] and bool(dones[0])==row['done']
                evidence[row['step']]=list(env.envs.get_admissible_commands[0])
        finally:
            env.close()
        selected,counts=choose_unknown(trace,evidence)
        exposures.append({'task_id':task['id'],**counts,'collection_status':read(directory/'result.json')['status']})
        if selected is None:
            omissions.append({'task_id':task['id'],'reason':'no UNKNOWN nonempty queue'})
            continue
        proof=AlfworldEvidence(selected['env_feedback'],evidence[selected['step']],complete=False).executable(selected['remaining_before_interrupt'][0])
        case={'id':task['id']+'-unknown','source_trace':str(directory/'conversation.json'),
            'source_sha256':digest(directory/'conversation.json'),'source_initial_sha256':digest(directory/'initial.json'),
            'game_sha256':task['game_sha256'],'step':selected['step'],'tail':selected['remaining_before_interrupt'],
            'kind':'unknown_independent','head_verdict':'unknown','reason':'UNKNOWN queued head: '+proof.reason}
        env=environment(task['gamefile'])
        try:
            _,_,checkpoint=replay_prefix(env,case)
        finally:
            env.close()
        case['checkpoint'],case['checkpoint_sha256']=checkpoint,config_hash(checkpoint)
        cases.append(case)
    frozen={'config_sha256':config_hash(cfg),'cases':cases,'exposures':exposures,'omissions':omissions,
        'job_order':[[case['id'],arm] for i,case in enumerate(cases) for arm in ARMS[i%3:]+ARMS[:i%3]]}
    save(ROOT/'cases.json',frozen)
    print('FROZEN_UNKNOWN_PREFIXES',len(cases),'exposures',exposures,flush=True)

def run_suffix(env, case, arm, cfg, planner, out):
    initial, history, checkpoint = replay_prefix(env, case)
    assert config_hash(checkpoint) == case['checkpoint_sha256']
    save(out / 'checkpoint.json', checkpoint)
    save(out / 'prefix.json', history)
    policy = ControlPolicy(**cfg['policy'])
    ctl = ControlExecutor(policy, dynamic=True, state_check=True)
    ctl.start(initial['observation'], initial['task'])
    ctl.action_history = [r['action'] for r in history]
    ctl.observation_history = [initial['observation']] + [r['env_feedback'] for r in history]
    for i, step in enumerate(history):
        ctl.monitor.check(step['action'], step['env_feedback'], [], i,
                          {'admissible': step['admissible_before']})
    original_tail = arm != 'interrupt_replan'
    if original_tail:
        ctl.dynamic = False
        ctl.load_chunk(case['tail'])
    else:
        ctl.last_replan_reason = case['reason'] + '; queued action lacks fresh support: UNKNOWN evidence, not a proven contradiction or an executed failure'
    observation = checkpoint['observation']
    calls, suffix, revisions = [], [], []
    repair_active = arm == 'evidence_repair'
    settings = Settings(**cfg['settings'])
    status = 'step_limit'
    started = time.perf_counter()
    for local_step in range(settings.max_steps):
        if time.perf_counter() - started >= settings.episode_seconds:
            status = 'wall_time_limit'
            break
        admissible = list(env.envs.get_admissible_commands[0])
        if repair_active and ctl.pending_actions:
            revision = revise_head(list(ctl.pending_actions), AlfworldEvidence(observation, admissible, complete=False))
            revisions.append({'suffix_step': len(suffix), 'revision': asdict(revision)})
            save(out / 'queue_revisions.json', revisions)
            if revision.decision == 'skip_head':
                assert ctl.pending_actions.popleft() == revision.skipped[0]
            elif revision.decision == 'replan':
                discarded = list(ctl.pending_actions)
                ctl.pending_actions.clear()
                revisions[-1]['discarded_for_replan'] = discarded
                save(out / 'queue_revisions.json', revisions)
                ctl.last_replan_reason = case['reason'] + '; queued action lacks fresh support: UNKNOWN evidence, not a proven contradiction or an executed failure'
                ctl.dynamic = True
                original_tail = repair_active = False
        if ctl.needs_plan:
            ctl.dynamic = True
            if len(calls) >= settings.max_calls:
                status = 'model_call_limit'
                break
            limit = min(policy.max_actions, settings.max_steps - len(suffix))
            context = {'task': initial['task'], 'initial': initial['observation'],
                       'observation': observation, 'admissible': admissible,
                       'trace': history, 'anomaly': ctl.last_replan_reason}
            prompt = make_prompt(cfg['task_skill'], limit, 'Replan from the current observation.')('', **context) + INSTRUCTIONS
            if context['anomaly']:
                prompt = recovery_prompt(prompt, **context, output_protocol='thought_actions')
            prompt += f'\nPost-prefix budget remaining: {settings.max_calls-len(calls)} model calls and {settings.max_steps-len(suffix)} environment actions.'
            try:
                planner.before_call()
            except CampaignHalt as halt:
                status = halt.status
                break
            call = {'call_id': len(calls) + 1, 'suffix_step': len(suffix), 'prompt': prompt,
                    'system_prompt': SYSTEM_SKILL, 'actions': [], 'action_limit': limit, 'status': 'pending'}
            calls.append(call)
            save(out / 'model_calls.json', calls)
            try:
                call.update(planner(prompt))
                usage = call.get('usage')
                if not isinstance(usage, dict) or not all(isinstance(usage.get(k), int) and usage[k] >= 0
                    for k in ('prompt_tokens', 'completion_tokens')):
                    status = call['status'] = 'unknown_cost_stop'
                    break
                details = (call.get('usage') or {}).get('completion_tokens_details') or {}
                if call.get('reasoning_content') or details.get('reasoning_tokens', 0):
                    status = call['status'] = 'thinking_policy_stop'
                    break
                text, normalized = normalize_response(call['response'])
                actions, reason = parse_response(text, limit)
                primitive, provenance = expand(actions, context, limit)
                call.update(status='ok', protocol_normalized=normalized, model_native_actions=actions,
                            actions=primitive, reasoning=reason, subskill_expansion=provenance)
                ctl.load_chunk(primitive)
            except ValueError as exc:
                status = call['status'] = 'format_error'
                call['parse_error'] = str(exc)
                break
            except Exception as exc:
                status = call['status'] = 'api_error'
                call['error_type'] = type(exc).__name__
                break
            finally:
                save(out / 'model_calls.json', calls)
            print(case['id'], arm, 'call', len(calls), 'suffix_steps', len(suffix), call['status'], flush=True)
        action = ctl.next_action()
        is_original_tail = original_tail
        ctl.before_action(action, observation, admissible)
        obs, rewards, dones, infos = env.step(['<think></think><action>' + action + '</action>'])
        observation = obs['anchor'][0]
        ctl.after_action(list(env.envs.get_admissible_commands[0]))
        event = ctl.observe(action, observation, reward=float(rewards[0]), done=bool(dones[0]), info=infos[0])
        row = {'step': len(history), 'suffix_step': len(suffix), 'action': action,
               'admissible_before': admissible, 'valid': action in admissible,
               'env_feedback': observation, 'reward': float(rewards[0]), 'done': bool(dones[0]),
               'won': bool(infos[0].get('won', False)), 'model_call_id': len(calls),
               'original_tail_action': is_original_tail, 'checker_status': event.check.status.value,
               'reasoning': calls[-1].get('reasoning', '') if calls else '',
               **ctl.last_execution_audit}
        history.append(row)
        suffix.append(row)
        save(out / 'suffix_trace.json', suffix)
        if original_tail and not ctl.pending_actions:
            original_tail = repair_active = False
            ctl.dynamic = True
            if event.check.should_interrupt:
                ctl.last_replan_reason = event.check.reason
        if dones[0]:
            status = 'success' if infos[0].get('won', False) or rewards[0] > 0 else 'environment_failure'
            break
        if event.decision.kind.value == 'stop_failure':
            status = 'interrupt_budget_exhausted'
            break
    result = {'case_id': case['id'], 'arm': arm, 'status': status, 'success': status == 'success',
              'checkpoint_sha256': config_hash(checkpoint), 'prefix_steps': len(read(out / 'prefix.json')),
              'model_calls': len(calls), 'suffix_steps': len(suffix),
              'invalid_actions': sum(r['checker_status'] == 'invalid_action' for r in suffix),
              'initial_discarded_actions': case['tail'] if arm == 'interrupt_replan' else [],
              'subsequent_interrupt_count': ctl.interruption_count,
              'source_kind': case['kind'], 'original_tail_executed': sum(r['original_tail_action'] for r in suffix),
              'format_errors': sum(c['status'] == 'format_error' for c in calls),
              'queue_revision_count': len(revisions),
              'skipped_falsified_heads': sum(len(r['revision']['skipped']) for r in revisions),
              'repair_initial_decision': revisions[0]['revision']['decision'] if revisions else None,
              'revision_replan_discarded': sum(len(r.get('discarded_for_replan', [])) for r in revisions)}
    save(out / 'result.json', result)
    return result



def summary():
    cfg=read(ROOT/'config.json')
    frozen=read(ROOT/'cases.json') if (ROOT/'cases.json').exists() else {'cases':[],'job_order':[],'exposures':[],'omissions':[]}
    rows=[read(p) for p in ROOT.glob('online/*/*/result.json')]
    pairs=[]
    for case in frozen['cases']:
        group={r['arm']:r for r in rows if r['case_id']==case['id']}
        pairs.append({'case_id':case['id'],'complete':len(group)==3,'checkpoint_identical':len({r['checkpoint_sha256'] for r in group.values()})==1,
            'protocol_valid':len(group)==3 and all(r['status'] not in BAD for r in group.values()),'arms':group})
    valid=[p for p in pairs if p['protocol_valid']]
    totals={a:{'successes':sum(p['arms'][a]['success'] for p in valid),
        **{k:sum(p['arms'][a][k] for p in valid) for k in ('model_calls','suffix_steps','invalid_actions','cost_usd')}} for a in ARMS}
    wins=sum(p['arms']['evidence_repair']['success'] and not p['arms']['continue_tail']['success'] for p in valid)
    losses=sum(p['arms']['continue_tail']['success'] and not p['arms']['evidence_repair']['success'] for p in valid)
    calls_gain=totals['evidence_repair']['successes']==totals['continue_tail']['successes'] and totals['evidence_repair']['model_calls']<totals['continue_tail']['model_calls'] and totals['evidence_repair']['cost_usd']<=totals['continue_tail']['cost_usd']
    gain=len(valid)>=cfg['gain_gate']['min_protocol_valid_triplets'] and ((wins>0 and losses==0) or calls_gain)
    report={'attempted':len(rows),'scheduled':len(frozen['job_order']),'complete':bool((ROOT/'cases.json').exists()) and len(rows)==len(frozen['job_order']),
        'new_collection_tasks':len(cfg['tasks']),'unknown_prefixes':len(frozen['cases']),'protocol_valid_triplets':len(valid),
        'exposures':frozen['exposures'],'omissions':frozen['omissions'],'valid_triplet_totals':totals,'wins':wins,'losses':losses,
        'task_gain_demonstrated':gain,'gain_gate':cfg['gain_gate'],'pairs':pairs,'batch':batch_accounting(ROOT)}
    save(ROOT/'summary.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('pairs','exposures','omissions','gain_gate')},ensure_ascii=False),flush=True)
    return report


def advance():
    with (prior.CAMPAIGN/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg,budget=open_batch(ROOT)
        for task in cfg['tasks']:
            directory=ROOT/'collection'/'predictions'/(cfg['model']+'__'+task['id']+'__fixed_chunk')
            if directory.exists():
                assert (directory/'result.json').exists(),'incomplete prior collection; audit required'
                continue
            assert digest(Path(task['gamefile']))==task['game_sha256']
            env=environment(task['gamefile']);api=ApiPlanner(Settings(**cfg['collection_settings']),system_prompt=SYSTEM_SKILL)
            try:
                result=run_control_episode(env,task,Settings(**cfg['collection_settings']),ROOT/'collection',
                    BudgetPlanner(api,budget,str(directory)),cfg['task_skill'],ControlPolicy(**cfg['policy']),
                    dynamic=False,history_evidence=True,state_check=False,macro_mode=True)
            finally:
                api.close();env.close()
            result['cost_usd']=label_cost(ROOT,str(directory));save(directory/'result.json',result)
            return
        if not (ROOT/'cases.json').exists():
            freeze_cases(cfg)
            data=budget.read();data[auth_key(ROOT)]['cases_sha256']=digest(ROOT/'cases.json');budget.write(data)
        frozen=read(ROOT/'cases.json');assert digest(ROOT/'cases.json')==budget.read()[auth_key(ROOT)]['cases_sha256']
        for case_id,arm in frozen['job_order']:
            out=ROOT/'online'/case_id/arm
            if out.exists():
                assert (out/'result.json').exists(),'incomplete suffix; audit required'
                continue
            case=next(c for c in frozen['cases'] if c['id']==case_id)
            env=environment(read(Path(case['source_trace']).parent/'initial.json')['gamefile'])
            api=ApiPlanner(Settings(**cfg['settings']),system_prompt=SYSTEM_SKILL)
            try:
                result=run_suffix(env,case,arm,cfg,BudgetPlanner(api,budget,str(out)),out)
            finally:
                api.close();env.close()
            result['cost_usd']=label_cost(ROOT,str(out));save(out/'result.json',result)
            summary()
            return
        summary()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','authorize','advance','summary'])
    args=parser.parse_args()
    {'prepare':prepare,'authorize':lambda:authorize(ROOT),'advance':advance,'summary':summary}[args.action]()
