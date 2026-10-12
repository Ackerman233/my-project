"""Explicitly authorized continuation; one separate infrastructure retry, no overwrite."""
import fcntl
import hashlib
import json
from pathlib import Path
import time
from dynamic_skill.campaign_budget import CampaignBudget, BudgetPlanner
from experiments.brief_protocol_pair import ROOT as ORIGINAL, brief_prompt
from experiments.paper_cold_start_gate import paper_prompt
from experiments.puzzle_pilot import ApiPlanner, Settings, run_episode, save
from experiments.cold_start_gate import gate

ROOT = Path('results/brief_protocol_resume_20260928')
AUTH = 'brief_protocol_resume_authorization'


def plan_jobs(config, previous):
    seen={(r['task_id'],a):r for a,rows in previous.items() for r in rows}
    missing=[(t,a,'unattempted') for t,a in config['run_order'] if (t,a) not in seen]
    retry=[(t,a,'infra_retry_once') for t,a in config['run_order']
           if (t,a) in seen and seen[t,a]['status']=='api_error']
    if retry != [('brief-sudoku-2','brief_reason','infra_retry_once')]:
        raise RuntimeError('unexpected retry target; inspect before spending')
    return missing+retry


def run():
    cfg=json.loads((ORIGINAL/'config.json').read_text())
    previous=json.loads((ORIGINAL/'results.json').read_text())
    jobs=plan_jobs(cfg,previous)
    if ROOT.exists(): raise RuntimeError('continuation already attempted; do not repeat')
    path=Path('results/gpt54_recovery_20260928/budget.json')
    with (path.parent/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        budget=CampaignBudget(path,deadline=0); state=budget.read()
        if (state.get(AUTH) or state['policy_violation'] or time.time()<state['cooldown_until']
            or any(r['status']=='pending' for r in state['requests'])):
            raise RuntimeError('authorization/policy/cooldown/pending guard closed')
        if state['limit_usd']!=20 or budget.effective_spend(state)+state['reservation_usd']>20:
            raise RuntimeError('existing budget unavailable')
        state[AUTH]={'reason':'User explicitly authorized remaining tasks and one separately recorded retry of the timed-out task.',
                    'at_unix':time.time(),'previous_requests':len(state['requests']),
                    'previous_charge':budget.effective_spend(state),'jobs':jobs,
                    'old_deadline':state['deadline_unix'],'old_max_requests':state['max_requests'],
                    'old_max_unknown':state['max_unknown_requests']}
        state['max_requests']=len(state['requests'])+sum(cfg['settings'][a]['max_calls'] for _,a,_ in jobs)
        state['max_unknown_requests']=sum(r['status']=='unknown_usage' for r in state['requests'])+1
        state['deadline_unix']=time.time()+1800
        budget.write(state); ROOT.mkdir()
        save(ROOT/'config.json',{**cfg,'resume_of':str(ORIGINAL),'resume_jobs':jobs,
            'original_config_sha256':hashlib.sha256((ORIGINAL/'config.json').read_bytes()).hexdigest(),
            'note':'Exact frozen original settings/skills/tasks. Unattempted jobs keep original relative order; one infra retry appended last.'})
        rows={a:[] for a in cfg['settings']}
        try:
            for tid,arm,kind in jobs:
                task=next(t for t in cfg['tasks'] if t['id']==tid)
                settings=Settings(**cfg['settings'][arm])
                api=ApiPlanner(settings,action_limit=1)
                planner=BudgetPlanner(api,budget,str(ROOT)+'/'+arm+'/'+tid+'/'+kind)
                print(f'START {arm} {tid} {kind}',flush=True)
                try:
                    row=run_episode(task,'react',settings,ROOT/arm,planner,
                        base_skill_text=cfg['base_skills'][task['benchmark']],
                        prompt_builder=paper_prompt if arm=='action_only' else brief_prompt)
                finally: api.close()
                row['attempt_kind']=kind
                rows[arm].append(row); save(ROOT/'results.json',rows)
                state=budget.read()
                if (row['status'] not in {'success','step_limit','model_call_limit','format_error'}
                    or state['policy_violation'] or state['consecutive_errors']): break
        finally:
            save(ROOT/'gate.json',{a:gate(r,cfg['tasks']) for a,r in rows.items()})
            state=budget.read(); state['max_requests']=len(state['requests']); state[AUTH]['finished']=True
            budget.write(state); save(ROOT/'budget_snapshot.json',state)
        print('Continuation closed; original records unchanged.',flush=True)


def audit():
    import experiments.audit_brief_protocol_pair as checker
    checker.ROOT=ROOT; checker.AUTH=AUTH
    checker.main()
    read=lambda p:json.loads(p.read_text())
    cfg=read(ROOT/'config.json'); original=read(ORIGINAL/'config.json')
    for key in ('settings','tasks','base_skills','base_hashes','tasks_sha256','run_order'):
        assert cfg[key]==original[key],key
    old=read(ORIGINAL/'results.json'); new=read(ROOT/'results.json')
    merged={}; raw={}; substitutions=[]
    for arm in old:
        first={r['task_id']:r for r in old[arm]}
        combined=dict(first)
        for row in new[arm]:
            tid=row['task_id']
            if tid in combined:
                assert combined[tid]['status']=='api_error' and row['attempt_kind']=='infra_retry_once'
                substitutions.append({'arm':arm,'task_id':tid,'original_status':combined[tid]['status'],
                    'retry_status':row['status'],'original_retained_in':str(ORIGINAL)})
            else: first[tid]=row
            combined[tid]=row
        raw[arm]=gate(list(first.values()),cfg['tasks'])
        merged[arm]={'gate_after_explicit_infra_retry':gate(list(combined.values()),cfg['tasks']),
                    'rows':list(combined.values())}
    paid=read(ROOT/'audit.json'); before=read(ORIGINAL/'audit.json')
    save(ROOT/'combined.json',{'raw_first_attempt_gates':raw,'arms':merged,'explicit_retry_substitutions':substitutions,
        'all_attempts_paid_requests':paid['paid_requests']+before['paid_requests'],
        'all_attempts_guarded_cost_usd':paid['incremental_conservative_usd']+before['incremental_conservative_usd'],
        'cumulative_guarded_usd':paid['cumulative_guarded_usd'],
        'note':'Retry-adjusted capability shown separately from raw infrastructure-inclusive first attempts. No original record overwritten.'})
    print(json.dumps({a:r['gate_after_explicit_infra_retry'] for a,r in merged.items()},indent=2))

if __name__=='__main__': audit()
