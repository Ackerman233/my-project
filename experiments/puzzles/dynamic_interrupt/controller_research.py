"""Bounded paid episodes and mechanical controller search; no hidden answers."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sys
import time
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))
from dynamic_skill.campaign_budget import CampaignBudget, BudgetPlanner, CampaignHalt
from dynamic_skill.puzzles import make_environment
from dynamic_skill.research_controller import Controller, POLICIES
from dynamic_skill.alfworld_rollout import parse_action_chunk
from experiments.puzzle_pilot import ApiPlanner, Settings, save

ROOT = Path('results/controller_research_20260928')
LEDGER = Path('results/gpt54_recovery_20260928/budget.json')
AUTH = 'controller_research_authorization_20260928'
DEADLINE = datetime(2026,9,28,20,40,37,tzinfo=timezone.utc).timestamp()
SYSTEM = ('Control the puzzle using a short decision summary and concrete commands. '
          'Output exactly <reason>one English sentence of at most 30 words</reason>'
          '<actions>["command"]</actions>. The array may contain 1 up to the requested action limit. '
          'For a chunk, each command applies to the state AFTER earlier commands. '
          'No think tags, detailed chain of thought, markdown, programs or solved boards.')


def settings():
    return Settings(model='gpt-5.4', max_tokens=256, max_calls=8, max_steps=18,
                    request_timeout=45, episode_seconds=240, temperature=0,
                    action_only=False, input_usd_per_million=25, output_usd_per_million=150)


def sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def prepare():
    from experiments.prepare_small_evolution_run import rush, sudoku
    if (ROOT/'config.json').exists(): raise RuntimeError('frozen config already exists')
    splits = {}; seen = set()
    for n, split in enumerate(('dev','validation','test')):
        tasks=[]
        for i,(count,depth) in enumerate(((7,3),(9,5))):
            board, _, _ = rush(202609281600+n*100+i,count,depth)
            tasks.append({'id':f'cr-{split}-rush-{i}', 'benchmark':'rush_hour','board':board})
        for i,holes in enumerate((4,6)):
            board, _ = sudoku(202609281650+n*100+i,holes)
            tasks.append({'id':f'cr-{split}-sudoku-{i}', 'benchmark':'sudoku','puzzle':board})
        for t in tasks:
            key=sha({k:v for k,v in t.items() if k!='id'})
            assert key not in seen
            seen.add(key)
        splits[split]=tasks
    skills={}
    for domain in ('rush_hour','sudoku'):
        text=Path(f'skills/{domain}/paper_v2.md').read_text().split('## Output')[0]
        text=text.replace('Choose one concrete slide, then use the new observation to choose the next.',
                          'Choose concrete slides in their execution order within the requested action limit.')
        text=text.replace('Fill one deduced cell, then inspect the updated observation.',
                          'Fill deduced cells in their execution order within the requested action limit.')
        text=text.replace('Recompute from the latest observation after every action.',
                          'Account for earlier actions when forming a multi-action plan.')
        skills[domain]=text
    cfg={'protocol':'controller-brief-v1','settings':asdict(settings()),'system':SYSTEM,
         'skills':skills,'splits':splits,'policies':{k:asdict(v) for k,v in POLICIES.items()},
         'score':'success_rate - 0.01*mean_calls - 0.002*mean_invalid',
         'deadline_unix':DEADLINE,'max_new_requests':384}
    cfg['hash']=sha(cfg)
    save(ROOT/'config.json',cfg)
    print(json.dumps({'prepared':True,'counts':{k:len(v) for k,v in splits.items()},'hash':cfg['hash']}))


def authorize():
    cfg=json.loads((ROOT/'config.json').read_text())
    with (LEDGER.parent/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        b=CampaignBudget(LEDGER,deadline=0); d=b.read()
        if AUTH in d: raise RuntimeError('authorization already recorded')
        if any(r['status']=='pending' for r in d['requests']): raise RuntimeError('pending request')
        if d['policy_violation'] or time.time()<d['cooldown_until']: raise RuntimeError('closed guard')
        if time.time()>=DEADLINE: raise RuntimeError('deadline expired')
        unknown=sum(r['status']=='unknown_usage' for r in d['requests'])
        d[AUTH]={'at_unix':time.time(),'previous_requests':len(d['requests']),
                 'previous_effective_usd':b.effective_spend(d),'old_limit_usd':d['limit_usd'],
                 'old_deadline':d['deadline_unix'],'old_max_requests':d['max_requests'],
                 'old_max_unknown':d['max_unknown_requests'],'initial_unknown':unknown,
                 'config_sha':cfg['hash'],'reason':'Explicit user: cumulative $30, at most five hours; unattended mechanical validation.'}
        d['limit_usd']=30.0; d['deadline_unix']=DEADLINE
        d['max_requests']=len(d['requests'])+cfg['max_new_requests']
        d['max_unknown_requests']=unknown+1
        b.write(d)
        print(json.dumps({'authorized':True,'effective_usd':b.effective_spend(d),'limit_usd':30,'deadline':DEADLINE}))


def check_config(cfg):
    raw={k:v for k,v in cfg.items() if k!='hash'}
    assert sha(raw)==cfg['hash']
    assert cfg['settings']==asdict(settings()) and cfg['system']==SYSTEM
    assert cfg['policies']=={k:asdict(v) for k,v in POLICIES.items()}


def run(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--split',choices=['dev','validation','test'],required=True)
    p.add_argument('--policy',choices=list(POLICIES),required=True); p.add_argument('--index',type=int,required=True)
    a=p.parse_args(argv)
    cfg=json.loads((ROOT/'config.json').read_text()); check_config(cfg)
    task=cfg['splits'][a.split][a.index]
    if cfg.get('evaluation_only'):
        assert a.split=='test' and a.policy in cfg['fixed_policies']
    elif a.split!='dev':
        selection=json.loads((ROOT/'selection.json').read_text())
        assert selection['config_sha']==cfg['hash']
        allowed=['interrupt',selection['candidate']] if a.split=='validation' else ['react','fixed','interrupt',selection['candidate']]
        assert a.policy in allowed
        if a.split=='test': assert (ROOT/'validation_evaluation.json').exists()
    out=ROOT/a.split/a.policy/task['id']
    with (LEDGER.parent/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        b=CampaignBudget(LEDGER,deadline=0); d=b.read(); auth=d[AUTH]
        assert auth['config_sha']==cfg['hash'] and d['limit_usd']==30
        assert d['deadline_unix']<=DEADLINE
        if any(r['status']=='pending' for r in d['requests']): raise RuntimeError('pending calls; audit before resuming')
        if sum(r['status']=='unknown_usage' for r in d['requests'])>auth['initial_unknown']:
            raise RuntimeError('new unknown usage; no autonomous retry')
        if out.exists():
            print(json.dumps({'status':'already_attempted','path':str(out)})); return
        out.mkdir(parents=True)
        save(out/'started.json',{'time':time.time(),'task_id':task['id'],'policy':a.policy,'config_sha':cfg['hash']})
        env=make_environment(task); controller=Controller(POLICIES[a.policy]); s=settings()
        api=ApiPlanner(s,system_prompt=SYSTEM); planner=BudgetPlanner(api,b,str(out))
        calls=[]; trajectory=[]; tail=[]; started=time.monotonic(); status='step_limit'
        spend_before=b.effective_spend(b.read()); request_before=len(b.read()['requests'])
        try:
            while len(trajectory)<s.max_steps:
                if time.monotonic()-started>190: status='episode_time_limit'; break
                if not tail:
                    if len(calls)>=s.max_calls: status='model_call_limit'; break
                    limit=min(controller.action_limit(),s.max_steps-len(trajectory))
                    state=env.render()
                    history=[{k:r[k] for k in ('action','valid','feedback')} for r in trajectory[-8:]]
                    prompt=(env.rules+'\n\nBasic task skill:\n'+cfg['skills'][task['benchmark']]
                            +'\nCurrent board:\n'+state+'\nRecent executed actions and feedback:\n'+json.dumps(history)
                            +f'\nPlan 1 to {limit} concrete actions. They execute sequentially without new model calls.'
                            +' Use the latest board and account for every earlier planned action. Return the specified reason/actions tags.')
                    memory=controller.memory(state)
                    if memory: prompt+='\nPreviously rejected commands in this EXACT board state; avoid repeating: '+json.dumps(memory)
                    try: planner.before_call()
                    except CampaignHalt as e: status=e.status; break
                    wire=({'reasoning_effort':'none','max_completion_tokens':s.max_tokens}
                          if s.model=='gpt-5.4' else {'enable_thinking':False,'max_tokens':s.max_tokens})
                    rec={'call_id':len(calls)+1,'prompt':prompt,'system_prompt':SYSTEM,'limit':limit,
                         'requested_model':s.model,**wire,'status':'pending'}
                    calls.append(rec); save(out/'calls.json',calls); tick=time.monotonic()
                    try:
                        rec.update(planner(prompt)); rec['status']='ok'
                    except Exception as e:
                        rec.update(status='api_error',error_type=type(e).__name__)
                    rec['latency_seconds']=time.monotonic()-tick
                    save(out/'calls.json',calls)
                    d=b.read()
                    if rec['status']!='ok': status='api_error'; break
                    if d['policy_violation']: status='thinking_policy_stop'; break
                    if d['requests'][-1]['status']!='returned_usage': status='unknown_cost_stop'; break
                    try:
                        response=rec['response']
                        match=re.fullmatch(r'\s*<reason>([^<]*)</reason>\s*<actions>(.*?)</actions>\s*',response,re.S)
                        if not match or len(match.group(1).split())>30: raise ValueError('brief protocol violation')
                        tail=parse_action_chunk(response,limit); rec['actions']=tail[:]
                    except ValueError as e:
                        rec['status']='format_error'; status='format_error'; save(out/'calls.json',calls); break
                    save(out/'calls.json',calls)
                before=env.render(); action=tail.pop(0); result=env.step(action)
                event=controller.observe(before,action,result.valid,result.observation,tail)
                row={'step':len(trajectory)+1,'action':action,'valid':result.valid,'feedback':result.reason,
                     'board_before':before,'board_after':result.observation,'won':bool(result.won),
                     'call_id':len(calls),'tail_before_decision':tail[:],**event}
                if event['interrupt']: tail=[]
                trajectory.append(row); save(out/'trajectory.json',trajectory)
                if result.won: status='success'; break
        finally: api.close()
        d=b.read(); row={'task_id':task['id'],'policy':a.policy,'split':a.split,'status':status,
             'success':status=='success','model_calls':len(calls),'environment_steps':len(trajectory),
             'invalid_actions':sum(not r['valid'] for r in trajectory),
             'interrupts':sum(r['interrupt'] for r in trajectory),
             'discarded_actions':sum(len(r['discarded']) for r in trajectory),
             'cost_usd':b.effective_spend(d)-spend_before,'latency_seconds':sum(r['latency_seconds'] for r in calls),
             'paid_requests':len(d['requests'])-request_before,'config_sha':cfg['hash']}
        save(out/'result.json',row); print(json.dumps(row))


def evaluate():
    cfg=json.loads((ROOT/'config.json').read_text()); check_config(cfg)
    report={}
    for split,tasks in cfg['splits'].items():
        report[split]={}
        for name in POLICIES:
            rows=[json.loads(p.read_text()) for p in (ROOT/split/name).glob('*/result.json')]
            if not rows: continue
            n=len(rows); sr=sum(r['success'] for r in rows)/n
            calls=sum(r['model_calls'] for r in rows)/n; invalid=sum(r['invalid_actions'] for r in rows)/n
            complete=n==len(tasks)
            infra=any(r['status'] in ('api_error','unknown_cost_stop','thinking_policy_stop','budget_stop','deadline_stop','episode_time_limit') for r in rows)
            report[split][name]={'pass':complete and not infra,'score':sr-.01*calls-.002*invalid,
                                'n':n,'successes':sum(r['success'] for r in rows),'mean_calls':calls,
                                'mean_invalid':invalid,'cost_usd':sum(r['cost_usd'] for r in rows),'rows':rows}
    save(ROOT/'evaluation.json',report)
    print(json.dumps({'pass':any(v['pass'] for v in report['dev'].values()),
                      'score':max((v['score'] for v in report['dev'].values() if v['pass']),default=0),'groups':report}))


def advance():
    """One idempotent episode per invocation. Search uses development data only."""
    if time.time()>=DEADLINE:
        print(json.dumps({'status':'deadline_stop'})); return
    cfg=json.loads((ROOT/'config.json').read_text()); check_config(cfg)
    # Policy order is fixed; each policy rotates task order to reduce ordering confounds.
    order=['react','fixed','interrupt','adaptive1','adaptive3','memory',
           'memory_adaptive1','memory_adaptive3','patient_memory']
    evaluate()
    report=json.loads((ROOT/'evaluation.json').read_text())
    for j,name in enumerate(order):
        for index in [(i+j)%4 for i in range(4)]:
            out=ROOT/'dev'/name/cfg['splits']['dev'][index]['id']
            if not out.exists():
                run(['--split','dev','--policy',name,'--index',str(index)]); return
            if not (out/'result.json').exists(): raise RuntimeError('incomplete episode requires audit')
    candidates=[(v['score'],name) for name,v in report['dev'].items()
                if name not in ('react','fixed','interrupt') and v['pass']]
    if not candidates: raise RuntimeError('no mechanically valid candidate')
    best=max(candidates)[1]
    chosen=ROOT/'selection.json'
    selection={'candidate':best,'config_sha':cfg['hash'],'source':'development_only',
               'dev_score':report['dev'][best]['score'],'frozen_at_unix':time.time()}
    if not chosen.exists(): save(chosen,selection)
    else: assert json.loads(chosen.read_text())['candidate']==best
    for name in ['interrupt',best]:
        for i,t in enumerate(cfg['splits']['validation']):
            out=ROOT/'validation'/name/t['id']
            if not out.exists(): run(['--split','validation','--policy',name,'--index',str(i)]); return
            if not (out/'result.json').exists(): raise RuntimeError('incomplete validation requires audit')
    base=report['validation']['interrupt']; cand=report['validation'][best]
    adopted=(base['pass'] and cand['pass'] and cand['successes']>=base['successes']
             and cand['score']>base['score']+.005)
    validation={'pass':adopted,'score':cand['score'],'baseline':base,'candidate':cand,
                'selected_policy':best,'note':'Screening gate only; test may falsify the gain. No statistical significance claim.'}
    if not (ROOT/'validation_evaluation.json').exists(): save(ROOT/'validation_evaluation.json',validation)
    for name in ['react','fixed','interrupt',best]:
        for i,t in enumerate(cfg['splits']['test']):
            out=ROOT/'test'/name/t['id']
            if not out.exists(): run(['--split','test','--policy',name,'--index',str(i)]); return
            if not (out/'result.json').exists(): raise RuntimeError('incomplete test requires audit')
    save(ROOT/'complete.json',{'status':'complete','validation_pass':adopted,'candidate':best,
                               'finished_unix':time.time(),'report':'evaluation.json'})
    print(json.dumps({'status':'complete','validation_pass':adopted,'candidate':best}))


if __name__=='__main__':
    {'prepare':prepare,'authorize':authorize,'evaluate':evaluate}[sys.argv[1]]()
