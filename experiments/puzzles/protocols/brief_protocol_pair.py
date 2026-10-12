"""One frozen, interleaved paired protocol test, no adaptive prompt edits."""
from dataclasses import asdict
import fcntl
import hashlib
import json
from pathlib import Path
import random
import time
from dynamic_skill.campaign_budget import CampaignBudget, BudgetPlanner
from experiments.cold_start_gate import gate
from experiments.paper_cold_start_gate import paper_prompt
from experiments.puzzle_pilot import ApiPlanner, Settings, run_episode, save

NAME = 'brief_protocol_pair_20260928'
ROOT = Path('results') / NAME
AUTH = 'brief_protocol_pair_authorization'


def brief_prompt(env, history, remaining, limit, text):
    # Change only the output-format paragraph, never task knowledge or rules.
    body, sep, old_output = text.rpartition('\n## Output\n')
    if not sep or not old_output.strip():
        raise ValueError('expected terminal Output section in frozen skill')
    effective = body + '\n## Output\n' + (
        'Return <reason>one brief English sentence, at most 30 words, summarizing the immediate '
        'subgoal or action justification</reason><actions>["one concrete native command"]</actions>. '
        'Do not provide detailed step-by-step reasoning, think tags, markdown, a program or a solved board.\n')
    return paper_prompt(env, history, remaining, limit, effective)


def run():
    path = Path('results/gpt54_recovery_20260928/budget.json')
    tasks_path = Path('data') / NAME / 'tasks.json'
    tasks = json.loads(tasks_path.read_text())
    texts = {b: Path(f'skills/{b}/paper_v2.md').read_text() for b in ('rush_hour', 'sudoku')}
    if ROOT.exists(): raise RuntimeError('never overwrite or repeat a paid batch')
    with (path.parent / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        budget = CampaignBudget(path, deadline=0)
        state = budget.read()
        if (state.get(AUTH) or state['policy_violation'] or time.time() < state['cooldown_until']
                or any(r['status']=='pending' for r in state['requests'])):
            raise RuntimeError('authorization/pending/policy/cooldown guard closed')
        if state['limit_usd'] != 20 or budget.effective_spend(state) + state['reservation_usd'] > 20:
            raise RuntimeError('existing USD20 budget unavailable')
        state[AUTH] = {'reason': 'User authorized output-protocol-only continuation, native thinking remains disabled.',
                       'previous_requests':len(state['requests']), 'at_unix':time.time(),
                       'previous_charge':budget.effective_spend(state),
                       'old_deadline':state['deadline_unix'], 'old_max_requests':state['max_requests'],
                       'old_max_unknown':state['max_unknown_requests'], 'additional_request_cap':72}
        state['max_requests'] = len(state['requests']) + 72
        state['max_unknown_requests'] = sum(r['status']=='unknown_usage' for r in state['requests']) + 1
        state['deadline_unix'] = time.time() + 1800
        budget.write(state)
        ROOT.mkdir()
        settings = {arm: Settings(model='gpt-5.4', action_only=arm=='action_only',
            output_protocol='legacy' if arm=='action_only' else 'brief_reason_v1',
            max_steps=6, max_calls=6, max_tokens=128, request_timeout=45, episode_seconds=180,
            input_usd_per_million=state['input_usd_per_million'], output_usd_per_million=state['output_usd_per_million'])
            for arm in ('action_only','brief_reason')}
        jobs=[]
        for i,task in enumerate(tasks):
            arms=['action_only','brief_reason'] if i%2==0 else ['brief_reason','action_only']
            jobs.extend((task,arm) for arm in arms)
        random.Random(20260928).shuffle(jobs)
        save(ROOT/'config.json', {'tasks':tasks, 'tasks_sha256':hashlib.sha256(tasks_path.read_bytes()).hexdigest(),
            'settings':{k:asdict(v) for k,v in settings.items()}, 'base_skills':texts,
            'base_hashes':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in texts.items()},
            'run_order':[[t['id'],a] for t,a in jobs], 'gate':gate([],tasks),
            'note':'Only system/user output instructions differ. Same 128-token cap. One sample per fresh task/arm; no test-driven edits.'})
        rows={a:[] for a in settings}
        try:
            for task,arm in jobs:
                api=ApiPlanner(settings[arm], action_limit=1)
                planner=BudgetPlanner(api,budget,str(ROOT)+'/'+arm+'/'+task['id'])
                try:
                    row=run_episode(task,'react',settings[arm],ROOT/arm,planner,
                        base_skill_text=texts[task['benchmark']],
                        prompt_builder=paper_prompt if arm=='action_only' else brief_prompt)
                finally: api.close()
                rows[arm].append(row); save(ROOT/'results.json',rows)
                state=budget.read()
                if (row['status'] not in {'success','step_limit','model_call_limit','format_error'}
                    or state['policy_violation'] or state['consecutive_errors']): break
        finally:
            save(ROOT/'gate.json',{a:gate(r,tasks) for a,r in rows.items()})
            state=budget.read(); state['max_requests']=len(state['requests'])
            state[AUTH]['finished']=True
            budget.write(state); save(ROOT/'budget_snapshot.json',state)
        print(json.dumps({a:gate(r,tasks) for a,r in rows.items()},indent=2),flush=True)
