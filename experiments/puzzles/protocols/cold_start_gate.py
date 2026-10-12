"""One authorized cold-start gate: no test-driven skill edits or automatic retry."""
from dataclasses import asdict
import fcntl
import hashlib
import json
from pathlib import Path
import time
from dynamic_skill.campaign_budget import CampaignBudget, BudgetPlanner, CampaignHalt
from dynamic_skill.puzzles import make_environment
from experiments.puzzle_pilot import ApiPlanner, Settings, run_episode, save

ROOT = Path('results/cold_start_gate_20260928')


def gate(rows, tasks):
    complete = len(rows) == len(tasks) and {r['task_id'] for r in rows} == {t['id'] for t in tasks}
    by_domain = {b: {'completed': sum(r['success'] for r in rows if r['benchmark']==b),
                     'scheduled': sum(t['benchmark']==b for t in tasks)} for b in ('rush_hour','sudoku')}
    invalid = sum(r['invalid_actions'] for r in rows)
    actions = sum(r['environment_steps'] for r in rows)
    success = sum(r['success'] for r in rows)
    clean = not any(r['model_errors'] or r['format_errors'] or r['usage_incomplete'] for r in rows)
    passed = complete and clean and success >= 5 and all(v['completed'] >= 2 for v in by_domain.values()) and invalid/max(1,actions) <= .1
    return {'passed': passed, 'complete': complete, 'successes': success, 'scheduled': len(tasks),
            'by_domain': by_domain, 'invalid_actions': invalid, 'environment_actions': actions,
            'threshold': '>=5/6 overall, >=2/3 per domain, invalid fraction <=10%, zero API/format/unknown-usage errors'}


def run():
    path = Path('results/gpt54_recovery_20260928/budget.json')
    if ROOT.exists():
        raise RuntimeError('never overwrite/retry the same gate')
    tasks = json.loads(Path('data/cold_start_gate_20260928/tasks.json').read_text())
    texts = {b: Path(f'skills/{b}/initial.md').read_text() for b in ('rush_hour','sudoku')}
    with (path.parent/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        budget = CampaignBudget(path, deadline=0)
        state = budget.read()
        if state.get('cold_start_authorization') or any(r['status']=='pending' for r in state['requests']):
            raise RuntimeError('already authorized or pending requests')
        if state['policy_violation'] or time.time() < state['cooldown_until']:
            raise RuntimeError('policy/cooldown guard closed')
        charged = budget.effective_spend(state)
        if charged >= state['limit_usd'] or state['limit_usd'] > 5:
            raise RuntimeError('insufficient existing authorized budget')
        unknown = sum(r['status']=='unknown_usage' for r in state['requests'])
        state['cold_start_authorization'] = {'reason':'User explicitly requested cold-start skills and a simple single-action gate.',
            'at_unix':time.time(), 'old_deadline':state['deadline_unix'], 'old_max_requests':state['max_requests'],
            'old_max_unknown':state['max_unknown_requests'], 'previous_requests':len(state['requests']),
            'previous_charge':charged, 'additional_request_cap':36, 'new_unknown_tolerance':1}
        state['max_requests'] = len(state['requests']) + 36
        state['max_unknown_requests'] = unknown + 1
        state['deadline_unix'] = time.time() + 1200
        budget.write(state)
        ROOT.mkdir()
        settings = Settings(model='gpt-5.4', action_only=True, max_steps=6, max_calls=6, max_tokens=128,
            request_timeout=45, episode_seconds=120, input_usd_per_million=state['input_usd_per_million'],
            output_usd_per_million=state['output_usd_per_million'])
        save(ROOT/'config.json', {'settings':asdict(settings), 'tasks':tasks, 'base_skills':texts,
            'base_hashes':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in texts.items()},
            'gate':gate([],tasks), 'note':'New synthetic smoke gate, no no-skill matched arm; not a causal effect estimate.'})
        rows=[]
        try:
            for task in tasks:
                api=ApiPlanner(settings, action_limit=1)
                planner=BudgetPlanner(api,budget,str(ROOT)+'/'+task['id'])
                class Guard:
                    def before_call(self):
                        self.ready=True
                    def __call__(self,prompt):
                        if not self.ready: raise RuntimeError('not prepared')
                        self.ready=False
                        d=budget.read()
                        upper=((len(prompt.encode())+2048)*d['input_usd_per_million']+128*d['output_usd_per_million'])/1e6
                        if upper > d['reservation_usd']: raise CampaignHalt('request_reservation_too_small')
                        # UTF-8 bytes bound text-token count, plus 2048 tokens for
                        # system/framing. Completion cap includes all output tokens.
                        # Preserve the full bound for unknown usage; no retries.
                        planner.before_call(reservation_usd=upper)
                        result=planner(prompt)
                        d=budget.read()
                        if d['policy_violation']: raise CampaignHalt('thinking_policy_stop')
                        if d['requests'][-1]['status']!='returned_usage': raise CampaignHalt('unknown_cost_stop')
                        return result
                try:
                    row=run_episode(task,'react',settings,ROOT,Guard(),base_skill_text=texts[task['benchmark']])
                finally:
                    api.close()
                rows.append(row); save(ROOT/'results.json',rows)
                if row['status'] not in {'success','step_limit','model_call_limit','format_error'}:
                    break
        finally:
            save(ROOT/'gate.json',gate(rows,tasks))
            state=budget.read()
            # Close ONLY this request grant while holding the shared lock.
            state['max_requests']=len(state['requests'])
            state['cold_start_authorization']['attempted_episodes']=len(rows)
            state['cold_start_authorization']['finished']=True
            budget.write(state); save(ROOT/'budget_snapshot.json',state)
        print(json.dumps(gate(rows,tasks),indent=2))
