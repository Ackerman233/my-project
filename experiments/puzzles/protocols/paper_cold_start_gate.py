"""Frozen single-action capability gate; paper-inspired, not a paper replication."""
from dataclasses import asdict
import fcntl
import hashlib
import json
from pathlib import Path
import time

from dynamic_skill.campaign_budget import CampaignBudget, BudgetPlanner
from experiments.cold_start_gate import gate
from experiments.puzzle_pilot import ApiPlanner, Settings, run_episode, save

NAME = 'paper_cold_start_gate_20260928'
ROOT = Path('results') / NAME


def paper_prompt(env, history, remaining, limit, text):
    if limit != 1:
        raise ValueError('this protocol is single-action only')
    goal = ('Fill the 9x9 grid consistently with every given clue.' if 'Sudoku' in env.rules
            else 'Move A to columns 5 and 6 of row 3, inside the right edge.')
    return (text + '\n\n## Goal\n' + goal + '\n\n## Rules\n' + env.rules
            + '\n\n## Recent History (executed actions and environment feedback)\n' + json.dumps(history)
            + '\n\n## Current Observation\n' + env.render()
            + f'\n\n## Remaining Steps\n{remaining}\nReturn exactly ONE action in the native actions format.')


def authorize(state, now):
    if state.get('paper_cold_start_authorization'):
        raise RuntimeError('this one-shot authorization was already used')
    if any(r['status'] == 'pending' for r in state['requests']):
        raise RuntimeError('unsettled pending request')
    if state['policy_violation'] or now < state['cooldown_until']:
        raise RuntimeError('policy/cooldown guard closed')
    spent = CampaignBudget.effective_spend(state)
    if spent + state['reservation_usd'] > 20:
        raise RuntimeError('insufficient cumulative budget')
    state['paper_cold_start_authorization'] = {
        'reason': 'User raised cumulative budget to USD20 and requested Fig14/15-inspired skills and a gate retry.',
        'at_unix': now, 'previous_requests': len(state['requests']), 'previous_charge': spent,
        'previous_limit_usd': state['limit_usd'], 'old_deadline': state['deadline_unix'],
        'old_max_requests': state['max_requests'], 'old_max_unknown': state['max_unknown_requests'],
        'additional_request_cap': 36, 'new_unknown_tolerance': 1,
        'note': 'Keep existing billing checkpoint and all historical requests unchanged.'}
    state['limit_usd'] = 20.0
    state['max_requests'] = len(state['requests']) + 36
    state['max_unknown_requests'] = sum(r['status'] == 'unknown_usage' for r in state['requests']) + 1
    state['deadline_unix'] = now + 1200


def run():
    path = Path('results/gpt54_recovery_20260928/budget.json')
    tasks = json.loads((Path('data') / NAME / 'tasks.json').read_text())
    texts = {b: Path(f'skills/{b}/paper_v2.md').read_text() for b in ('rush_hour', 'sudoku')}
    if ROOT.exists():
        raise RuntimeError('never overwrite or retry this frozen gate')
    with (path.parent / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        budget = CampaignBudget(path, deadline=0)
        state = budget.read()
        authorize(state, time.time())
        budget.write(state)
        ROOT.mkdir()
        settings = Settings(model='gpt-5.4', action_only=True, max_steps=6, max_calls=6,
                            max_tokens=128, request_timeout=45, episode_seconds=180,
                            input_usd_per_million=state['input_usd_per_million'],
                            output_usd_per_million=state['output_usd_per_million'])
        save(ROOT / 'config.json', {
            'settings': asdict(settings), 'prompt_protocol': 'paper_v2', 'tasks': tasks,
            'tasks_sha256': hashlib.sha256((Path('data') / NAME / 'tasks.json').read_bytes()).hexdigest(),
            'base_skills': texts,
            'base_hashes': {k: hashlib.sha256(v.encode()).hexdigest() for k, v in texts.items()},
            'gate': gate([], tasks), 'budget_ledger': str(path),
            'note': 'New frozen synthetic confirmation gate. No matched no-skill arm; not causal, not long-horizon.',
            'paper_adaptations': 'Fig14/15 structure; native action syntax; no thought/reason sections; reasoning_effort=none.'})
        rows = []
        try:
            for task in tasks:
                api = ApiPlanner(settings, action_limit=1)
                planner = BudgetPlanner(api, budget, str(ROOT) + '/' + task['id'])
                try:
                    row = run_episode(task, 'react', settings, ROOT, planner,
                                      base_skill_text=texts[task['benchmark']], prompt_builder=paper_prompt)
                finally:
                    api.close()
                rows.append(row)
                save(ROOT / 'results.json', rows)
                state = budget.read()
                if (row['status'] not in {'success', 'step_limit', 'model_call_limit', 'format_error'}
                        or state['policy_violation'] or state['consecutive_errors']):
                    break
        finally:
            save(ROOT / 'gate.json', gate(rows, tasks))
            state = budget.read()
            state['max_requests'] = len(state['requests'])
            state['paper_cold_start_authorization']['finished'] = True
            state['paper_cold_start_authorization']['attempted_episodes'] = len(rows)
            budget.write(state)
            save(ROOT / 'budget_snapshot.json', state)
        print(json.dumps(gate(rows, tasks), indent=2), flush=True)
