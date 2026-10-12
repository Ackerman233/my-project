"""User-authorized 12-episode replication of Qwen rung0; same GPT ledger."""
from dataclasses import asdict
import fcntl
import hashlib
import json
from pathlib import Path
import time
from dynamic_skill.campaign_budget import CampaignBudget, BudgetPlanner
from experiments.puzzle_pilot import ApiPlanner, Settings, run_episode, save, summarize

CAMPAIGN = Path('results/gpt54_recovery_20260928')
OUT = Path('results/gpt54_rollout_20260928')
BASE = Path('results/campaign_rung0_20260927T151906Z')


def authorize(budget):
    data = budget.read()
    if data.get('rollout_authorization') or len(data['requests']) != 7:
        raise RuntimeError('Expected seven prior GPT calls and unused rollout authorization')
    if any(r['status'] == 'pending' for r in data['requests']):
        raise RuntimeError('Unsettled prior request')
    now = time.time()
    data['rollout_authorization'] = dict(at_unix=now,
        reason='User explicitly requested GPT-5.4 full three-mode rollout replication; same $5 total.',
        old_deadline=data['deadline_unix'], old_max_requests=data['max_requests'],
        additional_request_cap=288, duration_seconds=3600)
    data['deadline_unix'] = now + 3600
    data['max_requests'] = 7 + 288
    budget.write(data)


def run():
    with (CAMPAIGN/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert (CAMPAIGN/'budget.json').exists(), 'Never initialize a replacement budget'
        budget = CampaignBudget(CAMPAIGN/'budget.json', deadline=0)
        config = json.loads((BASE/'config.json').read_text())
        task_path = Path('data/curriculum_20260927/tasks.json')
        assert hashlib.sha256(task_path.read_bytes()).hexdigest() == config['tasks_sha256']
        tasks = {t['id']: t for t in json.loads(task_path.read_text())}
        jobs = [(tasks[tid], mode) for tid, mode in config['run_order']]
        assert len(jobs) == 12 and len({(t['id'], m) for t,m in jobs}) == 12
        OUT.mkdir(exist_ok=False)
        authorize(budget)
        settings = Settings(model='gpt-5.4', chunk_size=4, max_steps=24, max_calls=24,
            max_tokens=512, request_timeout=45, episode_seconds=300, action_only=True,
            prompt_variant='baseline', controller_variant='strict')
        save(OUT/'config.json', {**asdict(settings), 'reasoning_effort':'none',
            'workers':1, 'provider_retries':0, 'run_order':config['run_order'],
            'tasks_sha256':config['tasks_sha256'], 'budget_ledger':str(budget.path),
            'protocol':'gpt-rung0-replication-v1', 'reference_run':str(BASE)})
        rows = []
        for task, mode in jobs:
            api = ApiPlanner(settings, action_limit=1 if mode=='react' else 4)
            planner = BudgetPlanner(api, budget, str(OUT)+'/'+task['id']+'__'+mode)
            try:
                result = run_episode(task, mode, settings, OUT, planner)
            finally:
                api.close()
            rows.append(result)
            with (OUT/'results.jsonl').open('a') as handle:
                handle.write(json.dumps(result)+'\n')
            save(OUT/'metrics.json', summarize(rows))
            state = budget.read()
            if (result['status'] not in {'success','format_error','step_limit','model_call_limit',
                                        'wall_time_limit','interrupt_budget_exhausted'}
                or state['policy_violation'] or state['requests'][-1]['status'] != 'returned_usage'):
                break
        save(OUT/'run_status.json', {'status':'finished' if len(rows)==12 else 'stopped_by_guard',
            'episodes':len(rows), 'scheduled_episodes':12})
        print(json.dumps(summarize(rows)), flush=True)
