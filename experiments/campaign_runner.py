"""One bounded serial batch; adaptive decisions remain with the supervising task."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import random
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dynamic_skill.campaign_budget import CampaignBudget, BudgetPlanner
from experiments.puzzle_pilot import ApiPlanner, Settings, run_episode, save, summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, default=Path('results/autonomous_campaign_20260927'))
    parser.add_argument('--tasks', type=Path, default=Path('data/curriculum_20260927/tasks.json'))
    parser.add_argument('--rung', type=int, required=True)
    parser.add_argument('--benchmark', choices=['rush_hour', 'sudoku'])
    parser.add_argument('--prompt-variant', choices=['baseline', 'neutral_rush_v1'], default='baseline')
    parser.add_argument('--controller-variant', choices=['strict', 'noop_tail_v1'], default='strict')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--chunk-size', type=int, default=4)
    parser.add_argument('--max-steps', type=int, default=24)
    parser.add_argument('--max-calls', type=int, default=24)
    parser.add_argument('--modes', nargs='+', choices=['react', 'fixed_chunk', 'dynamic_interrupt'],
                        default=['react', 'fixed_chunk', 'dynamic_interrupt'])
    args = parser.parse_args()
    if min(args.chunk_size, args.max_steps, args.max_calls) < 1:
        parser.error('budgets and chunk size must be positive')
    args.campaign.mkdir(parents=True, exist_ok=True)
    with (args.campaign / 'run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another campaign batch owns the lock; do not overlap.')
        deadline = datetime(2026, 9, 27, 19, 1, tzinfo=timezone.utc).timestamp()
        budget = CampaignBudget(args.campaign / 'budget.json', deadline=deadline)
        tasks = [t for t in json.loads(args.tasks.read_text()) if t['rung'] == args.rung]
        if args.benchmark:
            tasks = [t for t in tasks if t['benchmark'] == args.benchmark]
        if not tasks or len(set(args.modes)) != len(args.modes):
            raise SystemExit('invalid rung or duplicate modes')
        args.out.mkdir(parents=True, exist_ok=False)
        settings = Settings(chunk_size=args.chunk_size, max_steps=args.max_steps, max_calls=args.max_calls,
                            max_tokens=512, request_timeout=45, episode_seconds=300, action_only=True,
                            prompt_variant=args.prompt_variant,
                            controller_variant=args.controller_variant,
                            input_usd_per_million=0.0564, output_usd_per_million=0.4512)
        jobs = [(task, mode) for task in tasks for mode in args.modes]
        random.Random(20260927 + args.rung).shuffle(jobs)
        save(args.out / 'config.json', {**asdict(settings), 'enable_thinking': False, 'workers': 1,
             'tasks_sha256': hashlib.sha256(args.tasks.read_bytes()).hexdigest(), 'rung': args.rung,
             'run_order': [[t['id'], m] for t, m in jobs], 'budget_ledger': str(args.campaign / 'budget.json'),
             'note': 'Old price snapshot for result estimates; conservative ledger uses higher rates. No automatic retries.'})
        rows = []
        for task, mode in jobs:
            api = ApiPlanner(settings, action_limit=1 if mode == 'react' else args.chunk_size)
            planner = BudgetPlanner(api, budget, str(args.out) + '/' + task['id'] + '__' + mode)
            try:
                result = run_episode(task, mode, settings, args.out, planner)
            finally:
                api.close()
            rows.append(result)
            with (args.out / 'results.jsonl').open('a') as handle:
                handle.write(json.dumps(result) + '\n')
            save(args.out / 'metrics.json', summarize(rows))
            state = budget.read()
            if result['status'] in {'budget_stop', 'deadline_stop', 'thinking_policy_stop', 'unknown_cost_stop',
                                    'api_cooldown', 'request_cap_stop'} or state['policy_violation']:
                break
            if state['consecutive_errors'] >= 2:
                break
        save(args.out / 'run_status.json', {'status': 'finished' if len(rows) == len(jobs) else 'stopped_by_guard',
            'episodes': len(rows), 'scheduled_episodes': len(jobs), 'last_status': rows[-1]['status'] if rows else None})
        print(json.dumps(summarize(rows), indent=2))


if __name__ == '__main__':
    if sys.argv[1:] == ['--gpt-recovery-diagnostic']:
        from experiments.gpt_recovery_diagnostic import run
        run()
    else:
        main()
