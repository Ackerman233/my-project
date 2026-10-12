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
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))
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
    if sys.argv[1:] == ['--prefix-recovery-next']:
        from experiments.prefix_recovery_trial import advance
        advance()
    elif sys.argv[1:] == ['--subskill-fixed-next']:
        from experiments.subskill_fixed_trial import advance
        advance()
    elif sys.argv[1:] == ['--subskill-exposure-next']:
        from experiments.subskill_exposure_trial import advance
        advance()
    elif sys.argv[1:] == ['--protocol-diagnostic-v5-next']:
        from experiments.protocol_diagnostic_v5_trial import advance
        advance()
    elif sys.argv[1:] == ['--protocol-diagnostic-v4-next']:
        from experiments.protocol_diagnostic_v4_trial import advance
        advance()
    elif sys.argv[1:] == ['--protocol-diagnostic-next']:
        from experiments.protocol_diagnostic_trial import advance
        advance()
    elif sys.argv[1:] == ['--space-subskills-v2-next']:
        from experiments.space_subskill_v2_trial import advance
        advance()
    elif sys.argv[1:] == ['--space-subskills-next']:
        from experiments.space_subskill_trial import advance
        advance()
    elif sys.argv[1:] == ['--space-boundary-next']:
        from experiments.space_boundary_trial import advance
        advance()
    elif sys.argv[1:] == ['--interrupt-recovery-v3-next']:
        from experiments.interrupt_recovery_ablation import advance
        advance()
    elif sys.argv[1:] == ['--control-parameter-next']:
        from experiments.control_parameter_trial import advance
        advance()
    elif sys.argv[1:] == ['--micro-evolution-next']:
        from experiments.micro_evolution_loop import advance
        advance()
    elif sys.argv[1:] == ['--alfworld-staged-next']:
        from experiments.alfworld_staged_recovery import advance
        advance()
    elif sys.argv[1:] == ['--alfworld-final-remaining']:
        from experiments.alfworld_final_remaining import run
        run()
    elif sys.argv[1:] == ['--qwen-alfworld-next']:
        from experiments.qwen_alfworld_pair import advance
        advance()
    elif sys.argv[1:] == ['--qwen-pair-next']:
        from experiments.qwen_model_pair import advance
        advance()
    elif sys.argv[1:] == ['--controller-tw-next']:
        from experiments.proxy_controller_research import run
        run()
    elif sys.argv[1:] == ['--controller-resume-next']:
        from experiments.resume_controller_research import run
        run()
    elif sys.argv[1:2] == ['--generation-diagnostic']:
        from experiments.generation_diagnostic import run
        run(sys.argv[2:])
    elif sys.argv[1:] == ['--controller-research-next']:
        from experiments.controller_research import advance
        advance()
    elif sys.argv[1:2] == ['--controller-research']:
        from experiments.controller_research import run
        run(sys.argv[2:])
    elif sys.argv[1:] == ['--brief-protocol-recovered-once']:
        from experiments.recovered_brief_protocol_pair import run
        run()
    elif sys.argv[1:] == ['--brief-protocol-resume-once']:
        from experiments.resume_brief_protocol_pair import run
        run()
    elif sys.argv[1:] == ['--brief-protocol-pair']:
        from experiments.brief_protocol_pair import run
        run()
    elif sys.argv[1:] == ['--paper-cold-start-gate']:
        from experiments.paper_cold_start_gate import run
        run()
    elif sys.argv[1:] == ['--alfworld-infra-retries']:
        from experiments.alfworld_campaign import run_infra_retries
        run_infra_retries()
    elif sys.argv[1:] == ['--alfworld-twenty-resume']:
        from experiments.alfworld_campaign import run
        run(twenty_resume=True)
    elif sys.argv[1:] == ['--alfworld-billing-resume']:
        from experiments.alfworld_campaign import run
        run(billing_resume=True)
    elif sys.argv[1:] == ['--cold-start-gate']:
        from experiments.cold_start_gate import run
        run()
    elif sys.argv[1:] == ['--alfworld-actiononly-resume']:
        from experiments.alfworld_campaign import run
        run(resume=True)
    elif sys.argv[1:] == ['--alfworld-actiononly']:
        from experiments.alfworld_campaign import run
        run()
    elif sys.argv[1:] == ['--gpt-rollout-batch']:
        from experiments.gpt_rollout_batch import run
        run()
    elif sys.argv[1:2] == ['--skill-evolution']:
        from experiments.skill_evolution import main as evolution_main
        evolution_main(sys.argv[2:])
    elif sys.argv[1:] == ['--gpt-recovery-remaining']:
        from experiments.gpt_recovery_diagnostic import run
        run(remaining=True)
    elif sys.argv[1:] == ['--gpt-recovery-retry-once']:
        from experiments.gpt_recovery_diagnostic import run
        run(retry=True)
    elif sys.argv[1:] == ['--gpt-recovery-diagnostic']:
        from experiments.gpt_recovery_diagnostic import run
        run()
    else:
        main()
