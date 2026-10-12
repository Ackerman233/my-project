"""Paid entry is dispatched ONLY by campaign_runner.py --skill-evolution.

An existing authorized ledger is mandatory; no automatic deadline/cap reset.
"""
import argparse
from dataclasses import asdict, replace
import fcntl
import json
from pathlib import Path
import sys
import time

from dynamic_skill.campaign_budget import CampaignBudget, BudgetPlanner, CampaignHalt
from dynamic_skill.evolution import EvolvedSkill, evolve, validate_splits
from experiments.puzzle_pilot import ApiPlanner, Settings, run_episode, save


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits', type=Path, required=True, help='JSON object: train/validation/test task lists')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--budget-json', type=Path, required=True, help='Existing authorized campaign budget')
    parser.add_argument('--model', choices=['gpt-5.4', 'qwen3.5-35b-a3b'], default='gpt-5.4')
    parser.add_argument('--generations', type=int, default=2)
    parser.add_argument('--max-steps', type=int, default=48)
    parser.add_argument('--max-calls', type=int, default=48)
    parser.add_argument('--chunk-size', type=int, default=4)
    parser.add_argument('--preflight', action='store_true', help='Validate inputs and budget without API calls or writes')
    args = parser.parse_args(argv)
    if min(args.generations, args.max_steps, args.max_calls) < 1:
        parser.error('positive limits required')
    splits = json.loads(args.splits.read_text(encoding='utf-8'))
    validate_splits(splits)
    initial = EvolvedSkill(max_actions=args.chunk_size)
    # Fail early if upstream SkillOpt was not installed; never silently replace it.
    from dynamic_skill.evolution import skillopt_document
    skillopt_document(initial)
    if not args.budget_json.is_file():
        parser.error('budget must already exist; this runner never creates or resets spending authorization')
    with (args.budget_json.parent / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        budget = CampaignBudget(args.budget_json, deadline=0)
        state = budget.read()
        if state['limit_usd'] > 5:
            parser.error('this protocol is bounded to at most $5')
        if state['policy_violation'] or time.time() >= state['deadline_unix'] or time.time() < state['cooldown_until']:
            parser.error('existing deadline/policy/cooldown guard is closed')
        if len(state['requests']) >= state['max_requests'] or sum(r['status'] in ('pending', 'unknown_usage') for r in state['requests']) >= state['max_unknown_requests']:
            parser.error('existing request/unknown guard is closed')
        if sum(r['budget_charge_usd'] for r in state['requests']) + state['reservation_usd'] > state['limit_usd']:
            parser.error('insufficient reserved budget')
        if args.out.exists():
            parser.error('output already exists; no implicit resume/overwrite')
        if args.preflight:
            print('Inputs and existing ledger verified; no API requests. Full-run affordability is not guaranteed.')
            return
        settings = Settings(model=args.model, action_only=True, max_tokens=512, request_timeout=45,
                            episode_seconds=300, max_steps=args.max_steps, max_calls=args.max_calls,
                            input_usd_per_million=state['input_usd_per_million'],
                            output_usd_per_million=state['output_usd_per_million'])

        def guarded(api, label):
            planner = BudgetPlanner(api, budget, label)
            class Guarded:
                def before_call(self):
                    self.ready = True
                def __call__(self, prompt):
                    if not getattr(self, 'ready', False):
                        raise RuntimeError('request not prepared')
                    self.ready = False
                    ledger = budget.read()
                    # Pessimistic UTF-8 byte token bound plus framing/system margin.
                    # Reject before reserving if the request can exceed one reservation.
                    system = api.system_prompt or ''
                    upper = ((len(prompt.encode()) + len(system.encode()) + 2048) * ledger['input_usd_per_million']
                             + api.settings.max_tokens * ledger['output_usd_per_million']) / 1e6
                    if upper > ledger['reservation_usd']:
                        raise CampaignHalt('request_reservation_too_small')
                    planner.before_call()
                    reply = planner(prompt)
                    current = budget.read()
                    if current['policy_violation']:
                        raise CampaignHalt('thinking_policy_stop')
                    if current['requests'][-1]['status'] != 'returned_usage':
                        raise CampaignHalt('unknown_cost_stop')
                    return reply
            return Guarded()

        def evaluator(tasks, skill, mode, output):
            rows = []
            for task in tasks:
                api = ApiPlanner(settings, action_limit=skill.max_actions if skill else 1)
                try:
                    row = run_episode(task, mode, settings, output,
                                      guarded(api, str(output) + '/' + task['id']), skill=skill)
                finally:
                    api.close()
                rows.append(row)
                save(output / 'results.json', rows)
                if row['status'] not in {'success', 'step_limit', 'model_call_limit', 'interrupt_budget_exhausted', 'format_error'}:
                    raise CampaignHalt('incomplete_episode_' + row['status'])
            return rows

        proposal_index = 0
        def proposer(prompt):
            nonlocal proposal_index
            proposal_index += 1
            api = ApiPlanner(replace(settings, max_tokens=1024), system_prompt='You propose bounded skill updates. Return JSON only; no analysis or reasoning.')
            planner = guarded(api, str(args.out) + f'/proposal_{proposal_index}')
            start = time.perf_counter()
            try:
                planner.before_call()
                reply = planner(prompt)
                reply['latency_seconds'] = time.perf_counter() - start
                return reply
            finally:
                api.close()
        try:
            evolve(splits, initial, evaluator, proposer, args.out, args.generations)
        except Exception as exc:
            if args.out.exists():
                save(args.out / 'run_status.json', {'status': 'stopped', 'error_type': type(exc).__name__,
                    'guard': exc.status if isinstance(exc, CampaignHalt) else None,
                    'note': 'Partial results are not a completed comparison; ledger retains unknown reservations.'})
            raise SystemExit('Evolution stopped safely; inspect run_status.json and budget.json.') from None
        finally:
            if args.out.exists():
                save(args.out / 'settings.json', asdict(settings))
                save(args.out / 'budget_snapshot.json', budget.read())


if __name__ == '__main__':
    raise SystemExit('Use python experiments/campaign_runner.py --skill-evolution ...')
