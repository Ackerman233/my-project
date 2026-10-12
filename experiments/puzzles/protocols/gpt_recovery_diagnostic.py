"""Fresh user-authorized GPT-5.4 budget. Frozen-prompt diagnostic, not episodes."""
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

from dynamic_skill.campaign_budget import CampaignBudget, BudgetPlanner, CampaignHalt
from dynamic_skill.alfworld_rollout import parse_action_chunk
from dynamic_skill.puzzles import make_environment
from experiments.puzzle_pilot import save

ROOT = Path('results/gpt54_recovery_20260928')


def request_options():
    return {'model': 'gpt-5.4', 'reasoning_effort': 'none',
            'temperature': 0, 'max_completion_tokens': 512}


def cases():
    tasks = {t['id']: t for t in json.loads(Path('data/curriculum_20260927/tasks.json').read_text())}
    c02 = Path('results/campaign_c02_candidate_20260927T164456Z/predictions')
    base = Path('results/campaign_rung0_20260927T151906Z/predictions')
    specs = [(c02, 'easy-rush-1-1', 'dynamic_interrupt', n, 4) for n in (1, 2, 3)]
    specs += [(base, task, 'react', 1, 1) for task in ('easy-rush-1-0', 'easy-sudoku-1-0', 'easy-sudoku-1-1')]
    for root, task, mode, number, limit in specs:
        directory = root / (task + '__' + mode)
        calls = json.loads((directory / 'model_calls.json').read_text())
        record = calls[number - 1]
        env = make_environment(tasks[task])
        path = directory / 'conversation.json'
        trace = json.loads(path.read_text()) if path.exists() else []
        for step in trace:
            if step['step'] >= record['step']:
                break
            replay = env.step(step['action'])
            assert replay.observation == step['board_after']
        assert env.render() in record['prompt']
        yield {'id': task + '__call' + str(number), 'source': str(directory),
               'source_call_id': number, 'messages': [{'role': 'system', 'content': record['system_prompt']},
                                                    {'role': 'user', 'content': record['prompt']}],
               'original_status': record['status'], 'original_response': record['response'],
               'limit': limit, 'board': env.render()}, env


def evaluate(env, text, limit):
    actions = parse_action_chunk(text, limit)
    probe = deepcopy(env)
    steps = []
    for action in actions:
        result = probe.step(action)
        steps.append({'action': action, 'valid': result.valid, 'won': result.won,
                      'feedback': result.reason, 'board_after': result.observation})
        if result.won:
            break
    return {'actions': actions, 'first_action_valid': steps[0]['valid'],
            'all_executed_valid': all(s['valid'] for s in steps),
            'chunk_reaches_goal': probe.solved, 'steps': steps}


def authorize_single_retry(budget):
    data = budget.read()
    if len(data['requests']) != 2 or data['requests'][-1].get('error_type') != 'APIConnectionError':
        raise RuntimeError('retry only valid for the recorded second connection error')
    if data.get('retry_authorizations'):
        raise RuntimeError('single retry authorization already consumed')
    data['retry_authorizations'] = [{'at_unix': time.time(), 'retry_of_request_id': 2,
        'reason': 'User explicitly requested retry with the same endpoint and key; same $5 total.',
        'previous_max_unknown': data['max_unknown_requests'], 'previous_max_requests': data['max_requests']}]
    data['max_unknown_requests'] = 2
    data['max_requests'] = 3  # Exactly one additional request, not a restarted campaign.
    budget.write(data)


def error_chain(exc):
    # Types/errno only: never serialize headers, tokens, URLs or response bodies.
    rows, seen = [], set()
    while exc is not None and id(exc) not in seen and len(rows) < 8:
        seen.add(id(exc))
        rows.append({'type': type(exc).__name__, 'errno': getattr(exc, 'errno', None)})
        exc = exc.__cause__ or exc.__context__
    return rows


def authorize_remaining_probes(budget):
    data = budget.read()
    if len(data['requests']) != 3 or data.get('continuation_authorizations'):
        raise RuntimeError('remaining-probe authorization requires exactly three prior calls, once only')
    data['continuation_authorizations'] = [{'at_unix': time.time(),
        'reason': 'User requested next experiment; four previously unattempted frozen probes only.',
        'previous_max_requests': data['max_requests'], 'additional_requests': 4}]
    data['max_requests'] = 7
    budget.write(data)  # Preserve costs, unknown threshold, deadline, cooldown and policy guards.


def run(retry=False, remaining=False):
    from openai import OpenAI
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        budget = CampaignBudget(ROOT / 'budget.json',
            deadline=datetime(2026, 9, 28, 3, 17, tzinfo=timezone.utc).timestamp(),
            input_rate=25, output_rate=150, max_requests=6, max_unknown=1)
        if retry and remaining:
            raise ValueError('retry and remaining are mutually exclusive')
        out = ROOT / ('remaining_1' if remaining else 'retry_1' if retry else 'diagnostic')
        out.mkdir(exist_ok=False)
        selected = list(cases())
        if retry:
            selected = [selected[1]]
            authorize_single_retry(budget)
        if remaining:
            selected = selected[2:]
            authorize_remaining_probes(budget)
        save(out / 'config.json', {'request': request_options(), 'cases': [c for c, _ in selected],
             'budget': str(budget.path), 'price_note': '25/150 USD per million conservative reservation, not verified reseller billing.',
             'kind': 'four remaining frozen probes, protocol v2 separates output failures' if remaining else 'explicit one-request retry of connection error' if retry else 'six seen frozen-state development probes, not online episode success'})
        rows = []
        client = OpenAI(base_url='https://api.inferera.com/v1', api_key=os.environ['TARGET_AZURE_OPENAI_API_KEY'],
                        timeout=45, max_retries=0)
        try:
            for case, env in selected:
                def api(prompt):
                    reply = client.chat.completions.create(messages=case['messages'], **request_options())
                    choice = reply.choices[0]
                    return {'response': choice.message.content or '',
                            'response_id': reply.id, 'provider_model': reply.model,
                            'usage': reply.usage.model_dump() if reply.usage else None,
                            'reasoning_content': getattr(choice.message, 'reasoning_content', None),
                            'finish_reason': choice.finish_reason}
                planner = BudgetPlanner(api, budget, str(out) + '/' + case['id'])
                row = {**case, 'status': 'pending', 'request': request_options(),
                       'prompt_sha256': hashlib.sha256(json.dumps(case['messages']).encode()).hexdigest()}
                start = time.perf_counter()
                try:
                    planner.before_call()
                except CampaignHalt as halt:
                    save(out / 'halt.json', {'status': halt.status})
                    break
                rows.append(row)
                save(out / 'calls.json', rows)
                try:
                    row.update(planner(None))
                    row['api_status'] = 'ok'
                    row['status'] = 'ok'
                    row['evaluation'] = evaluate(env, row['response'], case['limit'])
                except Exception as exc:
                    row['status'] = 'output_error' if row.get('api_status') == 'ok' and isinstance(exc, ValueError) else 'error'
                    row['error_type'] = type(exc).__name__
                    row['http_status'] = getattr(exc, 'status_code', None)
                    row['error_chain'] = error_chain(exc)
                row['latency_seconds'] = time.perf_counter() - start
                if case['original_status'] == 'ok':
                    row['original_evaluation'] = evaluate(env, case['original_response'], case['limit'])
                save(out / 'calls.json', rows)
                print(json.dumps({k:row.get(k) for k in ('id', 'status', 'evaluation', 'error_type', 'http_status', 'latency_seconds')}), flush=True)
                state = budget.read()
                if (row['status'] != 'ok' and not (remaining and row['status'] == 'output_error')) or state['policy_violation'] or state['requests'][-1]['status'] != 'returned_usage':
                    break
        finally:
            client.close()
        save(out / 'summary.json', {'scheduled_probes': len(selected), 'attempted': len(rows),
             'ok': sum(r['status'] == 'ok' for r in rows),
             'valid_first': sum(r.get('evaluation', {}).get('first_action_valid', False) for r in rows),
             'goal_reached': sum(r.get('evaluation', {}).get('chunk_reaches_goal', False) for r in rows),
             'finished': len(rows) == len(selected) and all(r['status'] == 'ok' for r in rows)})
