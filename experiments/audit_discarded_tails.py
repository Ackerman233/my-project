"""Offline counterfactual only: replay logged tails, never generate new actions.

Not an online policy score: future model outputs cannot be reused after divergence.
No solver, oracle, API client, or hidden solution is read.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dynamic_skill.puzzles import make_environment


def inspect_tail(env, actions):
    original = env.render()
    probe = deepcopy(env)
    rows = []
    for action in actions:
        before = probe.render()
        result = probe.step(action)
        if not result.valid:
            assert probe.render() == before, 'Skipping invalid commands requires no-op semantics'
        rows.append({'action': action, 'valid': result.valid, 'won': result.won,
                     'board_after': probe.render()})
        if result.won:
            break
    assert env.render() == original
    return {'steps': rows, 'first_remaining_legal': bool(rows and rows[0]['valid']),
            'legal_actions': [r['action'] for r in rows if r['valid']],
            'invalid_actions': [r['action'] for r in rows if not r['valid']],
            'counterfactual_tail_wins': probe.solved}


def audit(root):
    tasks_path = root / 'tasks.json'
    tasks = {t['id']: t for t in json.loads(tasks_path.read_text())}
    run = root / 'online_attempt'
    results = [json.loads(s) for s in (run / 'results.jsonl').read_text().splitlines()]
    events, hashes = [], {}
    for result in results:
        if result['mode'] != 'dynamic_interrupt':
            continue
        path = run / 'predictions' / result['id'] / 'conversation.json'
        if not path.exists():
            assert result['environment_steps'] == 0
            continue
        hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        env = make_environment(tasks[result['task_id']])
        for row in json.loads(path.read_text()):
            actual = env.step(row['action'])
            assert (actual.valid, actual.observation, actual.won) == (row['valid'], row['board_after'], row['won'])
            if row['controller_decision'] != 'interrupt_and_replan' or not row['discarded_actions']:
                continue
            events.append({'episode': result['id'], 'step': row['step'],
                           'observed_failed_action': row['action'],
                           'checker_status': row['checker_status'],
                           **inspect_tail(env, row['discarded_actions'])})
    signatures = {(e['episode'], tuple((r['action'], r['board_after']) for r in e['steps'])) for e in events}
    counts = Counter()
    for e in events:
        counts['interrupt_events'] += 1
        counts['discarded_actions_tested'] += len(e['steps'])
        counts['legal_tail_actions'] += len(e['legal_actions'])
        counts['events_with_legal_tail'] += bool(e['legal_actions'])
        counts['first_remaining_legal'] += e['first_remaining_legal']
        counts['tail_reaches_goal'] += e['counterfactual_tail_wins']
    return {'protocol': 'offline_tail_replay_v1', 'online_performance_claim': False,
            'limitations': 'Local legality is not strategic usefulness. Repeated states are correlated. No future logged plans reused after divergence. Skip-invalid equivalence is limited to verified no-op failures.',
            'tasks_sha256': hashlib.sha256(tasks_path.read_bytes()).hexdigest(),
            'source_hashes': hashes, 'summary': dict(counts),
            'distinct_episode_tail_outcomes': len(signatures), 'events': events}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.root)
    with args.out.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({k:v for k,v in report.items() if k not in {'events', 'source_hashes'}}, indent=2))
