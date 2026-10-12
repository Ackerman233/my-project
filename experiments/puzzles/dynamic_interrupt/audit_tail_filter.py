"""Offline C02 development comparison; never an online model score."""
import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))
from dynamic_skill.puzzles import make_environment
from dynamic_skill.tail_filter import filter_tail
from experiments.audit_discarded_tails import audit


def compare(root):
    baseline = audit(root)
    tasks = {t['id']: t for t in json.loads((root / 'tasks.json').read_text())}
    rows = [json.loads(s) for s in (root / 'online_attempt/results.jsonl').read_text().splitlines()]
    events = []
    for row in rows:
        if row['mode'] != 'dynamic_interrupt':
            continue
        path = root / 'online_attempt/predictions' / row['id'] / 'conversation.json'
        if not path.exists():
            continue
        env = make_environment(tasks[row['task_id']])
        history = [env.render()]
        for step in json.loads(path.read_text()):
            before = env.render()
            result = env.step(step['action'])
            assert result.observation == step['board_after'] and result.valid == step['valid']
            if step['controller_decision'] == 'interrupt_and_replan' and step['discarded_actions']:
                filtered = filter_tail(env, step['discarded_actions'], failed_valid=result.valid,
                                       before_failed=before, recent_states=history[-4:])
                replay = deepcopy(env)
                states = set(history[-4:]) | {env.render()}
                for index, action in zip(filtered.retained_indices, filtered.retained):
                    assert action == step['discarded_actions'][index]
                    replay_result = replay.step(action)
                    assert replay_result.valid
                    assert replay.render() not in states
                    states.add(replay.render())
                assert list(filtered.retained_indices) == sorted(set(filtered.retained_indices))
                events.append({'episode': row['id'], 'step': step['step'],
                               'original_tail': step['discarded_actions'], **asdict(filtered),
                               'retained_replay_wins': replay.solved,
                               'independent_retained_replay_passed': True})
            history.append(env.render())
    assert len(events) == baseline['summary']['interrupt_events']
    return {'protocol': 'c02_offline_dev_v1', 'online_performance_claim': False,
            'source_tasks_sha256': baseline['tasks_sha256'], 'source_hashes': baseline['source_hashes'],
            'summary': {'events': len(events), 'decisions': dict(Counter(e['decision'] for e in events)),
                        'reasons': dict(Counter(e['reason'] for e in events)),
                        'retained_actions': sum(len(e['retained']) for e in events),
                        'skipped_actions': sum(len(e['skipped']) for e in events),
                        'simulation_checks': sum(e['checks'] for e in events),
                        'tail_wins': sum(e['retained_replay_wins'] for e in events),
                        'filter_seconds_total': sum(e['seconds'] for e in events),
                        'filter_seconds_median': statistics.median(e['seconds'] for e in events) if events else None},
            'limitations': 'Seen development states, correlated repeated tails. No future plans reused. Simulation and deepcopy costs are measured locally, not API savings. Legal retained tails do not imply strategic progress.',
            'events': events}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.root)
    with args.out.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report['summary'], indent=2))
