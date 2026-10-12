"""Offline verification of the completed one-run snapshot, no API or ledger edits."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))
from dynamic_skill.evolution import EvolvedSkill, digest, validation_gate
from dynamic_skill.puzzles import make_environment
from experiments.puzzle_pilot import save


def main():
    root = Path('results/evolution_small_20260928')
    read = lambda p: json.loads(p.read_text())
    status = read(root/'run_status.json')
    assert status['status'] == 'completed'
    manifest = read(root/'manifest.json')
    tasks = {t['id']:t for group in manifest['splits'].values() for t in group}
    candidate = read(root/'generation_000/candidate.json')
    best = read(root/'best_skill.json')
    assert best == candidate and best['revision'] == 1
    current = read(root/'generation_000/validation_incumbent/results.json')
    proposed = read(root/'generation_000/validation_candidate/results.json')
    assert validation_gate(current, proposed)['accept']
    reflection = read(root/'generation_000/proposal_evidence.json')
    train_ids = {t['id'] for t in manifest['splits']['train']}
    assert all(e['task_id'] in train_ids for e in reflection)
    assert {e['success'] for e in reflection} == {True, False}
    known_skills = {EvolvedSkill().hash, digest(best)}
    episodes, requests, steps, errors, reasoning, latencies = 0, 0, 0, 0, 0, []
    for result_path in root.glob('**/predictions/*/result.json'):
        result = read(result_path)
        env = make_environment(tasks[result['task_id']])
        trace = read(result_path.parent/'conversation.json')
        calls = read(result_path.parent/'model_calls.json')
        assert len(calls) == result['model_calls'] and len(trace) == result['environment_steps']
        for row in trace:
            reply = calls[row['model_call_id']-1]
            assert row['action'] == reply['actions'][row['source_action_index']]
            replay = env.step(row['action'])
            assert replay.observation == row['board_after'] and replay.valid == row['valid'] and replay.won == row['won']
        assert env.solved == result['success']
        assert result['skill_hash'] is None or result['skill_hash'] in known_skills
        for call in calls:
            assert call['status'] == 'ok'
            reasoning += ((call.get('usage') or {}).get('completion_tokens_details') or {}).get('reasoning_tokens') or 0
            latencies.append(call['latency_seconds'])
        requests += len(calls); steps += len(trace); episodes += 1
    assert reasoning == 0
    # Another authorized campaign can acquire the shared ledger after this run.
    # Audit the snapshot saved BEFORE releasing our lock; never mutate its budget.
    state = read(root/'budget_snapshot.json')
    auth = state['small_evolution_authorization']
    new = state['requests'][auth['previous_requests']:]
    assert len(new) == requests + 1
    assert all(r['status'] == 'returned_usage' and r['label'].startswith(str(root)) for r in new)
    assert not state['policy_violation']
    total = sum(r['budget_charge_usd'] for r in state['requests'])
    incremental = sum(r['budget_charge_usd'] for r in new)
    audit = {'verified': True, 'episodes': episodes, 'actor_calls': requests, 'reflection_calls': 1,
        'replayed_environment_steps': steps, 'reported_reasoning_tokens': reasoning,
        'incremental_conservative_charge_usd': incremental, 'total_conservative_charge_usd': total,
        'actor_latency_seconds': sum(latencies), 'unknown_reserve_usd': sum(r['budget_charge_usd'] for r in state['requests'] if r['status']=='unknown_usage'),
        'checks': ['fixed train/validation/test manifest', 'reflection evidence train-only', 'accepted candidate hash',
                   'validation gate recomputed', 'all actions tied to generated queues', 'all transitions replayed',
                   'no API failures or reported reasoning in this run', 'same cumulative budget snapshot at this run completion; shared live ledger left untouched']}
    save(root/'audit.json', audit)
    print(json.dumps(audit, indent=2))
    print(json.dumps(read(root/'comparison_summary.json'), indent=2))


if __name__ == '__main__':
    main()
