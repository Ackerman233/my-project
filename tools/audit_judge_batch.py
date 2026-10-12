"""Read-only role, queue, prefix and ledger audit. No model requests."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common.bounded_campaign import StageBudget, batch_accounting
from experiments.alfworld.judge_interrupt.cli import summarize
from experiments.common.puzzle_pilot import save


def read(path):
    return json.loads(path.read_text())


def audit(root):
    root = root.resolve()
    cfg = read(root / 'config.json')
    rows = read(root / 'run/results.json') if (root / 'run/results.json').exists() else []
    budget = StageBudget(root)
    ledger = budget.read()
    requests = [r for r in ledger['requests'] if r['label'].startswith(str(root) + '/')]
    gates, skips, decisions, changes = Counter(), Counter(), Counter(), []
    initials, actor_errors = {}, []
    coverage = {arm: Counter() for arm in cfg['arms']}
    unchanged_failures = []
    role_totals = {}
    for role in ('actor', 'judge'):
        subset = [r for r in requests if r['label'].endswith('/' + role)]
        role_totals[role] = {'requests': len(subset), 'cost_usd': sum(r['budget_charge_usd'] for r in subset),
                            'unknown_usage': sum(r['status'] == 'unknown_usage' for r in subset)}
        records = [read(p) for p in (root / 'transport').glob('*.json')
                   if p.name.count('.') == 1 and read(p).get('label', '').endswith('/' + role)]
        for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            role_totals[role][key] = sum((r.get('usage') or {}).get(key, 0) or 0 for r in records)

    for row in rows:
        out = root / 'run' / row['split'] / row['task_id'] / row['arm']
        pred = out / 'predictions' / row['id']
        trace = read(pred / 'conversation.json') if (pred / 'conversation.json').exists() else []
        actor = read(pred / 'model_calls.json') if (pred / 'model_calls.json').exists() else []
        judges = read(out / 'judge_calls.json') if (out / 'judge_calls.json').exists() else []
        assert row['actor_calls'] == len(actor) and row['judge_calls'] == len(judges)
        assert row['total_model_calls'] == row['model_calls'] == len(actor) + len(judges) <= cfg['settings']['max_calls']
        assert len(judges) <= cfg['judge_config']['max_calls']
        assert row['success'] == (row['status'] == 'success')
        assert row['environment_steps'] == len(trace)
        labels = (str(out / row['task_id'] / 'actor'), str(out / row['task_id'] / 'judge'))
        cost = sum(r['budget_charge_usd'] for r in requests if r['label'] in labels)
        assert abs(row['cost_usd'] - cost) < 1e-8
        sha = hashlib.sha256((pred / 'initial.json').read_bytes()).hexdigest()
        initials.setdefault(row['task_id'], {})[row['arm']] = sha
        for call in actor:
            if call['status'] != 'ok':
                actor_errors.append({'task_id': row['task_id'], 'arm': row['arm'],
                    'call_id': call['call_id'], 'status': call['status'],
                    'finish_reason': call.get('finish_reason'), 'parse_error': call.get('parse_error')})
        queried = set()
        for step in trace:
            coverage[row['arm']]['environment_steps'] += 1
            if step.get('remaining_before_interrupt') and not step['done']:
                coverage[row['arm']]['nonterminal_nonempty_tails'] += 1
            check = step.get('judge')
            if not check:
                continue
            gates.update(check.get('gate_reasons', []))
            coverage[row['arm']]['gated_opportunities'] += bool(check.get('gate_reasons'))
            skips.update([check.get('skip') or check.get('status')])
            if check.get('queried'):
                assert step['remaining_before_interrupt'] and not step['done']
                call_id = check['call_id']
                assert call_id not in queried
                queried.add(call_id)
                coverage[row['arm']]['paid_judge_queries'] += 1
                reply = judges[call_id - 1]
                assert reply['context']['remaining_queue'] == step['remaining_before_interrupt']
                assert reply['context']['observation'] == step['env_feedback']
                assert set(reply['context']) == {'task', 'last_action', 'observation', 'remaining_queue',
                    'admissible_now', 'new_admissible_commands', 'recent_observations', 'decisions_so_far'}
                decisions.update([reply.get('decision')])
                if reply.get('decision') == 'continue':
                    assert not step['actual_discarded_actions']
                if reply.get('decision') == 'interrupt':
                    assert step['actual_discarded_actions'] == step['remaining_before_interrupt']
            if check.get('changed_rule_decision'):
                coverage[row['arm']]['rule_overrides'] += 1
                changes.append({'task_id': row['task_id'], 'arm': row['arm'], 'step': step['step'],
                    'action': step['action'], 'observation': step['env_feedback'],
                    'tail': step['remaining_before_interrupt'], 'rule_interrupt': check['rule_should_interrupt'],
                    'judge_decision': check['decision'], 'evidence': check.get('evidence'),
                    'episode_success': row['success'],
                    'causal_gain': 'unknown_without_same_prefix_counterfactual'})
        assert len(queried) == len(judges)
        if not row['success'] and not row['rule_overrides']:
            unchanged_failures.append({'task_id': row['task_id'], 'arm': row['arm'], 'status': row['status']})
    assert all(len(set(v.values())) == 1 for v in initials.values())
    grouped = {split: summarize([r for r in rows if r['split'] == split], cfg)
               for split in ('selection', 'confirmation') if any(r['split'] == split for r in rows)}
    return {'completed_rows': len(rows), 'metrics': grouped, 'accounting': batch_accounting(root),
        'role_totals': role_totals, 'judge_decisions': dict(decisions), 'gate_signals': dict(gates),
        'coverage': {k: dict(v) for k, v in coverage.items()},
        'checks_or_skips': dict(skips), 'paired_initial_states_identical': True,
        'paired_initial_states': initials, 'overrides': changes, 'actor_errors': actor_errors,
        'failed_episodes_without_rule_override': unchanged_failures,
        'validation': 'counts, total-call limits, role costs, public contexts, actual queue effects and paired initials verified',
        'caveat': 'Fresh tasks but only one rollout per arm; temperature0 is not a determinism guarantee. '
                  'Full-episode outcome differences may occur even with no judge override. Selection is not final test.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    if args.out:
        save(args.out, result)
    print(json.dumps({k: result[k] for k in ('completed_rows', 'accounting', 'judge_decisions', 'validation')}, indent=2))
