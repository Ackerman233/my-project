"""Source-backed protocol/exposure audit, without API calls or hidden-state access."""
import argparse
from collections import Counter
from pathlib import Path
from experiments.interrupt_recovery_ablation import read, config_hash
from experiments.puzzle_pilot import save


def audit(root):
    cfg = read(root / 'config.json')
    episodes, review = [], []
    for path in sorted(root.glob('online/*/predictions/*/result.json')):
        result = read(path)
        calls = read(path.parent / 'model_calls.json')
        trace_path = path.parent / 'conversation.json'
        trace = read(trace_path) if trace_path.exists() else []
        executed = Counter(r['model_call_id'] for r in trace)
        multi = [c for c in calls if len(c.get('actions', [])) > 1]
        episodes.append({
            'episode': str(path.parent.relative_to(root)), 'status': result['status'],
            'success': result['success'], 'calls': len(calls),
            'normalized_calls': sum(c.get('protocol_normalized', False) for c in calls),
            'format_errors': sum(c['status'] == 'format_error' for c in calls),
            'parse_errors': [c.get('parse_error', 'not_recorded') for c in calls if c['status'] == 'format_error'],
            'planned_primitive_actions': sum(len(c.get('actions', [])) for c in calls),
            'executed_primitive_actions': len(trace), 'multi_action_calls': len(multi),
            'multi_action_calls_executing_multiple': sum(executed[c['call_id']] > 1 for c in multi),
            'program_subskill_calls': sum(bool(c.get('subskill_expansion')) for c in calls),
            'native_multi_primitive_calls': sum(len(c.get('model_native_actions', c.get('actions', []))) > 1
                and not any(a.startswith('@') for a in c.get('model_native_actions', c.get('actions', []))) for c in calls),
            'invalid_actions': result.get('invalid_actions', 0),
            'interrupt_count': result.get('interrupt_count', 0),
            'discovery_pauses': result.get('discovery_pauses', 0),
            'ledger_cost_usd': result.get('cost_usd'),
        })
        for step in trace:
            discarded = step.get('actual_discarded_actions', [])
            if discarded and step.get('interruption_reason'):
                review.append({'episode': str(path.parent.relative_to(root)), 'step': step['step'],
                    'reason': step['interruption_reason'], 'discarded': discarded,
                    'feedback': step['env_feedback'], 'label': 'unknown',
                    'note': 'Requires paired/counterfactual review; subsequent success alone is insufficient.'})
    statuses = Counter(e['status'] for e in episodes)
    complete = len(episodes) == len(cfg['job_order'])
    errors = sum(e['format_errors'] for e in episodes)
    infra = sum(n for status, n in statuses.items() if status not in (
        'success', 'format_error', 'model_call_limit', 'step_limit', 'wall_time_limit',
        'environment_failure', 'interrupt_budget_exhausted'))
    report = {'config_sha256': config_hash(cfg), 'scheduled': len(cfg['job_order']),
        'attempted': len(episodes), 'statuses': dict(statuses),
        'protocol_gate_passed': complete and errors == 0 and infra == 0,
        'mechanism_exposed': any(e['interrupt_count'] for e in episodes),
        'episodes': episodes, 'interrupt_review': review,
        'interpretation': 'Development diagnostic only. Program expansion is not model-native chunking. '
                          'No observed interrupt means interrupt efficacy is unmeasured. '
                          'Cost is conservative shared-ledger accounting, not provider billing.'}
    save(root / 'diagnostic_audit.json', report)
    print({k: v for k, v in report.items() if k not in ('episodes', 'interrupt_review')})
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    audit(parser.parse_args().root)
