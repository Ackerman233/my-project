"""Compare all four predeclared development pairs, preserving censoring and call provenance."""
from pathlib import Path
from experiments.interrupt_recovery_ablation import read
from experiments.puzzle_pilot import save


def compare():
    dynamic = Path('results/protocol_diagnostic_v5_20261007')
    fixed = Path('results/subskill_fixed_20261007')
    dc, fc = read(dynamic / 'config.json'), read(fixed / 'config.json')
    assert dc['tasks'] == fc['tasks'] and dc['settings'] == fc['settings']
    assert dc['policy'] == fc['policy'] and dc['task_skill'] == fc['task_skill']
    assert all(fc['source_hashes'][p] == sha for p, sha in dc['source_hashes'].items()
               if p != 'experiments/campaign_runner.py')
    dr = {read(p)['task_id']: p for p in dynamic.glob('online/*/predictions/*/result.json')}
    fr = {read(p)['task_id']: p for p in fixed.glob('online/*/predictions/*/result.json')}
    pairs = []
    for task in dc['tasks']:
        task_id = task['id']
        if task_id not in dr or task_id not in fr:
            pairs.append({'task_id': task_id, 'paired': False})
            continue
        d, f = read(dr[task_id]), read(fr[task_id])
        dcall = read(dr[task_id].parent / 'model_calls.json')
        fcall = read(fr[task_id].parent / 'model_calls.json')
        pairs.append({'task_id': task_id, 'paired': True,
            'initial_prompt_identical': dcall[0]['prompt'] == fcall[0]['prompt'] and
                                        dcall[0]['system_prompt'] == fcall[0]['system_prompt'],
            'dynamic': {k: d.get(k) for k in ['success', 'status', 'model_calls', 'environment_steps',
                 'interrupt_count', 'invalid_actions', 'cost_usd']},
            'fixed': {k: f.get(k) for k in ['success', 'status', 'model_calls', 'environment_steps',
                 'interrupt_count', 'invalid_actions', 'cost_usd']}})
    both = [p for p in pairs if p['paired'] and p['dynamic']['success'] and p['fixed']['success']]
    result = {'pairs': pairs, 'scheduled_pairs': len(pairs),
        'completed_pairs': sum(p['paired'] for p in pairs),
        'dynamic_successes': sum(p['dynamic']['success'] for p in pairs if p['paired']),
        'fixed_successes': sum(p['fixed']['success'] for p in pairs if p['paired']),
        'both_successful_pairs': len(both),
        'both_successful_calls': {arm: sum(p[arm]['model_calls'] for p in both) for arm in ['dynamic', 'fixed']},
        'note': 'Development pairs, one run per arm, fixed arm after dynamic. Gateway variation/order confounds remain. '
                'Unsuccessful/censored episodes are not counted as efficient successes. '
                'Differences without actual interrupts cannot be attributed to interrupt.'}
    save(fixed / 'paired_comparison.json', result)
    print({k: v for k, v in result.items() if k != 'pairs'})
    return result


if __name__ == '__main__':
    compare()
