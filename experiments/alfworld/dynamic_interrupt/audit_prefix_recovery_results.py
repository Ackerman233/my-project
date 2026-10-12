"""Inspect paired recovery outcomes, protocol censoring, and action/state use."""
from collections import Counter
import re
from pathlib import Path
from experiments.interrupt_recovery_ablation import read
from experiments.puzzle_pilot import save

ROOT = Path('results/prefix_recovery_20261007')


def audit():
    cfg = read(ROOT / 'config.json')
    pairs = []
    for case in cfg['cases']:
        rows = {}
        for arm in cfg['arms']:
            out = ROOT / 'online' / case['id'] / arm
            result = read(out / 'result.json')
            suffix = read(out / 'suffix_trace.json')
            prefix = read(out / 'prefix.json')
            prior_visits = Counter(r['action'][6:] for r in prefix if r['action'].startswith('go to '))
            revisits = []
            for step in suffix:
                if step['action'].startswith('go to '):
                    loc = step['action'][6:]
                    if prior_visits[loc]:
                        revisits.append({'suffix_step': step['suffix_step'], 'location': loc,
                                         'feedback': step['env_feedback']})
                    prior_visits[loc] += 1
            calls = read(out / 'model_calls.json')
            rows[arm] = {**result, 'suffix_revisits': revisits,
                'original_tail_invalid_actions': sum(r['original_tail_action'] and r['checker_status'] == 'invalid_action' for r in suffix),
                'original_tail_supported_actions': sum(r['original_tail_action'] and r['action'] in r['admissible_before'] for r in suffix),
                'parse_errors': [{'call_id': c['call_id'], 'error': c.get('parse_error'),
                                  'finish_reason': c.get('finish_reason')} for c in calls if c['status'] == 'format_error'],
                'lamp_discovery_suffix_steps': [r['suffix_step'] for r in suffix if
                    ('you see' in r['env_feedback'].lower() and re.search(r'\bdesklamp [0-9]+\b', r['env_feedback'].lower())) or
                    any(a.startswith('use desklamp ') for a in (
                        suffix[r['suffix_step'] + 1]['admissible_before'] if r['suffix_step'] + 1 < len(suffix)
                        else []))],
                'suffix_actions': [r['action'] for r in suffix]}
        assert rows['continue_tail']['checkpoint_sha256'] == rows['interrupt_replan']['checkpoint_sha256']
        comparable = all(r['status'] in ('success', 'model_call_limit', 'step_limit', 'wall_time_limit',
                                         'environment_failure', 'interrupt_budget_exhausted') for r in rows.values())
        pairs.append({'case_id': case['id'], 'kind': case['kind'],
                      'protocol_valid_pair': comparable, 'arms': rows})
    valid = [p for p in pairs if p['protocol_valid_pair']]
    report = {'paired_states': len(pairs), 'distinct_gamefiles': len({read(Path(c['source_trace']).parent / 'initial.json')['gamefile'] for c in cfg['cases']}),
              'distinct_checkpoint_hashes': len({c['checkpoint_sha256'] for c in cfg['cases']}),
              'protocol_valid_pairs': len(valid),
              'valid_pair_successes': {a: sum(p['arms'][a]['success'] for p in valid) for a in cfg['arms']},
              'all_statuses': {a: dict(Counter(p['arms'][a]['status'] for p in pairs)) for a in cfg['arms']},
              'pairs': pairs, 'note': 'Six state pairs are not six independent tasks. Protocol failures are excluded from the paired success interpretation. '
              'Suffix revisits can be necessary; correctness requires task/possession context. One model sample per arm, not a stable causal effect.'}
    save(ROOT / 'recovery_audit.json', report)
    print({k: v for k, v in report.items() if k != 'pairs'})


if __name__ == '__main__':
    audit()
