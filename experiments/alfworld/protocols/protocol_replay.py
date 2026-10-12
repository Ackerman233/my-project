"""Replay saved responses through the current parser; never execute actions or call APIs."""
import argparse
import json
from pathlib import Path

from dynamic_skill.micro_loop_protocol import parse_response
from dynamic_skill.space_subskills import expand, normalize_response
from experiments.interrupt_recovery_ablation import digest
from experiments.puzzle_pilot import save


def replay(roots, output):
    rows = []
    for root in roots:
        for path in sorted(root.glob('online/*/predictions/*/model_calls.json')):
            initial = json.loads((path.parent / 'initial.json').read_text())
            for call in json.loads(path.read_text()):
                row = {'source': str(path), 'source_sha256': digest(path),
                       'call_id': call['call_id'], 'original_status': call['status'],
                       'accepted_before': False, 'accepted_after': False}
                raw = call.get('response', '')
                limit = call.get('effective_action_limit', call['action_limit'])
                try:
                    parse_response(raw, limit)
                    row['accepted_before'] = True
                except ValueError:
                    pass
                text, changed = normalize_response(raw)
                row['protocol_normalized'] = changed
                try:
                    actions, _ = parse_response(text, limit)
                    # Use the recorded current public action list, not today's state.
                    tail = call['prompt'].split('\nAdmissible now:\n', 1)[1]
                    admissible, _ = json.JSONDecoder().raw_decode(tail)
                    primitive, provenance = expand(actions, {
                        'initial': initial['observation'], 'admissible': admissible}, limit)
                    row.update(accepted_after=True, model_actions=actions,
                               expanded_actions=primitive, expansion=provenance)
                except (ValueError, IndexError, KeyError) as exc:
                    row['replay_error'] = str(exc)
                rows.append(row)
    result = {'scope': 'offline parser and expansion replay; no execution or new task success',
              'calls': len(rows), 'original_format_errors': sum(r['original_status'] == 'format_error' for r in rows),
              'repaired_format_errors': sum(r['original_status'] == 'format_error' and r['accepted_after'] for r in rows),
              'remaining_rejections': sum(not r['accepted_after'] for r in rows),
              'source_hashes': {p: digest(Path(p)) for p in [
                  'dynamic_skill/space_subskills.py', 'dynamic_skill/micro_loop_protocol.py',
                  'experiments/protocol_replay.py']}, 'rows': rows}
    save(output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'source_hashes')}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots', nargs='+', type=Path, default=[
        Path('results/space_subskills_20261003'), Path('results/space_subskills_v2_20261003')])
    parser.add_argument('--output', type=Path, default=Path('results/protocol_diagnostic_20261007/replay.json'))
    args = parser.parse_args()
    replay(args.roots, args.output)
