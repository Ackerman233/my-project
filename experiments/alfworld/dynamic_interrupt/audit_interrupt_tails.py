"""Replay real prefixes then execute discarded tails in fresh real environments. No API."""
import argparse
import os
from pathlib import Path
from experiments.interrupt_recovery_ablation import read, digest
from experiments.puzzle_pilot import save


def audit(roots, output):
    os.environ.setdefault('ALFWORLD_DATA', '/opt/data/private/data/alfworld')
    from skillopt.envs.alfworld.rollout import build_alfworld_env
    rows = []
    for root in roots:
        cfg = read(root / 'config.json')
        hashes = {t['gamefile']: t['game_sha256'] for t in cfg['tasks']}
        for path in sorted(root.glob('online/*/predictions/*/conversation.json')):
            trace = read(path)
            initial = read(path.parent / 'initial.json')
            gamefile = initial['gamefile']
            assert digest(Path(gamefile)) == hashes[gamefile]
            for index, cut in enumerate(trace):
                tail = cut.get('actual_discarded_actions', [])
                if not tail or not cut.get('interruption_reason'):
                    continue
                row = {'source': str(path), 'source_sha256': digest(path), 'cut_step': cut['step'],
                       'reason': cut['interruption_reason'], 'discarded_actions': tail,
                       'prefix_verified': False, 'tail_steps': [], 'label': 'unknown'}
                env = build_alfworld_env(env_num=1, eval_dataset='eval_in_distribution', seed=42,
                    is_train=False, specific_gamefiles=[gamefile])
                try:
                    obs, _ = env.reset({})
                    assert obs['anchor'][0] == initial['observation'], 'initial observation mismatch'
                    for step in trace[:index + 1]:
                        assert list(env.envs.get_admissible_commands[0]) == step['admissible_before'], 'public action list mismatch'
                        obs, rewards, dones, infos = env.step(['<think></think><action>' + step['action'] + '</action>'])
                        assert obs['anchor'][0] == step['env_feedback'], 'prefix feedback mismatch'
                        assert bool(dones[0]) == step['done'], 'prefix terminal mismatch'
                    assert not dones[0], 'terminal cancellations are excluded'
                    row['prefix_verified'] = True
                    for action in tail:
                        admissible = list(env.envs.get_admissible_commands[0])
                        obs, rewards, dones, infos = env.step(['<think></think><action>' + action + '</action>'])
                        row['tail_steps'].append({'action': action, 'admissible_before': admissible,
                            'feedback': obs['anchor'][0], 'done': bool(dones[0]),
                            'won': bool(infos[0].get('won', False)), 'reward': float(rewards[0])})
                        if dones[0]:
                            break
                    if any(s['won'] or s['reward'] > 0 for s in row['tail_steps']):
                        row['label'] = 'tail_would_complete_task'
                    elif any(s['feedback'].strip().lower() == 'nothing happens.' for s in row['tail_steps']):
                        row['label'] = 'tail_contains_rejected_action'
                    row['note'] = 'Local counterfactual only; no replanning or full paired episode success comparison.'
                except AssertionError as exc:
                    row['replay_mismatch'] = str(exc)
                finally:
                    env.close()
                rows.append(row)
    report = {'paid_calls': 0, 'source': 'real-model discarded tails, fresh real environments',
              'cuts': len(rows), 'verified_prefixes': sum(r['prefix_verified'] for r in rows),
              'rows': rows, 'note': 'This audits local discarded actions, not overall interrupt performance.'}
    save(output, report)
    print({k: v for k, v in report.items() if k != 'rows'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.roots, args.output)
