"""Offline candidate: remove only the unsupported first action, retaining the rest."""
from pathlib import Path
from experiments.interrupt_recovery_ablation import read, config_hash
from experiments.prefix_recovery_trial import environment, replay_prefix
from experiments.puzzle_pilot import save

ROOT = Path('results/prefix_recovery_20261007')


def audit():
    cfg = read(ROOT / 'config.json')
    rows = []
    for case in cfg['cases']:
        initial = read(Path(case['source_trace']).parent / 'initial.json')
        env = environment(initial['gamefile'])
        try:
            _, _, checkpoint = replay_prefix(env, case)
            assert config_hash(checkpoint) == case['checkpoint_sha256']
            tail = list(case['tail'])
            removed = []
            if tail[0] not in checkpoint['admissible']:
                # Only the first action's precondition is testable at this exact state.
                removed.append(tail.pop(0))
            executed = []
            for action in tail:
                before = list(env.envs.get_admissible_commands[0])
                obs, rewards, dones, infos = env.step(['<think></think><action>' + action + '</action>'])
                executed.append({'action': action, 'admissible_before': before,
                                 'feedback': obs['anchor'][0], 'done': bool(dones[0]),
                                 'won': bool(infos[0].get('won', False))})
                if dones[0]:
                    break
            rows.append({'case_id': case['id'], 'checkpoint_sha256': config_hash(checkpoint),
                         'removed_head': removed, 'retained_tail': tail, 'executed': executed})
        finally:
            env.close()
    report = {'paid_calls': 0, 'cases': len(rows), 'removed_actions': sum(len(r['removed_head']) for r in rows),
              'retained_actions': sum(len(r['executed']) for r in rows),
              'retained_publicly_supported_actions': sum(e['action'] in e['admissible_before'] for r in rows for e in r['executed']),
              'rows': rows, 'scope': 'Offline local candidate only. No model recovery or task success comparison. '
                                   'Not a general justification for rejecting actions absent from a current list later in a queue.'}
    save(ROOT / 'selective_tail_audit.json', report)
    print({k: v for k, v in report.items() if k != 'rows'})


if __name__ == '__main__':
    audit()
