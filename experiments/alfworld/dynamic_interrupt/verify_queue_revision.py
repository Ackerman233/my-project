"""Seeded public-board counterexamples and real ALFWorld prefixes. Zero API calls."""
from dataclasses import asdict
from pathlib import Path
import random
from unittest.mock import patch

from dynamic_skill.queue_evidence import RushHourEvidence, SudokuEvidence
from dynamic_skill.queue_revision import revise_head
from dynamic_skill.alfworld_open_guard import review_open_head
from dynamic_skill.puzzles import RushHourEnv, SudokuEnv
from experiments.interrupt_recovery_ablation import read, digest, config_hash
from experiments.prefix_recovery_trial import environment, replay_prefix
from experiments.puzzle_pilot import save

ROOT = Path('results/queue_revision_principles_20261007')


def denied(*args, **kwargs):
    raise AssertionError('Solver access prohibited')


def randomized():
    rng = random.Random(20261007)
    rows = []
    for index in range(250):
        env = RushHourEnv('......' '..B...' 'AAB...' '......' '...CC.' '......')
        for _ in range(rng.randrange(8)):
            options = [(a, s) for a, s in env.neighbors(env.state) if s[env.ids.index('A')] != 4]
            if options:
                env.step(rng.choice(options)[0])
        before = env.render()
        actions = [f'{car}{sign}{n}' for car in 'ABC' for sign in '+-' for n in range(1, 6)]
        legal = {a for a, _ in env.neighbors(env.state)}
        head = rng.choice([a for a in actions if a not in legal])
        following = rng.choice(actions)
        adapter = RushHourEvidence(before)
        revision = revise_head([head, following], adapter)
        assert env.render() == before
        if revision.next_action:
            assert env.step(revision.next_action).valid
            assert head[0] != following[0], 'retained relative command of same vehicle'
        rows.append({'domain': 'rush_hour', 'index': index, 'public_observation': before,
            'actions': [head, following], 'revision': asdict(revision),
            'naive_next_locally_invalid': following not in legal,
            'naive_same_vehicle_position_dependency': head[0] == following[0]})
    for index in range(250):
        grid = [[(r * 3 + r // 3 + c) % 9 + 1 if rng.random() < .25 else 0 for c in range(9)] for r in range(9)]
        env = SudokuEnv(grid)
        for _ in range(rng.randrange(5)):
            r, c, v = rng.randrange(1, 10), rng.randrange(1, 10), rng.randrange(1, 10)
            env.step(f'set({r},{c},{v})')
        before = env.render()
        actions = [f'clear({r},{c})' for r in range(1, 10) for c in range(1, 10)]
        actions += [f'set({r},{c},{v})' for r in range(1, 10) for c in range(1, 10) for v in range(1, 10)]
        adapter = SudokuEvidence(before)
        # Single-action legality checks reconstruct only the public board; no solution search.
        bad = [a for a in actions if adapter.executable(a).value.value == 'false']
        head, following = rng.choice(bad), rng.choice(actions)
        revision = revise_head([head, following], adapter)
        assert env.render() == before
        next_invalid = not adapter.env().step(following).valid
        if revision.next_action:
            assert env.step(revision.next_action).valid
        rows.append({'domain': 'sudoku', 'index': index, 'public_observation': before,
            'actions': [head, following], 'revision': asdict(revision),
            'naive_next_locally_invalid': next_invalid})
    report = {'paid_calls': 0, 'seed': 20261007, 'cases': len(rows),
        'domains': {domain: {'cases': sum(r['domain'] == domain for r in rows),
            'retained_next_actions': sum(r['domain'] == domain and r['revision']['next_action'] is not None for r in rows),
            'naive_locally_invalid_tail_heads': sum(r['domain'] == domain and r['naive_next_locally_invalid'] for r in rows),
            'retained_next_legality_violations': 0} for domain in ('rush_hour', 'sudoku')},
        'rows': rows, 'scope': 'Generated local-contract counterexamples, not model rollout scores or global solving guarantees.'}
    save(ROOT / 'randomized_audit.json', report)
    print({k: v for k, v in report.items() if k != 'rows'})


def alfworld():
    cfg = read(Path('results/prefix_recovery_20261007/config.json'))
    rows = []
    for case in cfg['cases']:
        initial = read(Path(case['source_trace']).parent / 'initial.json')
        env = environment(initial['gamefile'])
        try:
            _, _, checkpoint = replay_prefix(env, case)
            assert config_hash(checkpoint) == case['checkpoint_sha256']
            revision = review_open_head(case['tail'], checkpoint['observation'], checkpoint['admissible'])
            executed = None
            if revision and revision.next_action:
                obs, rewards, dones, infos = env.step(['<think></think><action>' + revision.next_action + '</action>'])
                executed = {'action': revision.next_action, 'feedback': obs['anchor'][0],
                            'won': bool(infos[0].get('won', False))}
                assert obs['anchor'][0] != 'Nothing happens.'
            rows.append({'case_id': case['id'], 'checkpoint_sha256': config_hash(checkpoint),
                         'review': asdict(revision) if revision else None, 'executed_next': executed})
        finally:
            env.close()
    report = {'paid_calls': 0, 'real_prefixes': len(rows),
        'scoped_open_contradictions': sum(r['review'] is not None for r in rows),
        'skipped_open_heads': sum(bool(r['review'] and r['review']['skipped']) for r in rows),
        'retained_navigation_executed': sum(r['executed_next'] is not None for r in rows),
        'rows': rows, 'scope': 'Narrow surface/open adapter; rejected lamp/other actions remain outside scope. '
                             'Historical development states, not independent model performance validation.'}
    save(ROOT / 'alfworld_guard_audit.json', report)
    print({k: v for k, v in report.items() if k != 'rows'})


if __name__ == '__main__':
    with patch.object(RushHourEnv, 'shortest_solution', denied), patch('dynamic_skill.puzzles.sudoku_solutions', denied):
        randomized()
        alfworld()
    save(ROOT / 'source_hashes.json', {p: digest(Path(p)) for p in [
        'dynamic_skill/queue_revision.py', 'dynamic_skill/queue_evidence.py',
        'dynamic_skill/alfworld_open_guard.py', 'experiments/verify_queue_revision.py',
        'tests/test_queue_revision_principles.py']})
