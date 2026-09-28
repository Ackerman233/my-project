"""One user-authorized small run, 2026-09-28; offline dataset preparation only."""
import fcntl
import json
from pathlib import Path
import random
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dynamic_skill.puzzles import RushHourEnv, SudokuEnv, sudoku_solutions
from dynamic_skill.evolution import validate_splits, digest
from dynamic_skill.campaign_budget import CampaignBudget
from experiments.puzzle_pilot import save


def rush(seed, count, depth):
    rng = random.Random(seed)
    for attempt in range(20000):
        board = ['.'] * 36
        board[12:14] = 'AA'
        for index in range(count):
            letter = chr(ord('B') + index)
            options = []
            for length in (2, 3):
                for vertical in (False, True):
                    for row in range(6):
                        for col in range(6):
                            if row + (length - 1 if vertical else 0) >= 6 or col + (0 if vertical else length - 1) >= 6:
                                continue
                            cells = [(row + (j if vertical else 0)) * 6 + col + (0 if vertical else j) for j in range(length)]
                            if all(board[i] == '.' for i in cells):
                                options.append(cells)
            if not options:
                break
            for cell in rng.choice(options):
                board[cell] = letter
        if len(set(board) - {'.'}) != count + 1:
            continue
        env = RushHourEnv(''.join(board))
        try:
            solution = env.shortest_solution(max_states=20000)
        except RuntimeError:
            continue
        if solution and len(solution) == depth:
            return ''.join(board), solution, attempt
    raise RuntimeError('no suitable bounded puzzle generated')


def sudoku(seed, holes=3):
    rng = random.Random(seed)
    board = [[0] * 9 for _ in range(9)]
    def fill(index=0):
        if index == 81:
            return True
        row, col = divmod(index, 9)
        values = list(range(1, 10)); rng.shuffle(values)
        used = set(board[row]) | {board[r][col] for r in range(9)} | {board[r][c] for r in range(row//3*3, row//3*3+3) for c in range(col//3*3,col//3*3+3)}
        for value in values:
            if value in used:
                continue
            board[row][col] = value
            if fill(index+1):
                return True
        board[row][col] = 0
        return False
    assert fill()
    solution = [r[:] for r in board]
    for cell in rng.sample(range(81), holes):
        board[cell//9][cell%9] = 0
    assert len(sudoku_solutions(board, limit=2)) == 1
    SudokuEnv(board)
    return board, solution


def main():
    root = Path('data/evolution_small_20260928')
    if root.exists():
        raise SystemExit('dataset already exists; refusing overwrite/reselection')
    splits = {'train': [], 'validation': [], 'test': []}
    private = []
    for split, seed in [('train', 2026092801), ('validation', 2026092802), ('test', 2026092803)]:
        board, solution = sudoku(seed)
        identity = f'evo-{split}-sudoku'
        splits[split].append({'id': identity, 'benchmark': 'sudoku', 'puzzle': board,
            'minimum_actions': 3, 'family_id': f'synthetic-sudoku-{seed}'})
        private.append({'id': identity, 'solution': solution})
    for split, count, depth, seed in [('train', 8, 4, 2026092804), ('train', 10, 6, 2026092805),
                                      ('validation', 7, 3, 2026092806), ('test', 9, 3, 2026092807)]:
        board, solution, attempt = rush(seed, count, depth)
        identity = f'evo-{split}-rush-{depth}'
        splits[split].append({'id': identity, 'benchmark': 'rush_hour', 'board': board,
            'minimum_actions': depth, 'family_id': f'synthetic-rush-{count+1}-vehicles', 'seed': seed})
        private.append({'id': identity, 'oracle_actions': solution, 'attempt': attempt})
    validate_splits(splits)
    save(root/'splits.json', splits)
    save(root/'tasks.json', [t for tasks in splits.values() for t in tasks])
    save(root/'validation_private.json', private)
    save(root/'manifest.json', {'protocol': 'small-evolution-v1', 'split_hash': digest(splits),
        'counts': {k: len(v) for k,v in splits.items()}, 'answers_visible_to_model': False,
        'note': 'New synthetic instances, not a representative benchmark. Distinct Rush vehicle counts; Sudoku isomorphism not exhaustively checked.'})
    path = Path('results/gpt54_recovery_20260928/budget.json')
    with (path.parent/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        budget = CampaignBudget(path, deadline=0)
        data = budget.read()
        if data.get('small_evolution_authorization') or any(r['status']=='pending' for r in data['requests']):
            raise RuntimeError('authorization already used or pending calls')
        if data['policy_violation'] or time.time() < data['cooldown_until']:
            raise RuntimeError('policy/cooldown guard remains closed')
        data['small_evolution_authorization'] = {'at_unix': time.time(),
            'reason': 'User explicitly requested one small evolution loop; existing $5 total preserved.',
            'old_deadline': data['deadline_unix'], 'old_max_requests': data['max_requests'],
            'previous_requests': len(data['requests']), 'additional_cap': 79,
            'previous_charge': sum(r['budget_charge_usd'] for r in data['requests']),
            'split_hash': digest(splits)}
        data['max_requests'] = len(data['requests']) + 79
        data['deadline_unix'] = time.time() + 1800
        budget.write(data)
        print(json.dumps({'counts': {k:len(v) for k,v in splits.items()},
            'existing_requests': len(data['requests']),
            'existing_charge': sum(r['budget_charge_usd'] for r in data['requests']),
            'request_cap': data['max_requests'], 'deadline_unix': data['deadline_unix']}))


if __name__ == '__main__':
    main()
