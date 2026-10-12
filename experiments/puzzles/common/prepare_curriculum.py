"""Tiny easy-to-medium sanity curriculum, with strictly offline validation."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))
from dynamic_skill.puzzles import SudokuEnv, RushHourEnv, sudoku_solutions


def prepare(source, out):
    if out.exists():
        raise FileExistsError('never overwrite an existing dataset')
    originals = json.loads((source / 'tasks.json').read_text())
    private = {x['id']: x for x in json.loads((source / 'validation_private.json').read_text())}
    tasks, answers = [], []
    for base, task in enumerate(t for t in originals if t['benchmark'] == 'sudoku'):
        solution = private[task['id']]['solution']
        coordinates = list(range(81))
        random.Random(20260927 + base).shuffle(coordinates)
        for rung, holes in enumerate((1, 3, 6, 12)):
            board = [row[:] for row in solution]
            for cell in coordinates[:holes]:
                board[cell // 9][cell % 9] = 0
            SudokuEnv(board)
            solved = sudoku_solutions(board, limit=2)
            assert len(solved) == 1 and solved[0] == solution
            identity = f'easy-sudoku-{holes}-{base}'
            tasks.append({'id': identity, 'benchmark': 'sudoku', 'puzzle': board,
                          'minimum_actions': holes, 'rung': rung, 'split': 'dev' if base == 0 else 'validation',
                          'source_id': task['id'], 'note': 'Synthetic mask of verified solution; sanity test, not independent benchmark.'})
            answers.append({'id': identity, 'solution': solution})
    for base, task in enumerate(t for t in originals if t['benchmark'] == 'rush_hour'):
        initial = RushHourEnv(task['board'])
        full = initial.shortest_solution()
        for rung, length in enumerate((1, 2, 4, 8)):
            env = RushHourEnv(task['board'])
            for action in full[:-length]:
                assert env.step(action).valid
            remaining = env.shortest_solution()
            assert len(remaining) == length
            identity = f'easy-rush-{length}-{base}'
            tasks.append({'id': identity, 'benchmark': 'rush_hour', 'board': env.board_string(),
                          'minimum_actions': length, 'rung': rung, 'split': 'dev' if base == 0 else 'validation',
                          'source_id': task['id'], 'note': 'Derived intermediate state; shared graph across rungs.'})
            answers.append({'id': identity, 'oracle_actions': remaining})
    out.mkdir(parents=True)
    path = out / 'tasks.json'
    path.write_text(json.dumps(tasks, indent=2) + '\n')
    (out / 'validation_private.json').write_text(json.dumps(answers, indent=2) + '\n')
    (out / 'manifest.json').write_text(json.dumps({'tasks_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'seed': 20260927, 'task_count': len(tasks), 'answers_visible_to_runtime': False,
        'source_tasks_sha256': hashlib.sha256((source / 'tasks.json').read_bytes()).hexdigest()}, indent=2) + '\n')
    print(f'verified {len(tasks)} tasks; answers are offline only')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('data/puzzle_pilot_20260927'))
    parser.add_argument('--out', type=Path, default=Path('data/curriculum_20260927'))
    args = parser.parse_args()
    prepare(args.source, args.out)
