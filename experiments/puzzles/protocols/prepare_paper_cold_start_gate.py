"""Generate once before paid calls; offline answers never enter the planner."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))
from dynamic_skill.puzzles import make_environment
from dynamic_skill.evolution import digest
from experiments.prepare_small_evolution_run import rush, sudoku
from experiments.puzzle_pilot import save
from experiments.paper_cold_start_gate import NAME


def main():
    root = Path('data') / NAME
    if root.exists():
        raise RuntimeError('dataset already frozen')
    prior = set()
    for path in Path('data').glob('*/tasks.json'):
        rows = json.loads(path.read_text())
        if isinstance(rows, list):
            for t in rows:
                if isinstance(t, dict) and t.get('benchmark') in ('rush_hour', 'sudoku'):
                    prior.add(make_environment(t).render())
    tasks, private = [], []
    for depth in (1, 2, 3):
        puzzle, answer = sudoku(2026092890 + depth, holes=depth)
        task = {'id': f'paper-sudoku-{depth}', 'benchmark': 'sudoku', 'puzzle': puzzle, 'minimum_actions': depth}
        tasks.append(task); private.append({'id': task['id'], 'answer': answer})
        board, answer, attempt = rush(2026092900 + depth, 4 + depth, depth)
        task = {'id': f'paper-rush-{depth}', 'benchmark': 'rush_hour', 'board': board, 'minimum_actions': depth}
        tasks.append(task); private.append({'id': task['id'], 'answer': answer, 'attempt': attempt})
    states = [make_environment(t).render() for t in tasks]
    assert len(set(states)) == 6 and not set(states) & prior
    save(root / 'tasks.json', tasks)
    save(root / 'validation_private.json', private)
    save(root / 'manifest.json', {'tasks_hash': digest(tasks), 'count': 6,
                                'seeds': '2026092891..93/2901..03', 'prior_states_checked': len(prior),
                                'purpose': 'Frozen basic-capability confirmation set, never prompt-tuned on these results'})
    print('Frozen six new tasks, distinct from existing dataset states.')


if __name__ == '__main__':
    main()
