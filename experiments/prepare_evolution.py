"""Explicit split materialization; never randomly re-label seen tasks as held-out."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dynamic_skill.evolution import validate_splits
from experiments.puzzle_pilot import save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--train', nargs='+', required=True)
    parser.add_argument('--validation', nargs='+', required=True)
    parser.add_argument('--test', nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    tasks = json.loads(args.tasks.read_text(encoding='utf-8'))
    by_id = {t['id']: t for t in tasks}
    if len(by_id) != len(tasks):
        parser.error('duplicate IDs in source')
    try:
        splits = {name: [by_id[i] for i in getattr(args, name)] for name in ('train', 'validation', 'test')}
        validate_splits(splits)
    except (KeyError, ValueError) as exc:
        parser.error(str(exc))
    if args.out.exists():
        parser.error('refuse to overwrite existing split protocol')
    save(args.out, splits)
    print('Split manifest written. You must separately verify that test families were not used during prior development.')


if __name__ == '__main__':
    main()
