"""Select reproducible pilot tasks; verify difficulty offline, save solutions separately."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))
from dynamic_skill.puzzles import SudokuEnv, RushHourEnv, sudoku_solutions


def prepare(source, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite dataset: {output}")
    tasks, validation = [], []
    for row in map(json.loads, source.read_text().splitlines()):
        puzzle = row["puzzle"]
        blanks = sum(v == 0 for line in puzzle for v in line)
        if row.get("split") != "test" or not 41 <= blanks <= 45:
            continue
        SudokuEnv(puzzle)  # Validate clue consistency before counting solutions.
        solutions = sudoku_solutions(puzzle, limit=2)
        if len(solutions) != 1:
            continue
        task_id = f"sudoku-{row['id']}"
        tasks.append({"id": task_id, "benchmark": "sudoku", "puzzle": puzzle,
                      "minimum_actions": blanks, "horizon_definition": "one initially empty cell filled per set action",
                      "source": str(source.resolve()), "source_id": row["id"], "unique_solution_verified": True})
        validation.append({"id": task_id, "solution": solutions[0]})
        if len(tasks) == 2:
            break
    if len(tasks) != 2:
        raise RuntimeError("need two unique long Sudoku instances")
    # Public database examples, two separate vehicle configurations/clusters.
    seeds = [("GBBoLoGHIoLMGHIAAMCCCKoMooJKDDEEJFFo", 51, 20),
             ("ooIBBBooIKooAAJKoLCCJDDLGHEEoLGHFFoo", 50, 25)]
    for index, (board, advertised, horizon) in enumerate(seeds):
        env = RushHourEnv(board)
        solution = env.shortest_solution()
        if solution is None or len(solution) != advertised:
            raise AssertionError("BFS disagrees with source database move count")
        for action in solution[:-horizon]:
            assert env.step(action).valid
        derived_board = env.board_string()
        remaining = env.shortest_solution()
        assert len(remaining) == horizon
        task_id = f"rush-hour-{horizon}-{index}"
        tasks.append({"id": task_id, "benchmark": "rush_hour", "board": derived_board,
                      "minimum_actions": horizon, "horizon_definition": "one legal vehicle slide of any distance",
                      "source": "https://www.michaelfogleman.com/rush/", "source_board": board,
                      "source_minimum_moves": advertised,
                      "derivation": f"state after {len(solution)-horizon} moves along an offline shortest path"})
        validation.append({"id": task_id, "oracle_actions": remaining, "verified_shortest_moves": horizon})
    output.mkdir(parents=True)
    dataset = output / "tasks.json"
    dataset.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n")
    (output / "validation_private.json").write_text(json.dumps(validation, indent=2) + "\n")
    manifest = {"source_sudoku_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "tasks_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                "note": "Pilot only: local synthetic Sudoku, derived public Rush Hour. Not official paper benchmark replication.",
                "validation_answers_visible_to_agent": False, "tasks": [{k: t[k] for k in ("id", "benchmark", "minimum_actions")} for t in tasks]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sudoku-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.sudoku_source, args.out)
