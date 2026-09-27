import json
import hashlib
from pathlib import Path

import pytest

from dynamic_skill.puzzles import RushHourEnv, SudokuEnv, PuzzleChecker, sudoku_solutions
from dynamic_skill.checker import CheckStatus
from experiments.puzzle_pilot import Settings, run_episode


def solved_board():
    return [[(r * 3 + r // 3 + c) % 9 + 1 for c in range(9)] for r in range(9)]


def test_sudoku_clues_conflicts_clear_and_unique_solution():
    puzzle = solved_board()
    puzzle[0][0] = puzzle[0][1] = 0
    env = SudokuEnv(puzzle)
    assert not env.step("set(2,2,9)").valid  # Given clue.
    assert not env.step("set(1,1,3)").valid  # Local conflict.
    assert env.step("set(1,1,1)").valid
    assert "1*" in env.render() and "3*" not in env.render()
    assert env.step("clear(1,1)").valid
    assert not env.step("clear(1,1)").valid
    assert env.step("set(1,1,1)").valid
    assert env.step("set(1,2,2)").won
    assert len(sudoku_solutions(puzzle)) == 1


def test_sudoku_accepts_locally_valid_value_without_hidden_answer():
    env = SudokuEnv([[0] * 9 for _ in range(9)])
    assert env.step("set(1,1,9)").valid
    assert not env.step("set(1,2,9)").valid


def test_rush_moves_distance_one_action_and_blocking():
    # B is vertical in col 3, initially blocking target A.
    env = RushHourEnv("..B..." "..B..." "AAB..." "......" "......" "......")
    assert not env.step("A+4").valid
    assert not env.step("B-1").valid
    assert env.shortest_solution() == ["B+3", "A+4"]
    assert env.step("B+3").valid
    assert env.step("A+4").won
    assert not env.step("A+1").valid


def test_rush_rejects_wall_jump_and_malformed_vehicle():
    env = RushHourEnv("......" "......" "AA.x.." "......" "......" "......")
    assert not env.step("A+4").valid
    assert env.shortest_solution() is None
    with pytest.raises(ValueError):
        RushHourEnv("B....." "......" "AA...." "......" "......" "......")


def test_checker_only_uses_visible_feedback():
    check = PuzzleChecker()
    assert check.check("A+1", "new", info={"valid": False, "reason": "blocked"}).should_interrupt
    assert check.check("A+1", "board", observation_history=["board"], info={"valid": True}).should_interrupt
    assert check.check("set(1,1,1)", "new", info={"valid": True}).status is CheckStatus.PROGRESS


@pytest.mark.parametrize("mode,steps", [("fixed_chunk", 4), ("dynamic_interrupt", 3)])
def test_puzzle_runner_uses_shared_queue_and_discards_tail(tmp_path, mode, steps):
    task = {"id": "rush", "benchmark": "rush_hour", "minimum_actions": 2,
            "board": "..B..." "..B..." "AAB..." "......" "......" "......"}
    plans = iter([["A+4", "A+1"], ["B+3", "A+4"]])

    def planner(prompt):
        assert "oracle" not in prompt
        return {"response": "<actions>" + json.dumps(next(plans)) + "</actions>",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}

    result = run_episode(task, mode, Settings(), tmp_path, planner)
    assert result["success"]
    assert result["model_calls"] == 2
    assert result["environment_steps"] == steps
    assert result["total_tokens"] == 30
    assert result["interrupt_count"] == int(mode == "dynamic_interrupt")


def test_react_one_action_per_call_and_format_errors_no_fake_actions(tmp_path):
    puzzle = solved_board()
    puzzle[0][0] = 0
    task = {"id": "sudoku", "benchmark": "sudoku", "puzzle": puzzle, "minimum_actions": 1}
    def planner(prompt):
        assert "1 to 1 actions" in prompt
        return {"response": '<actions>["set(1,1,1)"]</actions>', "usage": None}
    row = run_episode(task, "react", Settings(), tmp_path, planner)
    assert row["success"] and row["model_calls"] == row["environment_steps"] == 1
    assert row["estimated_cost_usd"] is None
    bad = run_episode(task, "fixed_chunk", Settings(), tmp_path, lambda _: {"response": "oops", "usage": None})
    assert bad["status"] == "format_error" and bad["environment_steps"] == 0


def test_api_failure_is_separate_and_logged(tmp_path):
    task = {"id": "rush", "benchmark": "rush_hour", "minimum_actions": 1,
            "board": "......" "......" "AA...." "......" "......" "......"}
    def planner(prompt):
        raise TimeoutError("private error details must not be saved")
    row = run_episode(task, "react", Settings(), tmp_path, planner)
    assert row["status"] == "api_error" and row["environment_steps"] == 0
    log = (tmp_path / "predictions/rush__react/model_calls.json").read_text()
    assert "private error" not in log


@pytest.mark.parametrize("mode", ["react", "fixed_chunk", "dynamic_interrupt"])
def test_long_dataset_replay_is_offline_only(tmp_path, monkeypatch, mode):
    """Scripted validation of full paths, NOT model-solving benchmark results."""
    directory = Path(__file__).resolve().parents[1] / "data/puzzle_pilot_20260927"
    tasks = json.loads((directory / "tasks.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    assert hashlib.sha256((directory / "tasks.json").read_bytes()).hexdigest() == manifest["tasks_sha256"]
    answers = {row["id"]: row for row in json.loads((directory / "validation_private.json").read_text())}

    def forbidden(*args, **kwargs):
        raise AssertionError("runtime must not query an offline solver")
    monkeypatch.setattr(RushHourEnv, "shortest_solution", forbidden)
    monkeypatch.setattr("dynamic_skill.puzzles.sudoku_solutions", forbidden)
    for task in tasks:
        answer = answers[task["id"]]
        if task["benchmark"] == "rush_hour":
            actions = answer["oracle_actions"]
        else:
            actions = [f"set({r+1},{c+1},{answer['solution'][r][c]})"
                       for r in range(9) for c in range(9) if task["puzzle"][r][c] == 0]
        k = 1 if mode == "react" else 4
        plans = iter([actions[i:i+k] for i in range(0, len(actions), k)])
        def scripted_planner(prompt):
            return {"response": "<actions>" + json.dumps(next(plans)) + "</actions>", "usage": None}
        row = run_episode(task, mode, Settings(), tmp_path, scripted_planner)
        assert row["success"] and row["environment_steps"] == task["minimum_actions"]
        assert row["model_calls"] == (len(actions) + k - 1) // k
        assert row["invalid_actions"] == row["interrupt_count"] == 0
