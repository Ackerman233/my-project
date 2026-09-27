"""Opt-in real TextWorld/ALFWorld execution; scripted planner, no API charges.

ALFWORLD_DATA=/path/to/alfworld RUN_ALFWORLD_INTEGRATION=1 pytest -q tests/test_alfworld_integration.py
This verifies integration, not model quality or benchmark performance.
"""
import json
import os
from pathlib import Path

import pytest

from dynamic_skill import alfworld_rollout


@pytest.mark.skipif(os.environ.get("RUN_ALFWORLD_INTEGRATION") != "1", reason="requires ALFWorld game payload")
@pytest.mark.parametrize("mode", ["fixed_chunk", "dynamic_interrupt"])
def test_real_environment_queue_and_recovery(monkeypatch, tmp_path, mode):
    from skillopt.envs.alfworld.rollout import build_alfworld_env

    game = Path(os.environ["ALFWORLD_DATA"]) / (
        "json_2.1.1/valid_seen/look_at_obj_in_light-AlarmClock-None-DeskLamp-323/"
        "trial_T20190909_044715_250790/game.tw-pddl")
    assert game.is_file()
    plans = iter([["go to nowhere 999", "look"],
                  ["go to desk 1", "take alarmclock 2 from desk 1", "use desklamp 1"]])

    def scripted_planner(**kwargs):
        return "<actions>" + json.dumps(next(plans)) + "</actions>", {}

    monkeypatch.setattr(alfworld_rollout, "chat_target", scripted_planner)
    env = build_alfworld_env(env_num=1, eval_dataset="eval_in_distribution", seed=42,
                             is_train=False, specific_gamefiles=[str(game)])
    try:
        row = alfworld_rollout.run_dynamic_alfworld_batch(
            env, "", max_steps=8, mode=mode, out_root=str(tmp_path))[0]
    finally:
        env.close()
    assert row["hard"] == 1
    assert row["model_calls"] == 2
    assert row["model_errors"] == row["parse_errors"] == 0
    assert row["n_turns"] == (4 if mode == "dynamic_interrupt" else 5)
    assert row["interrupt_count"] == int(mode == "dynamic_interrupt")
    assert row["multi_action_chunks"] == 2
    trace = json.loads((tmp_path / "predictions/env_000/conversation.json").read_text())
    assert trace[0]["discarded_actions"] == (["look"] if mode == "dynamic_interrupt" else [])
