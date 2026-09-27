import json
import re

import pytest

from dynamic_skill import alfworld_rollout


class FakeEnvironment:
    def reset(self, options):
        return (
            {
                "text": ["Admissible actions: look"],
                "anchor": ["Your task is to: look at an object"],
            },
            [{"extra.gamefile": "/tmp/look_at_obj_in_light/game.tw-pddl"}],
        )

    def step(self, actions):
        assert "<action>look</action>" in actions[0]
        return (
            {"text": ["Done"], "anchor": ["Task completed successfully"]},
            [1.0],
            [True],
            [{"won": True}],
        )


def test_dynamic_rollout_preserves_skillopt_result_shape(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        alfworld_rollout,
        "chat_target",
        lambda **kwargs: ("<think>inspect</think><action>look</action>", {}),
    )
    results = alfworld_rollout.run_dynamic_alfworld_batch(
        FakeEnvironment(),
        "Use admissible actions.",
        max_steps=2,
        out_root=str(tmp_path),
        max_api_workers=1,
        result_ids=["sample-1"],
        mode="dynamic_interrupt",
    )
    assert results[0]["id"] == "sample-1"
    assert results[0]["hard"] == 1
    assert results[0]["controller_mode"] == "dynamic_interrupt"
    conversation = json.loads(
        (tmp_path / "predictions" / "sample-1" / "conversation.json").read_text()
    )
    assert conversation[0]["checker_status"] == "success"
    assert conversation[0]["controller_decision"] == "stop_success"
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["model_calls"] == 1
    assert metrics["model_errors"] == 0


class ScriptedEnvironment:
    """Independent episodes, including different lengths and early termination."""

    def __init__(self, scripts):
        self.scripts = scripts
        self.executed = [[] for _ in scripts]

    def reset(self, options):
        return ({"text": [f"episode={i}" for i in range(len(self.scripts))],
                 "anchor": ["Your task is to: inspect a bowl"] * len(self.scripts)},
                [{} for _ in self.scripts])

    def step(self, responses):
        anchors, rewards, dones = [], [], []
        for i, response in enumerate(responses):
            if response == "None":
                anchors.append("inactive")
                rewards.append(0)
                dones.append(False)
                continue
            action = re.fullmatch(r"<think>queued action</think><action>(.*?)</action>", response).group(1)
            expected, feedback, done = self.scripts[i][len(self.executed[i])]
            assert action == expected
            self.executed[i].append(action)
            anchors.append(feedback)
            rewards.append(int(done))
            dones.append(done)
        return ({"text": [f"episode={i}\n{a}" for i, a in enumerate(anchors)], "anchor": anchors},
                rewards, dones, [{"won": done} for done in dones])


def mock_plans(monkeypatch, plans):
    prompts = []

    def model(**kwargs):
        prompts.append(kwargs["user"])
        actions = plans.pop(0)
        return ("<actions>" + json.dumps(actions) + "</actions>",
                {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})

    monkeypatch.setattr(alfworld_rollout, "chat_target", model)
    return prompts


@pytest.mark.parametrize("mode", ["fixed_chunk", "dynamic_interrupt"])
def test_four_actions_need_one_call_and_usage_not_duplicated(monkeypatch, tmp_path, mode):
    mock_plans(monkeypatch, [["a", "b", "c", "d"]])
    env = ScriptedEnvironment([[ (a, f"state {a}", a == "d") for a in "abcd" ]])
    result = alfworld_rollout.run_dynamic_alfworld_batch(env, "", mode=mode, out_root=str(tmp_path))[0]
    assert result["n_turns"] == 4
    assert result["model_calls"] == 1
    assert result["total_tokens"] == 15
    assert result["interrupt_count"] == 0
    trace = json.loads((tmp_path / "predictions/env_000/conversation.json").read_text())
    assert [row["model_called"] for row in trace] == [True, False, False, False]
    assert [row["chunk_action_index"] for row in trace] == [0, 1, 2, 3]
    assert sum(row["model_latency_seconds"] for row in trace) == trace[0]["model_latency_seconds"]


@pytest.mark.parametrize("mode", ["fixed_chunk", "dynamic_interrupt"])
def test_rejection_discards_tail_only_in_dynamic_mode(monkeypatch, tmp_path, mode):
    dynamic = mode == "dynamic_interrupt"
    plans = [["bad", "old_tail"]] + ([["recover", "finish"]] if dynamic else [])
    prompts = mock_plans(monkeypatch, plans)
    script = [("bad", "Nothing happens.", False)]
    script += [("recover", "You move to desk.", False), ("finish", "Done", True)] if dynamic else [("old_tail", "Done", True)]
    result = alfworld_rollout.run_dynamic_alfworld_batch(
        ScriptedEnvironment([script]), "", mode=mode, out_root=str(tmp_path))[0]
    assert result["model_calls"] == (2 if dynamic else 1)
    assert result["interrupt_count"] == int(dynamic)
    assert result["discarded_actions"] == int(dynamic)
    if dynamic:
        assert "environment rejected action" in prompts[1]
    trace = json.loads((tmp_path / "predictions/env_000/conversation.json").read_text())
    assert trace[0]["discarded_actions"] == (["old_tail"] if dynamic else [])


def test_terminal_stops_before_tail_and_normal_boundary_is_not_interrupt(monkeypatch):
    mock_plans(monkeypatch, [["a", "b"], ["c", "unused"]])
    env = ScriptedEnvironment([[ (a, f"state {a}", a == "c") for a in "abc" ]])
    result = alfworld_rollout.run_dynamic_alfworld_batch(env, "")[0]
    assert result["n_turns"] == 3
    assert result["chunk_boundaries"] == 1
    assert result["interrupt_count"] == 0
    assert result["discarded_actions"] == 1


def test_chunk_one_and_step_limit(monkeypatch):
    mock_plans(monkeypatch, [["a"], ["b"]])
    env = ScriptedEnvironment([[ (a, f"state {a}", False) for a in "ab" ]])
    result = alfworld_rollout.run_dynamic_alfworld_batch(env, "", fixed_chunk_size=1, max_steps=2)[0]
    assert result["n_turns"] == result["model_calls"] == 2
    assert result["hard"] == 0
    assert "Timeout" in result["fail_reason"]


def test_multiple_environments_keep_independent_queues(monkeypatch):
    counts = {0: 0, 1: 0}

    def model(**kwargs):
        i = int(re.search(r"episode=(\d)", kwargs["user"]).group(1))
        actions = [["a", "b"]][counts[i]] if i == 0 else [["x"], ["y", "z"]][counts[i]]
        counts[i] += 1
        return "<actions>" + json.dumps(actions) + "</actions>", {}

    monkeypatch.setattr(alfworld_rollout, "chat_target", model)
    env = ScriptedEnvironment([[("a", "s1", False), ("b", "done", True)],
                               [("x", "s2", False), ("y", "s3", False), ("z", "done", True)]])
    rows = alfworld_rollout.run_dynamic_alfworld_batch(env, "", max_api_workers=2)
    assert [row["model_calls"] for row in rows] == [1, 2]
    assert [row["n_turns"] for row in rows] == [2, 3]
    assert counts == {0: 1, 1: 2}


@pytest.mark.parametrize("response", ['<actions>[]</actions>', '<actions>["a", "b", "c"]</actions>',
                                     '<actions>[1]</actions>', '<actions>["a\\nb"]</actions>',
                                     '<actions>not-json</actions>', ''])
def test_malformed_queues_are_rejected(response):
    with pytest.raises(ValueError):
        alfworld_rollout.parse_action_chunk(response, 2)


def test_parse_and_api_errors_are_visible(monkeypatch, tmp_path):
    outputs = iter(["bad json", RuntimeError("private error")])

    def model(**kwargs):
        value = next(outputs)
        if isinstance(value, Exception):
            raise value
        return value, {"total_tokens": 7}

    monkeypatch.setattr(alfworld_rollout, "chat_target", model)
    env = ScriptedEnvironment([[("look", "s1", False), ("look", "s2", False)]])
    row = alfworld_rollout.run_dynamic_alfworld_batch(env, "", max_steps=2, out_root=str(tmp_path))[0]
    assert row["parse_errors"] == row["model_errors"] == 1
    assert row["model_calls"] == 2
    assert row["total_tokens"] == 7
    assert not row["agent_ok"]


def test_interrupt_budget_stops_rollout(monkeypatch):
    from dynamic_skill.interrupter import Interrupter
    from dynamic_skill.skill_executor import ChunkExecutor
    monkeypatch.setattr(alfworld_rollout, "ChunkExecutor",
                        lambda **kwargs: ChunkExecutor(**kwargs, interrupter=Interrupter(max_interrupts=1)))
    mock_plans(monkeypatch, [["bad", "tail"], ["bad", "tail"]])
    env = ScriptedEnvironment([[ ("bad", "Nothing happens.", False) ] * 2])
    row = alfworld_rollout.run_dynamic_alfworld_batch(env, "", max_steps=20)[0]
    assert row["n_turns"] == 2
    assert row["fail_reason"] == "interrupt budget exhausted"


def test_reasoning_examples_do_not_override_action_queue():
    response = '<think>Example: <actions>["wrong"]</actions></think><actions>["right"]</actions>'
    assert alfworld_rollout.parse_action_chunk(response, 4) == ["right"]


def test_entrypoint_aggregates_sequential_batches(monkeypatch, tmp_path):
    import sys
    from experiments import _common
    from scripts import eval_only
    from skillopt.envs.alfworld import adapter

    original_batch = adapter.run_alfworld_batch
    # Ensure pytest restores the runtime adapter patch made by the entrypoint.
    monkeypatch.setattr(adapter, "run_alfworld_batch", original_batch)
    monkeypatch.setattr(sys, "argv", ["fixed_chunk.py", "--chunk-size", "4"])
    monkeypatch.chdir(tmp_path)
    mock_plans(monkeypatch, [["a", "b"], ["c", "d"]])

    def evaluate():
        for i, actions in enumerate(("ab", "cd")):
            env = ScriptedEnvironment([[(a, f"state {a}", j == 1) for j, a in enumerate(actions)]])
            adapter.run_alfworld_batch(env_manager=env, skill_content="", out_root=str(tmp_path), result_ids=[str(i)])

    monkeypatch.setattr(eval_only, "main", evaluate)
    _common.run("fixed_chunk")
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["episodes"] == 2
    assert metrics["environment_steps"] == 4
    assert metrics["model_calls"] == 2
    assert metrics["total_tokens"] == 30
