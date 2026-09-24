import json

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

