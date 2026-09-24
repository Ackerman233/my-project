import json

from dynamic_skill.react_metrics import run_instrumented_react_batch


def test_react_metrics_wrap_official_rollout(monkeypatch, tmp_path) -> None:
    from skillopt.envs.alfworld import rollout as official

    monkeypatch.setenv("MODEL_INPUT_USD_PER_MILLION", "1")
    monkeypatch.setenv("MODEL_OUTPUT_USD_PER_MILLION", "2")
    monkeypatch.setattr(
        official,
        "chat_target",
        lambda *args, **kwargs: (
            "<think>ok</think><action>look</action>",
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        ),
    )

    def fake_rollout(*args, **kwargs):
        official.chat_target(system="s", user="u")
        return [{"id": "one", "hard": 1}]

    monkeypatch.setattr(official, "run_alfworld_batch", fake_rollout)
    results = run_instrumented_react_batch(out_root=str(tmp_path))
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert results[0]["hard"] == 1
    assert metrics["model_calls"] == 1
    assert metrics["total_tokens"] == 15
    assert metrics["estimated_cost_usd"] == 0.00002

