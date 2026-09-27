import json

import httpx
import pytest
from openai import OpenAI
from openai.resources.chat.completions import Completions

from dynamic_skill.thinking_policy import non_thinking_body, non_thinking_requests
from experiments.puzzle_pilot import ApiPlanner, Settings


def payload():
    return {"id": "test-id", "object": "chat.completion", "created": 0, "model": "qwen3.5-35b-a3b",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "OK"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
                      "completion_tokens_details": {"reasoning_tokens": 0}}}


def test_body_copies_and_overrides_conflicting_switches():
    old = {"enable_thinking": True, "chat_template_kwargs": {"enable_thinking": True, "other": 3}, "other": 4}
    new = non_thinking_body(old)
    assert new["enable_thinking"] is False and new["chat_template_kwargs"]["enable_thinking"] is False
    assert old["enable_thinking"] is True and old["chat_template_kwargs"]["enable_thinking"] is True
    assert new["other"] == 4 and new["chat_template_kwargs"]["other"] == 3


def test_puzzle_sends_false_in_actual_http_json(monkeypatch):
    bodies = []
    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=payload())
    monkeypatch.setenv("TARGET_AZURE_OPENAI_API_KEY", "test-not-a-real-key")
    planner = ApiPlanner(Settings())
    planner.client.close()
    planner.client = OpenAI(base_url="https://test.invalid/v1", api_key="test-only",
                            http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    try:
        assert planner("test")["response"] == "OK"
    finally:
        planner.close()
    assert len(bodies) == 1 and bodies[0]["enable_thinking"] is False
    assert "extra_body" not in bodies[0]  # SDK merges into top-level HTTP JSON.


def test_shared_skillopt_sdk_path_and_audit_restore(tmp_path):
    bodies = []
    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=payload())
    original = Completions.create
    audit = tmp_path / "thinking_policy.jsonl"
    with OpenAI(base_url="https://test.invalid/v1", api_key="test-only",
                http_client=httpx.Client(transport=httpx.MockTransport(handler))) as client:
        with non_thinking_requests(audit):
            client.chat.completions.create(model="qwen", messages=[{"role": "user", "content": "test"}],
                                           extra_body={"enable_thinking": True})
    assert Completions.create is original
    assert bodies[0]["enable_thinking"] is False
    rows = [json.loads(line) for line in audit.read_text().splitlines()]
    assert [r["event"] for r in rows] == ["request", "response"]
    assert rows[1]["thinking_verification"] == "zero_reported"
    assert "test-only" not in audit.read_text()


def test_raw_qwen_also_overrides_and_reports_ignored_switch(monkeypatch, tmp_path):
    from skillopt.model import qwen_backend
    bodies = []
    def fake_post(body, timeout, config):
        bodies.append(body)
        response = payload()
        response["choices"][0]["message"]["reasoning_content"] = "still thinking"
        return response
    monkeypatch.setattr(qwen_backend, "_post_chat_completion", fake_post)
    audit = tmp_path / "policy.jsonl"
    with pytest.raises(RuntimeError), non_thinking_requests(audit):
        qwen_backend._post_chat_completion({"model": "qwen", "chat_template_kwargs": {"enable_thinking": True}}, 3, None)
        raise RuntimeError("test restore on error")
    assert qwen_backend._post_chat_completion is fake_post
    assert bodies[0]["enable_thinking"] is False
    assert bodies[0]["chat_template_kwargs"]["enable_thinking"] is False
    assert "thinking_returned" in audit.read_text()


def test_real_skillopt_request_builder_has_false_and_no_conflicting_effort():
    from skillopt.model import azure_openai
    bodies = []
    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=payload())
    with OpenAI(base_url="https://test.invalid/v1", api_key="test-only",
                http_client=httpx.Client(transport=httpx.MockTransport(handler))) as client:
        with non_thinking_requests():
            text, usage = azure_openai._chat_impl(client, "qwen3.5-35b-a3b", "system", "user",
                                                  32, 1, "policy_test", reasoning_effort="high")
    assert text == "OK" and usage["total_tokens"] == 2
    assert bodies[0]["enable_thinking"] is False
    assert "reasoning_effort" not in bodies[0]
