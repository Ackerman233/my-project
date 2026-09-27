"""Project-only, explicit non-thinking wire policy. Never edits upstream files."""
from contextlib import contextmanager
from functools import wraps
import json
from pathlib import Path
import threading
from unittest.mock import patch


def non_thinking_body(body=None):
    """Copy caller options, overriding both supported Qwen switch locations."""
    result = dict(body or {})
    result["enable_thinking"] = False
    result.pop("reasoning_effort", None)  # Do not send a conflicting reasoning request.
    if "chat_template_kwargs" in result:
        result["chat_template_kwargs"] = {**result["chat_template_kwargs"], "enable_thinking": False}
    return result


@contextmanager
def non_thinking_requests(audit_path=None):
    """Cover SkillOpt SDK and raw-Qwen paths for one experiment invocation.

    Install once outside worker threads; restore all hooks even if eval fails.
    Only request/response metadata is audited, never keys, headers or prompts.
    """
    from openai.resources.chat.completions import Completions
    from skillopt.model import qwen_backend

    sdk_create = Completions.create
    qwen_post = qwen_backend._post_chat_completion
    lock = threading.Lock()

    def audit(row):
        if audit_path is not None:
            path = Path(audit_path)
            with lock:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")

    def response_metadata(response):
        raw = response if isinstance(response, dict) else response.model_dump()
        choice = (raw.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = raw.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        reasoning_tokens = details.get("reasoning_tokens")
        reasoning_chars = len(message.get("reasoning_content") or "")
        return {"response_id": raw.get("id"), "reasoning_tokens": reasoning_tokens,
                "reasoning_content_chars": reasoning_chars,
                "thinking_verification": "thinking_returned" if reasoning_chars or (reasoning_tokens or 0) > 0
                    else "zero_reported" if reasoning_tokens == 0 else "not_reported"}

    @wraps(sdk_create)
    def create(resource, *args, **kwargs):
        kwargs.pop("reasoning_effort", None)
        kwargs["extra_body"] = non_thinking_body(kwargs.get("extra_body"))
        row = {"transport": "sdk", "model": kwargs.get("model"), "enable_thinking": False}
        audit({**row, "event": "request"})
        response = sdk_create(resource, *args, **kwargs)
        if hasattr(response, "model_dump"):
            audit({**row, "event": "response", **response_metadata(response)})
        return response

    @wraps(qwen_post)
    def post(payload, *args, **kwargs):
        payload = non_thinking_body(payload)
        row = {"transport": "qwen_raw", "model": payload.get("model"), "enable_thinking": False}
        audit({**row, "event": "request"})
        response = qwen_post(payload, *args, **kwargs)
        audit({**row, "event": "response", **response_metadata(response)})
        return response

    with patch.object(Completions, "create", create), patch.object(qwen_backend, "_post_chat_completion", post):
        yield
