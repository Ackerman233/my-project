from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time
from typing import Any


def _rate(name: str) -> float | None:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else None


def run_instrumented_react_batch(*args, **kwargs) -> list[dict]:
    """Run the untouched SkillOpt React rollout while recording API metrics."""
    from skillopt.envs.alfworld import rollout as official

    original_chat_target = official.chat_target
    lock = threading.Lock()
    metrics: dict[str, Any] = {
        "model_calls": 0,
        "model_errors": 0,
        "latencies": [],
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    def measured_chat_target(*chat_args, **chat_kwargs):
        started = time.perf_counter()
        try:
            response, usage = original_chat_target(*chat_args, **chat_kwargs)
        except Exception:
            with lock:
                metrics["model_calls"] += 1
                metrics["model_errors"] += 1
                metrics["latencies"].append(time.perf_counter() - started)
            raise
        normalized = {
            "prompt_tokens": int((usage or {}).get("prompt_tokens", 0) or 0),
            "completion_tokens": int((usage or {}).get("completion_tokens", 0) or 0),
            "total_tokens": int((usage or {}).get("total_tokens", 0) or 0),
        }
        with lock:
            metrics["model_calls"] += 1
            metrics["latencies"].append(time.perf_counter() - started)
            for key, value in normalized.items():
                metrics[key] += value
        return response, usage

    rollout_started = time.perf_counter()
    official.chat_target = measured_chat_target
    try:
        results = official.run_alfworld_batch(*args, **kwargs)
    finally:
        official.chat_target = original_chat_target

    latencies = list(metrics.pop("latencies"))
    aggregate = {
        "controller_mode": "react",
        "episodes": len(results),
        **metrics,
        "model_latency_total_seconds": round(sum(latencies), 4),
        "model_latency_mean_seconds": round(sum(latencies) / len(latencies), 4) if latencies else 0.0,
        "rollout_wall_seconds": round(time.perf_counter() - rollout_started, 4),
    }

    input_rate = _rate("MODEL_INPUT_USD_PER_MILLION")
    output_rate = _rate("MODEL_OUTPUT_USD_PER_MILLION")
    if input_rate is not None and output_rate is not None:
        input_cost = aggregate["prompt_tokens"] * input_rate / 1_000_000
        output_cost = aggregate["completion_tokens"] * output_rate / 1_000_000
        aggregate.update(
            {
                "input_cost_usd": round(input_cost, 10),
                "output_cost_usd": round(output_cost, 10),
                "estimated_cost_usd": round(input_cost + output_cost, 10),
                "pricing": {
                    "input_usd_per_million_tokens": input_rate,
                    "output_usd_per_million_tokens": output_rate,
                    "source": os.environ.get("MODEL_PRICE_SOURCE", ""),
                },
            }
        )
    else:
        aggregate["estimated_cost_usd"] = None

    out_root = str(kwargs.get("out_root") or "")
    if out_root:
        path = Path(out_root) / "metrics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return results

