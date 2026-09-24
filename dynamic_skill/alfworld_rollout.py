from __future__ import annotations

import concurrent.futures
import json
import os
import re
import time
from typing import Any

from skillopt.model import chat_target

from .failure_analyzer import FailureAnalyzer
from .skill_executor import ChunkExecutor
from .skill_updater import SkillUpdater


def _extract(response: str, tag: str) -> str | None:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", response or "", re.DOTALL)
    return match.group(1).strip() if match else None


def _task_type(gamefile: str) -> str:
    names = (
        "pick_and_place",
        "pick_two_obj_and_place",
        "look_at_obj_in_light",
        "pick_heat_then_place_in_recep",
        "pick_cool_then_place_in_recep",
        "pick_clean_then_place_in_recep",
    )
    return next((name for name in names if name in gamefile), "other")


def run_dynamic_alfworld_batch(
    env_manager,
    skill_content: str,
    max_steps: int = 50,
    out_root: str = "",
    max_api_workers: int = 8,
    temperature: float = 0.4,
    max_completion_tokens: int = 16384,
    diagnostic_mode: bool = False,
    diagnostic_instruction: str = "",
    result_ids: list[str] | None = None,
    *,
    mode: str = "dynamic_interrupt",
    fixed_chunk_size: int = 4,
) -> list[dict]:
    """Drop-in replacement for SkillOpt's batch rollout.

    It preserves the official environment manager, model backend, output shape,
    and scoring path. Only prompt construction and trajectory bookkeeping are
    extended with a controller-side chunk executor.
    """
    if mode not in {"fixed_chunk", "dynamic_interrupt", "dynamic_evolution"}:
        raise ValueError(f"unsupported dynamic mode: {mode}")
    dynamic = mode != "fixed_chunk"

    obs, infos = env_manager.reset({})
    env_num = len(obs["text"])
    env_dones = [False] * env_num
    successes = [False] * env_num
    conversations: list[list[dict[str, Any]]] = [[] for _ in range(env_num)]
    model_metrics: list[dict[str, Any]] = [
        {
            "calls": 0,
            "latencies": [],
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "errors": 0,
        }
        for _ in range(env_num)
    ]
    controllers = [ChunkExecutor(chunk_size=fixed_chunk_size, dynamic=dynamic) for _ in range(env_num)]
    metadata: list[dict[str, str]] = []
    rollout_started = time.perf_counter()

    for index in range(env_num):
        info = infos[index] if isinstance(infos[index], dict) else {}
        gamefile = str(info.get("extra.gamefile", ""))
        anchor = obs.get("anchor", [""] * env_num)[index]
        controllers[index].start(anchor)
        metadata.append(
            {
                "gamefile": gamefile,
                "task_type": _task_type(gamefile),
                "task_description": controllers[index].plan.task,  # type: ignore[union-attr]
            }
        )

    skill_prompt = ""
    if skill_content.strip():
        skill_prompt = "## Skill knowledge\n" + skill_content.strip() + "\n\n"

    for step_index in range(max_steps):
        active = [index for index in range(env_num) if not env_dones[index]]
        if not active:
            break
        prompts = {}
        for index in active:
            prompt = skill_prompt + controllers[index].guidance() + "\n\n" + obs["text"][index]
            if diagnostic_mode and diagnostic_instruction.strip():
                prompt += "\n\n## Training readout\n" + diagnostic_instruction.strip()
            prompts[index] = prompt

        responses = ["None"] * env_num

        def call_model(index: int) -> tuple[int, str, dict[str, int], float, bool]:
            started = time.perf_counter()
            try:
                response, usage = chat_target(
                    system="You are an expert agent operating in the ALFRED Embodied Environment.",
                    user=prompts[index],
                    max_completion_tokens=max_completion_tokens,
                    retries=5,
                    stage="rollout",
                    timeout=None,
                )
                response = (response or "").strip()
                if not response or _extract(response, "action") is None:
                    response = "<think>missing usable action</think><action>look</action>"
                normalized_usage = {
                    "prompt_tokens": int((usage or {}).get("prompt_tokens", 0) or 0),
                    "completion_tokens": int((usage or {}).get("completion_tokens", 0) or 0),
                    "total_tokens": int((usage or {}).get("total_tokens", 0) or 0),
                }
                return index, response, normalized_usage, time.perf_counter() - started, False
            except Exception as exc:  # noqa: BLE001
                return (
                    index,
                    f"<think>model error: {type(exc).__name__}</think><action>look</action>",
                    {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    time.perf_counter() - started,
                    True,
                )

        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, max_api_workers)) as pool:
            futures = [pool.submit(call_model, index) for index in active]
            for future in concurrent.futures.as_completed(futures):
                index, response, usage, latency, had_error = future.result()
                responses[index] = response
                metric = model_metrics[index]
                metric["calls"] += 1
                metric["latencies"].append(latency)
                metric["prompt_tokens"] += usage["prompt_tokens"]
                metric["completion_tokens"] += usage["completion_tokens"]
                metric["total_tokens"] += usage["total_tokens"]
                metric["errors"] += int(had_error)

        previous_responses = {index: responses[index] for index in active}
        obs, rewards, dones, infos = env_manager.step(responses)

        for index in active:
            response = previous_responses[index]
            action = _extract(response, "action") or "look"
            anchor = obs.get("anchor", [""] * env_num)[index]
            info = infos[index] if isinstance(infos[index], dict) else {}
            event = controllers[index].observe(
                action,
                anchor,
                reward=float(rewards[index]),
                done=bool(dones[index]),
                info=info,
            )
            conversations[index].append(
                {
                    "step": step_index,
                    "action": action,
                    "reasoning": _extract(response, "think"),
                    "model_response": response,
                    "env_feedback": anchor,
                    "reward": float(rewards[index]),
                    "done": bool(dones[index]),
                    "chunk": event.active_step,
                    "checker_status": event.check.status.value,
                    "checker_reason": event.check.reason,
                    "controller_decision": event.decision.kind.value,
                    "plan_revision": event.plan_revision,
                    "interrupt_count": event.interruption_count,
                    "model_latency_seconds": model_metrics[index]["latencies"][-1],
                }
            )
            if dones[index]:
                env_dones[index] = True
                successes[index] = bool(info.get("won", False)) or float(rewards[index]) > 0

    results: list[dict] = []
    prediction_root = os.path.join(out_root, "predictions") if out_root else ""
    analyzer = FailureAnalyzer()
    updater = SkillUpdater()
    evolution_rows: list[dict] = []

    for index in range(env_num):
        task_id = str(result_ids[index]) if result_ids and index < len(result_ids) else f"env_{index:03d}"
        won = successes[index]
        fail_reason = ""
        if not won:
            fail_reason = "Episode ended without completing the task" if env_dones[index] else f"Timeout after {max_steps} steps"
        controller = controllers[index]
        metric = model_metrics[index]
        latency_total = float(sum(metric["latencies"]))
        result = {
            "id": task_id,
            "hard": int(won),
            "soft": float(won),
            "n_turns": len(conversations[index]),
            "fail_reason": fail_reason,
            "agent_ok": True,
            "task_type": metadata[index]["task_type"],
            "gamefile": metadata[index]["gamefile"],
            "task_description": metadata[index]["task_description"],
            "instruction_type": metadata[index]["task_type"],
            "controller_mode": mode,
            "interrupt_count": controller.interruption_count,
            "plan_revisions": controller.plan.revision if controller.plan else 0,
            "model_calls": metric["calls"],
            "model_errors": metric["errors"],
            "model_latency_total_seconds": round(latency_total, 4),
            "model_latency_mean_seconds": round(latency_total / metric["calls"], 4) if metric["calls"] else 0.0,
            "prompt_tokens": metric["prompt_tokens"],
            "completion_tokens": metric["completion_tokens"],
            "total_tokens": metric["total_tokens"],
            "estimated_cost_usd": None,
        }
        results.append(result)
        if prediction_root:
            directory = os.path.join(prediction_root, task_id)
            os.makedirs(directory, exist_ok=True)
            with open(os.path.join(directory, "conversation.json"), "w", encoding="utf-8") as handle:
                json.dump(conversations[index], handle, ensure_ascii=False, indent=2)
        if mode == "dynamic_evolution" and not won:
            report = analyzer.analyze(conversations[index])
            evolution_rows.append({"id": task_id, "suggestion": updater.suggest(report)})

    if out_root and mode == "dynamic_evolution" and evolution_rows:
        os.makedirs(out_root, exist_ok=True)
        path = os.path.join(out_root, "skill_update_suggestions.jsonl")
        with open(path, "a", encoding="utf-8") as handle:
            for row in evolution_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if out_root:
        all_latencies = [latency for metric in model_metrics for latency in metric["latencies"]]
        aggregate = {
            "controller_mode": mode,
            "episodes": env_num,
            "model_calls": sum(metric["calls"] for metric in model_metrics),
            "model_errors": sum(metric["errors"] for metric in model_metrics),
            "model_latency_total_seconds": round(sum(all_latencies), 4),
            "model_latency_mean_seconds": round(sum(all_latencies) / len(all_latencies), 4) if all_latencies else 0.0,
            "rollout_wall_seconds": round(time.perf_counter() - rollout_started, 4),
            "prompt_tokens": sum(metric["prompt_tokens"] for metric in model_metrics),
            "completion_tokens": sum(metric["completion_tokens"] for metric in model_metrics),
            "total_tokens": sum(metric["total_tokens"] for metric in model_metrics),
            "estimated_cost_usd": None,
            "cost_note": "Inferera unit prices were not supplied; calculate from recorded token counts.",
        }
        with open(os.path.join(out_root, "metrics.json"), "w", encoding="utf-8") as handle:
            json.dump(aggregate, handle, ensure_ascii=False, indent=2)
    return results
