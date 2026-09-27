from __future__ import annotations

import concurrent.futures
import json
import os
from pathlib import Path
import re
import time
from typing import Any

from skillopt.model import chat_target

from .failure_analyzer import FailureAnalyzer
from .interrupter import DecisionType
from .skill_executor import ChunkExecutor
from .skill_updater import SkillUpdater


def _extract(response: str, tag: str) -> str | None:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", response or "", re.DOTALL)
    return match.group(1).strip() if match else None


def parse_action_chunk(response: str, limit: int) -> list[str]:
    """Validate the complete queue before any action reaches the environment."""
    answer = re.sub(r"<think>.*?</think>", "", response or "", flags=re.DOTALL)
    chunks = re.findall(r"<actions>(.*?)</actions>", answer, re.DOTALL)
    if chunks:
        if len(chunks) != 1:
            raise ValueError("expected exactly one actions block")
        try:
            actions = json.loads(chunks[0])
        except json.JSONDecodeError as exc:
            raise ValueError("actions must be a JSON array") from exc
    else:
        # Legacy single-action output is an explicit length-one chunk.
        actions = re.findall(r"<action>(.*?)</action>", answer, re.DOTALL)
        if len(actions) != 1:
            raise ValueError("missing actions JSON array")
    if not isinstance(actions, list) or not 1 <= len(actions) <= limit:
        raise ValueError("chunk length outside requested bounds")
    if any(not isinstance(a, str) or not a.strip() or any(c in a for c in "\n\r<>") for a in actions):
        raise ValueError("each entry must be one non-empty action string")
    return [a.strip() for a in actions]


def _task_type(gamefile: str) -> str:
    names = ("pick_and_place", "pick_two_obj_and_place", "look_at_obj_in_light",
             "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep", "pick_clean_then_place_in_recep")
    return next((name for name in names if name in gamefile), "other")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _cost(metric: dict) -> None:
    rates = [os.environ.get(name, "").strip() for name in
             ("MODEL_INPUT_USD_PER_MILLION", "MODEL_OUTPUT_USD_PER_MILLION")]
    metric["estimated_cost_usd"] = None
    if all(rates):
        input_rate, output_rate = map(float, rates)
        incoming = metric["prompt_tokens"] * input_rate / 1_000_000
        outgoing = metric["completion_tokens"] * output_rate / 1_000_000
        metric.update(input_cost_usd=round(incoming, 10), output_cost_usd=round(outgoing, 10),
                      estimated_cost_usd=round(incoming + outgoing, 10),
                      pricing={"input_usd_per_million_tokens": input_rate,
                               "output_usd_per_million_tokens": output_rate,
                               "source": os.environ.get("MODEL_PRICE_SOURCE", "")})


def summarize_results(results: list[dict], mode: str, wall_seconds: float) -> dict:
    """Aggregate all episodes, including sequential adapter batches."""
    fields = ("model_calls", "model_errors", "parse_errors", "prompt_tokens", "completion_tokens",
              "total_tokens", "n_turns", "chunk_count", "multi_action_chunks", "planned_actions",
              "interrupt_count", "anomaly_replans", "chunk_boundaries", "discarded_actions",
              "model_latency_total_seconds")
    aggregate = {key: sum(row.get(key, 0) for row in results) for key in fields}
    calls = aggregate["model_calls"]
    chunks = aggregate["chunk_count"]
    aggregate.update(
        controller_mode=mode, implementation="action_queue_v2", episodes=len(results),
        environment_steps=aggregate.pop("n_turns"), rollout_wall_seconds=round(wall_seconds, 4),
        model_latency_mean_seconds=round(aggregate["model_latency_total_seconds"] / calls, 4) if calls else 0.0,
        mean_planned_chunk_length=aggregate["planned_actions"] / chunks if chunks else 0.0,
        model_call_definition="logical chat_target calls; backend retries are not counted separately",
    )
    aggregate["actions_per_model_call"] = aggregate["environment_steps"] / calls if calls else 0.0
    _cost(aggregate)
    return aggregate


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
    """Plan at queue boundaries and execute queued primitives one at a time.

    Modes share a planning contract. Only dynamic modes discard unexecuted
    actions on checker anomalies. All modes stop on terminal feedback.
    max_steps bounds environment actions, not model decisions.
    """
    if mode not in {"fixed_chunk", "dynamic_interrupt", "dynamic_evolution"}:
        raise ValueError(f"unsupported dynamic mode: {mode}")
    if fixed_chunk_size < 1:
        raise ValueError("fixed_chunk_size must be positive")
    rollout_started = time.perf_counter()
    obs, infos = env_manager.reset({})
    env_num = len(obs["text"])
    env_dones = [False] * env_num
    successes = [False] * env_num
    stop_reasons = [""] * env_num
    conversations: list[list[dict]] = [[] for _ in range(env_num)]
    calls: list[list[dict]] = [[] for _ in range(env_num)]
    controllers = [ChunkExecutor(chunk_size=fixed_chunk_size, dynamic=mode != "fixed_chunk") for _ in range(env_num)]
    metadata = []
    for index, controller in enumerate(controllers):
        info = infos[index] if isinstance(infos[index], dict) else {}
        gamefile = str(info.get("extra.gamefile", ""))
        controller.start(obs.get("anchor", obs["text"])[index])
        metadata.append({"gamefile": gamefile, "task_type": _task_type(gamefile),
                         "task_description": controller.plan.task})

    skill_prompt = "## Skill knowledge\n" + skill_content.strip() + "\n\n" if skill_content.strip() else ""
    for step_index in range(max_steps):
        active = [i for i in range(env_num) if not env_dones[i]]
        if not active:
            break
        needs_plan = [i for i in active if controllers[i].needs_plan]
        limit = min(fixed_chunk_size, max_steps - step_index)
        prompts = {}
        for i in needs_plan:
            prompt = skill_prompt + obs["text"][i]
            if diagnostic_mode and diagnostic_instruction.strip():
                prompt += "\n\n## Training readout\n" + diagnostic_instruction.strip()
            # Last instruction supersedes SkillOpt's embedded single-action format.
            prompt += "\n\n" + controllers[i].guidance()
            prompt += f"\nRemaining episode budget permits at most {limit} actions in this response."
            prompts[i] = prompt

        def call_model(index: int) -> tuple[int, dict]:
            started = time.perf_counter()
            record = {"call_id": len(calls[index]) + 1, "step": step_index,
                      "prompt": prompts[index], "response": "", "usage": {},
                      "model_error": None, "parse_error": None}
            try:
                response, usage = chat_target(
                    system="You are an expert agent operating in the ALFRED Embodied Environment.",
                    user=prompts[index], max_completion_tokens=max_completion_tokens,
                    retries=5, stage="rollout", timeout=None,
                )
                record["response"] = (response or "").strip()
                record["usage"] = {key: int((usage or {}).get(key, 0) or 0)
                                   for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
            except Exception as exc:
                record["model_error"] = type(exc).__name__
            record["latency_seconds"] = time.perf_counter() - started
            if record["model_error"] is None:
                try:
                    record["actions"] = parse_action_chunk(record["response"], limit)
                except ValueError as exc:
                    record["parse_error"] = str(exc)
            record["fallback"] = bool(record["model_error"] or record["parse_error"])
            if record["fallback"]:
                record["actions"] = ["look"]
            return index, record

        if needs_plan:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, max_api_workers)) as pool:
                futures = [pool.submit(call_model, i) for i in needs_plan]
                for future in concurrent.futures.as_completed(futures):
                    i, record = future.result()
                    controllers[i].load_chunk(record["actions"])
                    record["chunk_id"] = controllers[i].chunk_id
                    calls[i].append(record)
                    print(f"  [{mode}] env={i} step={step_index} call={record['call_id']} "
                          f"chunk_length={len(record['actions'])} latency={record['latency_seconds']:.1f}s "
                          f"fallback={record['fallback']}", flush=True)

        actions = {i: controllers[i].next_action() for i in active}
        # Official history sees only the executed primitive, never an unexecuted tail.
        responses = ["None"] * env_num
        for i in active:
            responses[i] = f"<think>queued action</think><action>{actions[i]}</action>"
        obs, rewards, dones, infos = env_manager.step(responses)
        for i in active:
            controller = controllers[i]
            anchor = obs.get("anchor", obs["text"])[i]
            info = infos[i] if isinstance(infos[i], dict) else {}
            event = controller.observe(actions[i], anchor, reward=float(rewards[i]), done=bool(dones[i]), info=info)
            if event.decision.kind in {DecisionType.REPLAN, DecisionType.STOP_FAILURE}:
                print(f"  [{mode}] env={i} step={step_index} {event.decision.kind.value}: "
                      f"{event.decision.reason}; discarded={len(event.discarded_actions)}", flush=True)
            called_now = i in needs_plan
            call = calls[i][-1]
            conversations[i].append({
                "step": step_index, "action": actions[i],
                "reasoning": _extract(call["response"], "think") if called_now else None,
                "model_response": call["response"] if called_now else None,
                "model_called": called_now, "model_call_id": call["call_id"],
                "model_latency_seconds": call["latency_seconds"] if called_now else 0.0,
                "chunk": controller.chunk_id, "chunk_id": controller.chunk_id,
                "chunk_action_index": controller.actions_in_chunk - 1, "chunk_length": controller.chunk_length,
                "remaining_actions": list(controller.pending_actions), "discarded_actions": list(event.discarded_actions),
                "env_feedback": anchor, "reward": float(rewards[i]), "done": bool(dones[i]),
                "checker_status": event.check.status.value, "checker_reason": event.check.reason,
                "controller_decision": event.decision.kind.value, "controller_reason": event.decision.reason,
                "plan_revision": event.plan_revision, "interrupt_count": event.interruption_count,
                "chunk_boundaries": controller.boundary_count,
            })
            if dones[i]:
                env_dones[i] = True
                successes[i] = bool(info.get("won", False)) or float(rewards[i]) > 0
            elif event.decision.kind is DecisionType.STOP_FAILURE:
                env_dones[i] = True
                stop_reasons[i] = event.decision.reason
            if step_index == max_steps - 1 and controller.pending_actions:
                conversations[i][-1]["discarded_at_step_limit"] = list(controller.pending_actions)
                controller.pending_actions.clear()

    results = []
    evolution_rows = []
    for i, controller in enumerate(controllers):
        task_id = str(result_ids[i]) if result_ids else f"env_{i:03d}"
        count = len(calls[i])
        latency = sum(row["latency_seconds"] for row in calls[i])
        result = {
            "id": task_id, "hard": int(successes[i]), "soft": float(successes[i]), "n_turns": len(conversations[i]),
            "fail_reason": "" if successes[i] else stop_reasons[i] or (
                "Episode ended without completing the task" if env_dones[i] else f"Timeout after {max_steps} steps"),
            "agent_ok": not any(row["model_error"] or row["parse_error"] for row in calls[i]),
            **metadata[i], "instruction_type": metadata[i]["task_type"],
            "controller_mode": mode, "implementation": "action_queue_v2", "chunk_size": fixed_chunk_size,
            "interrupt_count": controller.interruption_count,
            "anomaly_replans": controller.anomaly_replan_count, "chunk_boundaries": controller.boundary_count,
            "plan_revisions": controller.plan.revision,
            "chunk_count": count, "multi_action_chunks": sum(len(row["actions"]) > 1 for row in calls[i]),
            "planned_actions": sum(len(row["actions"]) for row in calls[i]),
            "discarded_actions": sum(len(row["discarded_actions"]) + len(row.get("discarded_at_step_limit", [])) for row in conversations[i]),
            "model_calls": count, "model_errors": sum(bool(row["model_error"]) for row in calls[i]),
            "parse_errors": sum(bool(row["parse_error"]) for row in calls[i]),
            "model_latency_total_seconds": round(latency, 4),
            "model_latency_mean_seconds": round(latency / count, 4) if count else 0.0,
            "actions_per_model_call": len(conversations[i]) / count if count else 0.0,
            **{key: sum(row["usage"].get(key, 0) for row in calls[i])
               for key in ("prompt_tokens", "completion_tokens", "total_tokens")},
        }
        _cost(result)
        results.append(result)
        if out_root:
            directory = Path(out_root) / "predictions" / task_id
            _write_json(directory / "conversation.json", conversations[i])
            _write_json(directory / "model_calls.json", calls[i])
        if mode == "dynamic_evolution" and not successes[i]:
            report = FailureAnalyzer().analyze(conversations[i])
            evolution_rows.append({"id": task_id, "suggestion": SkillUpdater().suggest(report)})
    if out_root:
        _write_json(Path(out_root) / "metrics.json", summarize_results(results, mode, time.perf_counter() - rollout_started))
        if evolution_rows:
            with (Path(out_root) / "skill_update_suggestions.jsonl").open("a", encoding="utf-8") as handle:
                for row in evolution_rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return results
