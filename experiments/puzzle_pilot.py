"""Training-free, paired puzzle pilot using the same action queue as ALFWorld."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dynamic_skill.alfworld_rollout import parse_action_chunk
from dynamic_skill.interrupter import DecisionType
from dynamic_skill.puzzles import PuzzleChecker, make_environment
from dynamic_skill.skill_executor import ChunkExecutor
from dynamic_skill.thinking_policy import non_thinking_body
from dynamic_skill.campaign_budget import CampaignHalt


@dataclass
class Settings:
    model: str = "qwen3.5-35b-a3b"
    base_url: str = "https://api.inferera.com/v1"
    chunk_size: int = 4
    max_steps: int = 90
    max_calls: int = 64
    max_tokens: int = 4096
    temperature: float = 0.0
    request_timeout: float = 60.0
    episode_seconds: float = 1200.0
    input_usd_per_million: float | None = None
    output_usd_per_million: float | None = None
    action_only: bool = False
    prompt_variant: str = "baseline"


def task_rules(env, settings):
    if settings.prompt_variant == "baseline":
        return env.rules
    if settings.prompt_variant == "neutral_rush_v1":
        return env.rules.replace("Plan the necessary rearrangements, not only moves that immediately advance A.", "").rstrip()
    raise ValueError("Unknown prompt variant")


def planner_system_prompt(settings, action_limit=None):
    if settings.action_only:
        if action_limit == 1:
            return ('You control a puzzle environment. Output ONLY <actions>["action"]</actions>. '
                    'The array must contain exactly ONE concrete legal environment command, not a description. '
                    'Do not output analysis, reasoning, explanations, markdown, or any text outside the actions tags. '
                    'Never include a second action. Do not try to narrate a full solution.')
        return ('You control a puzzle environment. Output ONLY <actions>["action", "action"]</actions>. '
                'The array must contain concrete legal environment commands, not descriptions. '
                'Do not output analysis, reasoning, explanations, markdown, or any text outside the actions tags. '
                'Respect the requested maximum number of actions. Do not try to narrate a full solution.')
    return "You solve puzzles by issuing concrete environment actions. Follow the requested action format."


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class ApiPlanner:
    def __init__(self, settings, action_limit=None):
        from openai import OpenAI
        self.client = OpenAI(base_url=settings.base_url, api_key=os.environ["TARGET_AZURE_OPENAI_API_KEY"],
                             timeout=settings.request_timeout, max_retries=0)
        self.settings = settings
        self.action_limit = action_limit

    def __call__(self, prompt):
        response = self.client.chat.completions.create(
            model=self.settings.model,
            messages=[{"role": "system", "content": planner_system_prompt(self.settings, self.action_limit)},
                      {"role": "user", "content": prompt}],
            temperature=self.settings.temperature, max_tokens=self.settings.max_tokens,
            extra_body=non_thinking_body(),
        )
        choice = response.choices[0]
        usage = response.usage.model_dump() if response.usage else None
        return {"response": choice.message.content or "", "usage": usage,
                "response_id": getattr(response, "id", None),
                "finish_reason": choice.finish_reason, "provider_model": response.model,
                "reasoning_content": getattr(choice.message, "reasoning_content", None)}

    def close(self):
        self.client.close()


def run_episode(task, mode, settings, output, planner):
    env = make_environment(task)
    k = 1 if mode == "react" else settings.chunk_size
    controller = ChunkExecutor(chunk_size=k, dynamic=mode == "dynamic_interrupt", checker=PuzzleChecker())
    controller.start(env.render(), task=env.rules)
    calls, trajectory = [], []
    started = time.perf_counter()
    status = "step_limit"
    episode_id = task["id"] + "__" + mode
    directory = output / "predictions" / episode_id
    directory.mkdir(parents=True, exist_ok=False)
    for step_index in range(settings.max_steps):
        if time.perf_counter() - started >= settings.episode_seconds:
            status = "wall_time_limit"
            break
        called_now = controller.needs_plan
        if called_now:
            if len(calls) >= settings.max_calls:
                status = "model_call_limit"
                break
            if hasattr(planner, "before_call"):
                try:
                    planner.before_call()
                except CampaignHalt as halt:
                    status = halt.status
                    break  # No request, no phantom model-call record.
            limit = min(k, settings.max_steps - step_index)
            history = [{key: row[key] for key in ("action", "valid", "feedback")} for row in trajectory[-8:]]
            prompt = (task_rules(env, settings) + "\n\nCurrent board:\n" + env.render()
                      + "\nRecent executed actions and feedback:\n" + json.dumps(history)
                      + f"\nPlan 1 to {limit} actions in order. Use several actions when you can predict their effects. "
                      "They will execute one at a time without another model call between them. "
                      "Do not output a solved board or a program. Keep reasoning brief. "
                      'Return <actions>["action", "action"]</actions> containing a JSON string array. ')
            if controller.last_replan_reason:
                prompt += "\nPrevious anomaly: " + controller.last_replan_reason
            if settings.action_only:
                prompt = prompt.replace("Keep reasoning brief. ", "")
                prompt += "\nOutput only the actions tags and JSON array. No explanation or analysis."
            if limit == 1:
                prompt = prompt.replace("Use several actions when you can predict their effects. ",
                                        "Return exactly ONE action; never include a second action. ")
                prompt = prompt.replace('["action", "action"]', '["action"]')
            before = time.perf_counter()
            record = {"call_id": len(calls) + 1, "step": step_index, "prompt": prompt,
                      "system_prompt": planner_system_prompt(settings, k),
                      "request_extra_body": non_thinking_body(),
                      "status": "pending", "actions": [], "response": "", "usage": None}
            calls.append(record)
            save(directory / "model_calls.json", calls)
            try:
                record.update(planner(prompt))
                record["status"] = "ok"
            except Exception as exc:
                record["status"] = "api_error"
                record["error_type"] = type(exc).__name__
                record["http_status"] = getattr(exc, "status_code", None)
            record["latency_seconds"] = time.perf_counter() - before
            if record["status"] == "ok":
                try:
                    record["actions"] = parse_action_chunk(record["response"], limit)
                except ValueError as exc:
                    record["status"] = "format_error"
                    record["parse_error"] = str(exc)
            save(directory / "model_calls.json", calls)
            print(f"[{episode_id}] call={len(calls)} step={step_index} k={len(record['actions'])} "
                  f"latency={record['latency_seconds']:.1f}s {record['status']}", flush=True)
            if record["status"] != "ok":
                status = record["status"]
                break
            controller.load_chunk(record["actions"])
        action = controller.next_action()
        result = env.step(action)
        event = controller.observe(action, result.observation, reward=float(result.won), done=result.won,
                                   info={"valid": result.valid, "reason": result.reason, "won": result.won})
        row = {"step": step_index, "action": action, "valid": result.valid, "feedback": result.reason,
               "board_after": result.observation, "won": result.won, "model_called": called_now,
               "model_call_id": len(calls), "chunk_id": controller.chunk_id,
               "chunk_action_index": controller.actions_in_chunk - 1,
               "remaining_actions": list(controller.pending_actions),
               "discarded_actions": list(event.discarded_actions),
               "checker_status": event.check.status.value, "checker_reason": event.check.reason,
               "controller_decision": event.decision.kind.value, "controller_reason": event.decision.reason,
               "interrupt_count": controller.interruption_count}
        trajectory.append(row)
        save(directory / "conversation.json", trajectory)
        if result.won:
            status = "success"
            break
        if event.decision.kind is DecisionType.STOP_FAILURE:
            status = "interrupt_budget_exhausted"
            break
    usage_missing = any(row.get("usage") is None for row in calls)
    usage = {key: sum((row.get("usage") or {}).get(key, 0) or 0 for row in calls)
             for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
    cost = None
    if not usage_missing and settings.input_usd_per_million is not None and settings.output_usd_per_million is not None:
        cost = (usage["prompt_tokens"] * settings.input_usd_per_million + usage["completion_tokens"] * settings.output_usd_per_million) / 1e6
    result = {"id": episode_id, "task_id": task["id"], "benchmark": task["benchmark"], "mode": mode,
              "status": status, "success": status == "success", "minimum_actions": task["minimum_actions"],
              "environment_steps": len(trajectory), "model_calls": len(calls),
              "model_errors": sum(c["status"] == "api_error" for c in calls),
              "format_errors": sum(c["status"] == "format_error" for c in calls),
              "invalid_actions": sum(not row["valid"] for row in trajectory),
              "interrupt_count": controller.interruption_count, "anomaly_replans": controller.anomaly_replan_count,
              "chunk_boundaries": controller.boundary_count,
              "multi_action_chunks": sum(len(c["actions"]) > 1 for c in calls),
              "planned_actions": sum(len(c["actions"]) for c in calls),
              "discarded_actions": sum(len(row["discarded_actions"]) for row in trajectory),
              "unexecuted_at_stop": list(controller.pending_actions),
              "actions_per_model_call": len(trajectory) / len(calls) if calls else 0,
              "latency_total_seconds": sum(c.get("latency_seconds", 0) for c in calls),
              "wall_seconds": time.perf_counter() - started, "usage_incomplete": usage_missing,
              **usage, "estimated_cost_usd": cost}
    save(directory / "result.json", result)
    print(f"[{episode_id}] FINISHED {status}: steps={len(trajectory)} calls={len(calls)}", flush=True)
    return result


def summarize(rows):
    summary = {}
    for benchmark in ("rush_hour", "sudoku"):
        for mode in ("react", "fixed_chunk", "dynamic_interrupt"):
            selected = [r for r in rows if r["benchmark"] == benchmark and r["mode"] == mode]
            if not selected:
                continue
            entry = {"episodes": len(selected), "successes": sum(r["success"] for r in selected),
                     "api_failures": sum(r["status"] == "api_error" for r in selected),
                     "format_failures": sum(r["status"] == "format_error" for r in selected),
                     "incomplete_limits": sum(r["status"] in {"wall_time_limit", "model_call_limit"} for r in selected)}
            for key in ("environment_steps", "model_calls", "invalid_actions", "interrupt_count",
                        "multi_action_chunks", "prompt_tokens", "completion_tokens", "total_tokens"):
                entry[key] = sum(r[key] for r in selected)
            entry["estimated_cost_usd"] = (sum(r["estimated_cost_usd"] for r in selected)
                if all(r["estimated_cost_usd"] is not None for r in selected) else None)
            entry["mean_episode_wall_seconds"] = sum(r["wall_seconds"] for r in selected) / len(selected)
            summary[f"{benchmark}/{mode}"] = entry
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--task-ids", nargs="+", help="Optional exact task ids, e.g. one task for a smoke test")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model", default="qwen3.5-35b-a3b")
    parser.add_argument("--base-url", default="https://api.inferera.com/v1")
    parser.add_argument("--request-timeout", type=float, default=60)
    parser.add_argument("--episode-seconds", type=float, default=1200)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=90)
    parser.add_argument("--max-calls", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--action-only", action="store_true", help="Require actions only, without prose (recorded prompt variant)")
    parser.add_argument("--modes", nargs="+", choices=["react", "fixed_chunk", "dynamic_interrupt"],
                        default=["react", "fixed_chunk", "dynamic_interrupt"])
    parser.add_argument("--input-price", type=float)
    parser.add_argument("--output-price", type=float)
    args = parser.parse_args()
    if any(v <= 0 for v in (args.workers, args.request_timeout, args.episode_seconds, args.max_tokens, args.max_steps, args.max_calls, args.chunk_size)):
        parser.error("budgets and workers must be positive")
    tasks = json.loads(args.tasks.read_text())
    if args.task_ids:
        missing = set(args.task_ids) - {task["id"] for task in tasks}
        if missing:
            parser.error(f"unknown task ids: {sorted(missing)}")
        tasks = [task for task in tasks if task["id"] in args.task_ids]
    if len({task["id"] for task in tasks}) != len(tasks) or not tasks:
        parser.error("task ids must be unique and dataset must be nonempty")
    if len(set(args.modes)) != len(args.modes):
        parser.error("modes must be unique")
    args.out.mkdir(parents=True, exist_ok=False)
    settings = Settings(model=args.model, base_url=args.base_url, request_timeout=args.request_timeout,
                        episode_seconds=args.episode_seconds, max_tokens=args.max_tokens, max_steps=args.max_steps,
                        max_calls=args.max_calls, chunk_size=args.chunk_size,
                        action_only=args.action_only,
                        input_usd_per_million=args.input_price, output_usd_per_million=args.output_price)
    jobs = [(task, mode) for task in tasks for mode in args.modes]
    random.Random(20260927).shuffle(jobs)
    save(args.out / "config.json", {**asdict(settings), "tasks_sha256": hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
                                   "enable_thinking": False, "request_extra_body": non_thinking_body(),
                                   "episode_count": len(jobs), "workers": args.workers, "provider_retries": 0,
                                   "run_order": [[t["id"], m] for t, m in jobs], "planning_answers_hidden": True,
                                   "react_note": "Puzzle-specific one-action baseline, not the ALFWorld official prompt",
                                   "pricing_note": "Optional caller-supplied rates; estimates, not provider billing"})
    rows = []

    def run(job):
        planner = ApiPlanner(settings, action_limit=1 if job[1] == "react" else settings.chunk_size)
        try:
            return run_episode(*job, settings, args.out, planner)
        finally:
            planner.close()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run, job): job for job in jobs}
        for future in as_completed(futures):
            rows.append(future.result())
            with (args.out / "results.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")
            save(args.out / "metrics.json", summarize(rows))
    save(args.out / "run_status.json", {"status": "finished", "episodes": len(rows),
                                        "api_failures": sum(r["status"] == "api_error" for r in rows)})
    print(json.dumps(summarize(rows), indent=2))


if __name__ == "__main__":
    main()
