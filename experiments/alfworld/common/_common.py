from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time


PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
SKILLOPT_ROOT = (PROJECT_ROOT / "skillopt").resolve()


def _absolute_cli_paths(arguments: list[str], base: Path) -> list[str]:
    path_flags = {
        "--config",
        "--skill",
        "--out_root",
        "--data_path",
        "--split_dir",
        "--split_output_dir",
    }
    resolved = list(arguments)
    for index, token in enumerate(resolved[:-1]):
        if token in path_flags:
            path = Path(resolved[index + 1]).expanduser()
            if not path.is_absolute():
                path = base / path
            resolved[index + 1] = str(path.resolve())
    return resolved


def run(mode: str) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--chunk-size", type=int, default=4)
    custom, remaining = parser.parse_known_args()
    if custom.chunk_size < 1:
        raise SystemExit("--chunk-size must be positive")

    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(SKILLOPT_ROOT))
    if mode == "react":
        from dynamic_skill.react_metrics import run_instrumented_react_batch
        from skillopt.envs.alfworld import adapter as adapter_module

        adapter_module.run_alfworld_batch = run_instrumented_react_batch
    else:
        from dynamic_skill.alfworld_rollout import run_dynamic_alfworld_batch, summarize_results
        from skillopt.envs.alfworld import adapter as adapter_module

        collected: dict[str, dict] = {}

        def measured_batch(*args, **kwargs):
            started = time.perf_counter()
            rows = run_dynamic_alfworld_batch(*args, **kwargs, mode=mode, fixed_chunk_size=custom.chunk_size)
            output = str(kwargs.get("out_root") or "")
            bucket = collected.setdefault(output, {"rows": [], "wall": 0.0})
            bucket["rows"].extend(rows)
            bucket["wall"] += time.perf_counter() - started
            if output:
                (Path(output) / "metrics.json").write_text(
                    json.dumps(summarize_results(bucket["rows"], mode, bucket["wall"]), indent=2) + "\n",
                    encoding="utf-8",
                )
            return rows

        adapter_module.run_alfworld_batch = measured_batch

    from scripts import eval_only

    remaining = _absolute_cli_paths(remaining, Path.cwd())
    from dynamic_skill.thinking_policy import non_thinking_requests
    audit_path = None
    for index, token in enumerate(remaining):
        if token == "--out_root" and index + 1 < len(remaining):
            audit_path = Path(remaining[index + 1]) / "thinking_policy.jsonl"
        elif token.startswith("--out_root="):
            audit_path = Path(token.split("=", 1)[1]).resolve() / "thinking_policy.jsonl"
    os.chdir(SKILLOPT_ROOT)
    sys.argv = [sys.argv[0], *remaining]
    # Applied AFTER all upstream config resolution, at the actual send boundary.
    with non_thinking_requests(audit_path):
        eval_only.main()
