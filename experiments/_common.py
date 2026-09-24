from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKILLOPT_ROOT = (PROJECT_ROOT / "skillopt").resolve()


def run(mode: str) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--chunk-size", type=int, default=4)
    custom, remaining = parser.parse_known_args()
    if custom.chunk_size < 1:
        raise SystemExit("--chunk-size must be positive")

    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(SKILLOPT_ROOT))
    if mode != "react":
        from dynamic_skill.alfworld_rollout import run_dynamic_alfworld_batch
        from skillopt.envs.alfworld import adapter as adapter_module

        adapter_module.run_alfworld_batch = partial(
            run_dynamic_alfworld_batch,
            mode=mode,
            fixed_chunk_size=custom.chunk_size,
        )

    from scripts import eval_only

    sys.argv = [sys.argv[0], *remaining]
    eval_only.main()

