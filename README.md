# Dynamic Skill MVP

Training-free long-horizon agent prototype built on SkillOpt's ALFWorld
environment and evaluation infrastructure. The official repository is referenced
through the `skillopt` symlink; no SkillOpt source file is modified.

## What is implemented

- structured skill plans and deterministic task induction;
- per-action observation checking for invalid actions, loops, stalled state,
  environment failure, and success;
- fixed-size and dynamically interrupted chunks;
- semantic rollback to the last stable plan boundary followed by replanning;
- trajectory failure analysis and deterministic skill-update suggestions;
- four comparable experiment entry points.

ALFWorld does not provide a reversible simulator snapshot. Therefore "rollback"
means rolling back the controller's plan state, not undoing physical actions that
have already happened in the environment.

## Setup

```bash
cd /opt/data/private/src/my-project
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e './skillopt[alfworld]' pytest
.venv/bin/python -m pip install -e .
alfworld-download
```

## Baseline: two ALFWorld samples

```bash
.venv/bin/python experiments/react.py \
  --config skillopt/configs/alfworld/default.yaml \
  --skill skillopt/skillopt/envs/alfworld/skills/initial.md \
  --split valid_seen \
  --test_env_num 2 \
  --workers 1 \
  --max_api_workers 1 \
  --out_root results/react_2
```

The model backend and credentials use the same SkillOpt configuration and
environment variables as the official evaluator.

## Ablations

Replace `react.py` with one of:

- `fixed_chunk.py --chunk-size 4`
- `dynamic_interrupt.py --chunk-size 4`
- `dynamic_evolution.py --chunk-size 4`

All ordinary arguments are passed unchanged to SkillOpt's official
`scripts/eval_only.py` entry point.

