# Dynamic Skill MVP

Training-free long-horizon agent prototype built on SkillOpt's ALFWorld
environment and evaluation infrastructure. The official repository is referenced
through the `skillopt` symlink; no SkillOpt source file is modified.

## What is implemented

- structured skill plans and deterministic task induction;
- per-action observation checking for invalid actions, loops, stalled state,
  environment failure, and success;
- real model-generated action queues (1..K primitive actions per call);
- fixed chunk execution and observation-triggered queue cancellation/replanning;
- trajectory failure analysis and deterministic skill-update suggestions;
- four comparable experiment entry points.

On interruption, only unexecuted actions are discarded. Recovery plans from the
current environment observation; already executed physical actions are not undone.
Version `action_queue_v2` replaces the earlier per-step controller. Old experiment
directories are preserved, but are NOT action-chunking results.

## Setup

```bash
cd /opt/data/private/src/my-project
export ALFWORLD_DATA=/opt/data/private/data/alfworld
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e './skillopt[alfworld]' pytest
.venv/bin/python -m pip install -e .
alfworld-download
```

## Baseline: two ALFWorld samples

```bash
cd /opt/data/private/src/my-project
export ALFWORLD_DATA=/opt/data/private/data/alfworld
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

## Action queue semantics (2026-09-27)

The model returns `<think>...</think><actions>["go to desk 1", "take bowl 1 from desk 1"]</actions>`.
Each action executes separately through the official environment manager. A new
model call occurs only when the queue is empty. K is a maximum, not a forced
length: plans should stop before an unknown observation or object ID is needed.
Both modes have the same planning prompt, input history, and output parser.

- Fixed chunk consumes the queue despite nonterminal checker anomalies.
- Dynamic interrupt discards the remaining queue on invalid action, loop, or
  stalled observation, then plans from the latest state and rejection reason.
- Success or terminal failure stops both modes immediately, including mid-queue.
- `--chunk-size 1` gives one action per decision with this planner prompt; the
  separate `react.py` entry still runs the official React baseline.
- Malformed chunks and API failures are explicitly recorded and use a one-step
  `look` fallback. Legacy `<action>...</action>` output is accepted as length one.
- The original model backend configuration is retained. The rollout's legacy
  `temperature` argument, as in upstream, does not configure backend sampling.

Use a NEW `--out_root` per run: the official evaluator can reuse cached results.
For example, use the baseline command above with `fixed_chunk.py --chunk-size 4`
and `--out_root results/fixed_chunk_v2_2`; then use `dynamic_interrupt.py` and
`--out_root results/dynamic_interrupt_v2_2`. Add `--target_model qwen3.5-35b-a3b`
when using the previously configured Inferera backend.

## What to inspect

- `metrics.json`: environment_steps, model_calls, actions_per_model_call,
  multi_action_chunks, mean_planned_chunk_length, token usage, latency, errors.
- `results.jsonl`: per-episode success, steps, calls and queue statistics.
- `predictions/<id>/model_calls.json`: one row per logical model call, containing
  prompt, raw response, planned actions, usage, latency and parsing/API errors.
- `predictions/<id>/conversation.json`: one row per executed action, containing
  model_call_id, model_called, chunk_id, chunk_action_index (zero-based), remaining
  and discarded actions, checker_reason and controller_reason. Queued actions
  have zero new model latency and no repeated model response.

`chunk_boundaries` counts normal nonterminal queue exhaustion. `interrupt_count`
counts anomaly-triggered cancellation of a nonempty tail. `anomaly_replans` also
includes anomalies at the last action, where no tail remains. Terminal cancellation
does not count as an anomaly interrupt. `discarded_actions` includes all unused
tail actions, including terminal stops. Internal provider retries are not counted
as separate model calls. Usage and cost can therefore differ from provider billing.

Set `MODEL_INPUT_USD_PER_MILLION`, `MODEL_OUTPUT_USD_PER_MILLION` and optionally
`MODEL_PRICE_SOURCE` to compute estimated cost; otherwise cost remains null.
The experiment entry point aggregates metrics across all sequential environment
batches as well as concurrent episodes. React's existing instrumentation is unchanged.

Verification: `.venv/bin/python -m pytest -q`. Tests cover multi-action execution,
queue cancellation, fixed-mode continuation after rejection, terminal stops,
independent concurrent queues, step/interrupt limits and malformed model output.
