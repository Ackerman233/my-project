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

Real-environment integration checks (scripted model responses; no API requests):

```bash
ALFWORLD_DATA=/opt/data/private/data/alfworld RUN_ALFWORLD_INTEGRATION=1 \
  .venv/bin/python -m pytest -q tests/test_alfworld_integration.py
```

These inject a rejected action and verify that fixed mode executes the queued
tail while dynamic mode discards it, then both finish the real ALFWorld task.
They validate execution mechanics, not model performance.

## Rush Hour and Sudoku pilot

See [experiments/PUZZLE_PILOT.md](experiments/PUZZLE_PILOT.md) for the verified
four-task dataset and paired React / fixed chunk / dynamic interrupt runner.
It reuses this project's action queue without modifying official SkillOpt.
Offline solvers only validate tasks; the runtime checker has no solution access.
The initial 2026-09-27 API attempt was timeout-censored, not a usable performance
comparison. Calls, latency, partial usage and full action traces are retained.

## Non-thinking policy (2026-09-27)

All project experiment entry points now explicitly send `enable_thinking: false`
in the top-level Qwen Chat Completions request JSON. This covers ALFWorld React,
fixed chunk, dynamic interrupt/evolution and all Rush Hour/Sudoku modes. There
is no enable-thinking CLI override. Conflicting `reasoning_effort` is removed;
if a backend emits `chat_template_kwargs.enable_thinking`, that is false too.
Official SkillOpt files are unchanged; project ALFWorld entry points install a
temporary request-boundary hook for the duration of their run. Directly running
`skillopt/scripts/eval_only.py` bypasses this project policy.

ALFWorld writes `thinking_policy.jsonl` under an explicitly specified `--out_root`:
request switch, response id, returned reasoning-token count and reasoning-content
length. `thinking_returned` means the provider returned reasoning despite the
switch; `not_reported` is unknown, not proof of zero reasoning. Puzzle runs save
the switch in `config.json` and every `model_calls.json` row, along with returned
usage and reasoning content. Existing runs are not rewritten. New non-thinking
runs must use fresh output directories and must not be pooled with old runs.

Offline HTTP-transport tests verify the actual outgoing JSON, not only config
values. They do not establish that the remote gateway honors the switch; no
paid run was made as part of this policy change.
