# Rush Hour + Sudoku pilot (2026-09-27)

This is a training-free mechanism pilot, not a reproduction of published scores.
It uses the same `ChunkExecutor` / `Interrupter` as the real ALFWorld action queue.
The model generates all actions. Offline solvers are used only to select and
verify datasets, never inside the policy, observation checker or model prompt.

## Tasks and horizon definitions

- Sudoku: two local synthetic **test-split** L7 boards from
  `../ontraining-long-horizon/data/sudoku_paper_scale.jsonl`. Each has 41 blanks;
  an independent backtracking check confirms exactly one solution. One `set(r,c,v)`
  fills one cell. `clear(r,c)` enables correction of agent entries. Local row,
  column and box conflicts are rejected. Valid guesses are **not** compared with
  the hidden solution. Given clues cannot be altered. Agent entries are marked
  with `*` so their reversibility remains observable after old history is dropped.
  Success requires a complete
  valid grid. Forty-one blanks measure required writes, not reasoning difficulty.
- Rush Hour: two distinct vehicle configurations from the public examples in
  [Michael Fogleman's database description](https://www.michaelfogleman.com/rush/).
  We independently reproduce their 51- and 50-move optimal lengths using BFS,
  then select intermediate boards with remaining shortest paths of 20 and 25
  moves. One action slides one vehicle **any legal distance**; moving four cells
  is not artificially counted as four actions. Both boards are solvable and have
  different vehicle configurations. Target A reaches the right edge of row 3.

`data/puzzle_pilot_20260927/tasks.json` contains only tasks and provenance.
`validation_private.json` holds offline solutions and is not read by the runner.
`manifest.json` preserves source and task hashes. These are tiny derived pilot
samples, not a representative benchmark or an independent held-out paper dataset.

## Controlled comparison

- `react`: puzzle-specific one-action-per-call baseline (not the official ALFWorld prompt).
- `fixed_chunk`: the same planner with up to four actions per call; consume the
  queue even when a nonterminal action is rejected.
- `dynamic_interrupt`: same up-to-four-action planner; checker can discard the
  remaining queue and replan after illegal moves or revisiting a recent board.

All modes receive the same board/rules and last eight executed actions/feedback.
Only maximum chunk length and runtime interruption differ. No solution hints,
solver-generated candidates, optimum path, or future state are provided. The
checker reads legality feedback and recent visible states; it is not an oracle.
Normal queue exhaustion is not an interrupt. Chunk size is a cap, so shorter
plans are allowed and measured. Dynamic mode retains the existing limit of eight
anomaly replans; an attempted ninth terminates the episode and is reported.

Initial pilot: same `qwen3.5-35b-a3b`, temperature 0, requested `max_tokens=4096`,
four concurrent episodes, deterministic shuffled job order (seed 20260927),
90 primitive-action and 64 model-call budgets per episode. One HTTP attempt per
logical call, no SDK retries; 60-second request timeout and 1200-second episode
wall limit checked between steps. A final in-flight request can extend the wall
budget by roughly its timeout (transport overhead can add time). All settings
are saved in `config.json`. The provider returned 8,350 completion tokens in one
call despite `max_tokens=4096`, including 8,294 reasoning tokens: do not treat
this parameter as a verified total-token or billing cap on this endpoint.

Transport/auth/timeouts (`api_error`) are infrastructure failures, not evidence
of puzzle-solving failure. Malformed/truncated output (`format_error`) is reported
separately. Neither is silently replaced with a plausible action. Missing usage
makes cost unknown; reported token counts then cover returned usage only.
Prices are caller-supplied estimates, not billing; the initial run reuses the
prior ALFWorld price snapshot, not a freshly verified quote.

## Run

The API key must already be in `TARGET_AZURE_OPENAI_API_KEY`; never commit it.
From `/opt/data/private/src/my-project`:

```bash
.venv/bin/python experiments/prepare_puzzles.py \
  --sudoku-source ../ontraining-long-horizon/data/sudoku_paper_scale.jsonl \
  --out data/puzzle_pilot_20260927

.venv/bin/python experiments/puzzle_pilot.py \
  --tasks data/puzzle_pilot_20260927/tasks.json \
  --out results/puzzle_pilot_NEW_RUN \
  --model qwen3.5-35b-a3b --chunk-size 4 --workers 4
```

Preparation and evaluation refuse to overwrite existing outputs. Skip preparation
when the verified dataset already exists. Tests: `.venv/bin/python -m pytest -q`.

## Read the results

- `results.jsonl`: one summary per task/mode, status, success, environment steps,
  model calls, invalid actions, actual interrupts, chunk lengths, tokens and cost.
- `metrics.json`: running aggregates by benchmark/mode; API/format/limit failures
  explicitly separated. Counts may describe partial runs until `run_status.json`
  says finished and all 12 episode rows exist.
- `predictions/<task>__<mode>/model_calls.json`: full call prompts/responses,
  planned queues, usage, latency, provider finish reason and error type. Updated
  before/after each request so stalls remain diagnosable.
- `conversation.json`: each actual action and resulting board, checker reason,
  discarded tail, model call id, chunk id and action offset. Updated every step.

Only compare efficiency after inspecting success and completion status. Dynamic
interruption has no demonstrated benefit if the planner makes no harmful errors,
or if no queue is actually interrupted. If both chunk modes beat the one-action
baseline, that supports batching, not automatically the interrupt mechanism.
Two tasks per benchmark and one run per method cannot establish generality or
statistical significance. Repeat across more tasks/seeds and chunk lengths next.

## Initial online attempt: infrastructure-censored

`results/puzzle_pilot_20260927T130525Z` contains all 12 episode records, but all
terminated with `APITimeoutError`. There were 13 API attempts: one successful
four-action Sudoku plan followed by a timeout, and eleven first-call timeouts.
Four Sudoku actions were accepted; there is no completed task and no valid
success-rate or efficiency comparison. Token totals are partial and total cost
is unknown, not zero. Two additional diagnostic puzzle requests tried the
Qwen-documented direct/template non-thinking switches with 30-second timeouts;
both timed out. This does not establish whether the provider supports them.
A final minimal `Reply only OK.` request also timed out at about 30 seconds, so
the root cause cannot be narrowed to puzzle reasoning alone. No batch is left
running. See `results/puzzle_api_health_20260927.json` and
`results/puzzle_api_probe_20260927.json` for the diagnostics.

The initial attempt predates the `*` marker for mutable Sudoku entries; its
full prompts preserve the original observation format. Do not pool future
runs with it. Start with one request/episode and verify endpoint behavior and
returned reasoning usage before launching another 12-episode batch.
To restrict a diagnostic run, add `--task-ids sudoku-L7-6502031 --modes fixed_chunk
--max-calls 1 --workers 1`. A `model_call_limit` result in this one-call diagnostic
is expected and is not a completed episode.
