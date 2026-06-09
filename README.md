# llm-batch

Transparent, resumable **interactive ↔ batch** LLM processing on top of
[litellm](https://github.com/BerriAI/litellm). Flip one parameter (`mode`) and
the same call site runs immediately or as a ~50%-cheaper batch job — with
persistent, cron-resumable state and a deadline-aware fallback.

## Why

litellm exposes batch and interactive primitives but does not make them
interchangeable behind one call site, nor does it persist state for resumable,
cron-driven pipelines. `llm-batch` does both.

## Install

```bash
pip install llm-batch              # core (litellm)
pip install "llm-batch[postgres]"  # + PostgresStorage backend
```

## Quick start

```python
from llm_batch import Pipeline

pipeline = Pipeline(
    state_dir="./state/doc_abc/",
    mode="batch",                       # or "interactive"
    deadline="2026-06-11T10:00:00Z",    # batch falls back to interactive near the deadline
    rerun_cmd="python continue.py doc_abc",
)

# Step 1 — a bundle of requests advancing as one resumable unit.
ocr = pipeline.job("ocr", requests=[msgs_p1, msgs_p2], model="vertex_ai/gemini-2.5-flash")
pages = ocr.execute()                   # list[str] | None  (None while the batch runs)
if pages is None:
    raise SystemExit(1)                 # re-run later; poller advances it

# Step 2 — a single request.
analysis = pipeline.job("analysis", messages=build_msgs(pages), model="vertex_ai/gemini-2.5-flash")
result = analysis.execute()             # str | None
if result is None:
    raise SystemExit(1)

print(pipeline.cost_summary())          # {'charged_usd': ..., 'saved_usd': ...}
pipeline.finish()                       # delete local state files
```

`execute()` is idempotent: re-running submits what is unsent, polls what is
sent, and returns the stored result once complete. Exit codes for cron:
`0` done, `1` pending, `2` failed.

## Modes and the deadline fallback

- `mode="interactive"` — always call the model now (full price, instant).
- `mode="batch"` — submit as batch **unless** `deadline − now < max_batch_turnaround_hours`
  (default 24h), in which case it transparently runs interactively so the hard
  deadline is still met. Pass `time_budget_hours=` as sugar for a relative deadline.

## Cron / polling

Two patterns, one core:

1. **Self-resuming script** — your pipeline script is the cron line; each tick
   polls and advances.
2. **Central poller** — for fleets:

   ```bash
   llm-batch poll ./state                      # poll all SENT jobs, advance via rerun_cmd
   llm-batch poll ./state --sweep-orphans 3    # also delete provider orphans >3 days old
   ```

The library is **filesystem-only and never touches git** — you own
`git pull` / `commit` / `push` once per run. State files use deterministic JSON
(sorted keys, trailing newline) for clean diffs.

## Storage backends

- `FileStorage(dir)` — one atomic JSON file per job (default).
- `PostgresStorage(conn, namespace)` — JSONB rows for callers already on a DB.

## Provider-side cleanup

After a result is durably saved, provider-side batch artifacts are deleted by
default (`cleanup_provider_files=True`): OpenAI file objects (quota) and Vertex
GCS objects (billing). Gemini-API and Anthropic auto-expire (no-op).

See `examples/government_contracts/` for a full GitHub Actions deployment.
