# Transparent Interactive/Batch LLM Processing Library

## Problem

When building pipelines that call LLMs, you often want to switch between:
- **Interactive mode**: immediate request → response (high cost, instant result)
- **Batch mode**: submit job → poll later → collect response (50% cheaper, up to 24h wait)

The calling code is currently forced to handle these two modes explicitly, with different
code paths, state files, and polling logic. This library should make the difference
**transparent** — the caller changes one parameter and everything else stays the same,
including resuming multi-step pipelines from where they left off.

## Existing Prototype

`docs/tmp/pdf-to-text/` contains `llm_batch_processing.py` (used by `pdf_to_txt_llm.py`)
which implements a basic version of this as `AsyncPrompt`. Key patterns to preserve:

- State machine: `NOT_SENT_YET → SENT → COMPLETED | FAILED`
- Per-job JSON state file with `batch_job_id`, `submitted_at`, `response`, `rerun_cmd`
- `send()` submits (or executes immediately in interactive mode)
- `gather_response()` polls once; caller retries via cron
- `set_completed(text)` shortcircuits the LLM when fitz can extract text directly
- Exit code protocol: `0` = done, `1` = pending, `2` = failed

The new library should generalise this pattern.

## Core Design Goals

1. **Single call-site**: calling code calls `job.execute()` the same way regardless of mode
2. **Idempotent**: safe to call repeatedly (cron-friendly); re-running picks up from last state
3. **Time-budget-aware**: if remaining time < threshold, fall back from batch to interactive
4. **Multi-step pipelines**: each step is its own job; pipeline resumes from the last completed step
5. **Pluggable storage**: default file-based (one JSON per job); optional DB backend
6. **litellm-based**: works with any provider supported by litellm (Gemini, OpenAI, Anthropic, etc.)

## Proposed Interface

```python
from llm_batch import Job, Pipeline

# Single job
job = Job(
    key="my_pipeline/step_ocr",        # unique stable key for state persistence
    messages=[{"role": "user", "content": [...]}],
    model="vertex_ai/gemini-2.5-flash",
    mode="batch",                       # "interactive" | "batch" | "auto"
    time_budget_hours=24,               # optional; enables batch→interactive fallback
    storage=FileStorage("./state/"),    # pluggable
)

result = job.execute()   # str if completed, None if still pending

if result is None:
    sys.exit(1)          # re-run later (cron job)

# Multi-step pipeline
pipeline = Pipeline(state_dir="./state/doc_abc/", mode="batch", time_budget_hours=24)

step1 = pipeline.job("ocr", messages=ocr_messages, model="vertex_ai/gemini-2.5-flash")
if step1.execute() is None:
    return False   # pending

step2 = pipeline.job("extract", messages=build_extract_messages(step1.result),
                     model="vertex_ai/gemini-2.5-flash")
if step2.execute() is None:
    return False   # pending

# All steps done
pipeline.finish()   # cleans up state files
return True
```

## Time-Budget Logic

```
time_remaining = time_budget - (now - pipeline.started_at)

if time_remaining < min_batch_turnaround_estimate:
    # not enough time for another batch round → fall back to interactive
    job.mode = "interactive"
```

`min_batch_turnaround_estimate` should be configurable (default: 1–2h as a conservative
estimate). This enables the pattern: "first batch round ready in 2h → still 22h left →
submit second round as batch too; first batch ready in 23h → only 1h left → do second
round interactively."

## State File Schema

```json
{
  "key": "pipeline_abc/step_ocr",
  "state": "sent",
  "mode": "batch",
  "model": "vertex_ai/gemini-2.5-flash",
  "batch_job_id": "batches/abc123",
  "submitted_at": "2026-06-05T10:00:00Z",
  "time_budget_hours": 24,
  "pipeline_started_at": "2026-06-05T10:00:00Z",
  "completed_at": null,
  "response": null,
  "rerun_cmd": "python manager.py ./docs ./state"
}
```

## Polling / Cron

The library should provide a poller that can be run as a cron job:

```python
from llm_batch import Poller

poller = Poller(storage=FileStorage("./state/"))
poller.poll_all()   # checks all SENT jobs, updates state for completed ones
```

Or via CLI:
```bash
llm-batch poll ./state/
```

After polling, the parent process re-runs its pipeline script, which resumes from
the next incomplete step.

## Providers and Batch API Support

litellm's batch API (`create_batch`, `retrieve_batch`) is supported for:

| Mode        | Model prefix                  | Auth                          |
|-------------|-------------------------------|-------------------------------|
| Interactive | `gemini/gemini-2.5-flash`     | `GEMINI_API_KEY`              |
| Batch       | `vertex_ai/gemini-2.5-flash`  | GCP service account           |
| Batch       | `gpt-4o`                      | `OPENAI_API_KEY`              |
| Batch       | `claude-opus-4-7`             | `ANTHROPIC_API_KEY`           |

The direct `gemini/` prefix does not support batch in litellm — only `vertex_ai/` does.

## Interface Required by pdf-to-text

The `pdf-to-text` library (see `docs/superpowers/specs/` once written) will depend on
this library and needs the following from it:

- `Job` (or `Pipeline.job()`) with `execute() → str | None`
- Time-budget-aware batch→interactive fallback
- Per-job idempotent state (file-based, one file per page per pipeline step)
- A way to mark a job as pre-completed without an LLM call (`job.set_completed(text)`)
  for the case where fitz extracts text directly and no LLM is needed

## Out of Scope

- Model cascade / confidence-based escalation (this lives in the calling code, e.g. pdf-to-text)
- Job priorities or queues
- Parallel job execution (cron handles this naturally by re-running the pipeline script)
