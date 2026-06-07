# Transparent Interactive/Batch LLM Processing Library — Design

**Date:** 2026-06-07
**Status:** Approved design, ready for implementation planning
**Source task:** `docs/todo/llm-interactive-batch-lib.md`
**Prototype:** `tmp/llm_batch_processing.py` (`AsyncPrompt`)
**Provider research:** `docs/research/{openai,gemini,claude}-batch.md`

## 1. Problem

Pipelines that call LLMs often want to switch between:

- **Interactive mode** — immediate request → response (full price, instant).
- **Batch mode** — submit job → poll later → collect response (~50% cheaper, up to 24h, sometimes 48h end-to-end across a multi-step pipeline).

Today the calling code handles these two modes with different code paths, state files, and polling logic. litellm exposes batch and interactive primitives but does not make them interchangeable behind one call site, nor does it persist state for resumable, cron-driven pipelines.

This library makes the difference **transparent**: the caller flips one parameter (`mode`) and everything else — including resuming a multi-step pipeline from where it left off — stays the same.

## 2. Goals

1. **Single call site.** Calling code calls `job.execute()` identically regardless of mode.
2. **Idempotent.** Safe to call repeatedly (cron-friendly); re-running resumes from the last persisted state.
3. **Deadline-aware fallback.** When too little time remains before a hard deadline to risk another batch round, transparently fall back to interactive.
4. **Multi-step pipelines.** Each step is its own job; the pipeline resumes from the last completed step.
5. **Pluggable storage.** Default file-based (one JSON per job); optional Postgres backend.
6. **Provider-agnostic via litellm.** Works with any provider litellm's batch API supports (Vertex/Gemini, OpenAI, Anthropic).
7. **Caller stays out of provider plumbing.** Batch submission, polling, result parsing, and provider-side data cleanup are the library's job.

## 3. Core Model

### 3.1 `Job` — the unit of work and of resumable state

A `Job` wraps one *or many* requests, a model, a mode, and a storage handle. The caller decides granularity (one page = one job, or 50 pages = one job); **the library imposes no grouping policy** — grouping carries no pricing benefit (see §10), only caller-side convenience and provider limits.

### 3.2 `execute()` — one idempotent call that submits *and* polls

`execute()` is driven entirely by the job's persisted state:

| Current state | `execute()` does | Returns |
|---|---|---|
| `NOT_SENT` | Interactive: call now, store result → `COMPLETED`. Batch: create batch, store `batch_job_id` → `SENT`. | result, or `None` (batch) |
| `SENT` | Poll once. Completed → store result → `COMPLETED`. Still running → no change. Failed/expired → `FAILED`. | result, `None`, or raises |
| `COMPLETED` | Nothing (no API call). | stored result |
| `FAILED` | Nothing. | raises `BatchJobFailed` |

The call site is therefore identical for interactive and batch and safe to run repeatedly from cron:

```python
result = job.execute()   # str (single) | list[str] (bundle) | None (pending)
if result is None:
    sys.exit(1)          # re-run later
```

### 3.3 Job shape — single vs. bundle (overloaded)

```python
# Single request → str
job = pipeline.job("ocr_page_3", messages=[...], model="vertex_ai/gemini-2.5-flash")
text = job.execute()                 # str | None

# Bundle → list[str], advances as ONE resumable unit
job = pipeline.job("ocr_doc", requests=[msgs_p1, msgs_p2, ...], model="...")
pages = job.execute()                # list[str] | None  (None until ALL complete)
```

`messages=` and `requests=` are mutually exclusive. A bundle's `response` is `None` until every request in the batch has a result; then it is a `list[str]` aligned to submission order.

### 3.4 `Pipeline` — convenience over a shared state dir + config

`Pipeline` is a thin wrapper that removes boilerplate; a bare `Job` works standalone.

```python
pipeline = Pipeline(state_dir="./data/state/doc_abc/", mode="batch",
                    deadline="2026-06-11T10:00:00Z",
                    rerun_cmd="python continue.py doc_abc")

step1 = pipeline.job("ocr", requests=ocr_reqs, model="vertex_ai/gemini-2.5-flash")
if (r1 := step1.execute()) is None:
    return False                      # pending

step2 = pipeline.job("extract", messages=build_extract_msgs(r1), model="...")
if (r2 := step2.execute()) is None:
    return False                      # pending

pipeline.finish()                     # cleans up local state files
return True
```

- `pipeline.job(key, ...)` mints a `Job` whose key is namespaced under the pipeline and which inherits the pipeline's `mode`, `deadline`, `rerun_cmd`, `storage`, and `cleanup_provider_files`.
- The pipeline records `pipeline_started_at` **once** in a `pipeline.json` meta file — the anchor for deadline math (§5.3).
- `pipeline.finish()` deletes the pipeline's local state files.

## 4. Modes and Deadline-Aware Fallback

### 4.1 Two modes

`mode` is `"interactive"` or `"batch"`. There is no third "auto" mode — the deadline drives fallback within batch mode, which keeps the concept count minimal.

- `mode="interactive"` — always call the model directly.
- `mode="batch"` — submit as batch, **unless** the deadline check (below) says there is not enough time.

### 4.2 Mechanism in the library, values from the caller

The library owns the *mechanical* decision (it is the thing performing the submit); the caller owns the *numbers* (only it knows the domain).

At every batch submit, the library evaluates:

```
if (deadline - now) < max_batch_turnaround_hours:
    submit interactively instead of as batch
```

- `deadline` — absolute timestamp by which all processing must be done. The caller computes it from domain knowledge, e.g. `deadline = legal_signup_deadline − reviewer_reading_time`. `time_budget_hours` is accepted as sugar (`deadline = pipeline_started_at + time_budget_hours`).
- `max_batch_turnaround_hours` — conservative worst-case batch round duration (default **48**, configurable). Below this margin, batch is too risky.

The library never needs to know how many steps remain. It only ever asks *"room for one more batch round?"*. Once the budget tightens, **every** remaining step naturally flips to interactive — exactly the desired behavior for a multi-step pipeline approaching a hard deadline.

If neither `deadline` nor `time_budget_hours` is set, batch mode never falls back.

## 5. Storage

### 5.1 Pluggable backend interface

A minimal interface (preserved from the prototype):

```python
class Storage(Protocol):
    def load(self) -> bool: ...          # True if existing state was loaded
    def save(self, data: dict) -> None: ...
    def get(self, key, default=None): ...
    def finish(self) -> None: ...         # delete / mark finished
```

### 5.2 `FileStorage(dir)` — default

- One JSON file per job, named from the job key (e.g. `state/doc_abc/ocr.json`).
- One `pipeline.json` meta file per pipeline (`pipeline_started_at`, shared config).
- **Atomic writes** (write temp + `os.replace`) so a crash mid-write cannot corrupt state — important when poller and main script can both touch files.
- **Deterministic, merge-friendly serialization** (sorted keys, stable indentation, trailing newline) so the caller's once-per-run `git commit` produces clean diffs.

### 5.3 `PostgresStorage(ref)` — optional

The prototype's `(conn, entity_id, task_type)` upsert model, for callers already on a database.

### 5.4 Per-job state schema

```json
{
  "key": "doc_abc/ocr",
  "state": "sent",
  "mode": "batch",
  "model": "vertex_ai/gemini-2.5-flash",
  "n_requests": 50,
  "batch_job_id": "batches/abc123",
  "input_file_id": "files/xyz",
  "output_file_id": null,
  "submitted_at": "2026-06-05T10:00:00Z",
  "completed_at": null,
  "deadline": "2026-06-11T10:00:00Z",
  "response": null,
  "error_message": null,
  "rerun_cmd": "python continue.py doc_abc",
  "cleanup_provider_files": true
}
```

- `response` is `str` (single) or `list[str]` (bundle).
- `n_requests` lets the poller verify it received every result.
- `input_file_id` / `output_file_id` are recorded so provider-side cleanup (§7) can run later.
- `deadline` is persisted so the fallback decision survives across runs.

States: `not_sent` → `sent` → `completed` | `failed`.

## 6. Cron, Continuation, and Concurrency

Two patterns share one core. Both are supported.

### 6.1 Self-resuming script

The caller's pipeline script builds the pipeline and calls `execute()` on each step; it is idempotent. The cron line is literally the script. First run submits step 1 and exits pending; later runs poll, and when step 1 completes, submit step 2, and so on.

### 6.2 Central poller (`Poller` / `llm-batch poll`)

For fleets of independent pipelines. The poller:

1. Scans all `SENT` job files, retrieves each batch once, and updates state to `COMPLETED` / `FAILED`.
2. For each *pipeline* that had a job complete, execs that pipeline's configured `rerun_cmd` **once** (de-duped per pipeline) to advance it immediately — so a batch finishing at hour 3 advances then, not at the next daily run.

`rerun_cmd` must be an **explicit, environment-stable command** the caller configures (e.g. `"python continue.py doc_abc"`), pointing at a *continue-only* entrypoint (no scraping). It defaults to the captured argv for the simple local case but must be overridden for CI, where captured argv would not replay correctly.

### 6.3 Deployment shape (GitHub Actions + separate "data" git repo)

State lives in a separate git "data" repo; runs happen on GitHub Actions or locally. The library is **filesystem-only and never touches git** (per-job commits would shred history). The caller owns `git pull` at the start and `git commit && git push` at the end of each run — one commit per run.

- **Daily workflow:** `git pull` → `python scrape_and_process.py` (scrape new docs + start/advance pipelines) → `git commit && git push`.
- **Hourly workflow:** `git pull` → `llm-batch poll ./data/state` (poll + advance via `rerun_cmd`) → `git commit && git push`.

### 6.4 Concurrency

- The main (daily) and poller (hourly) runs do not normally touch the same job: the main script starts new work, the poller only advances already-`SENT` jobs.
- The only realistic collision is **poller-vs-poller overlap** (a poller run exceeding the hourly interval). This is eliminated by **GitHub Actions `concurrency:` groups** (one run per workflow at a time).
- **git push-with-rebase-retry** (caller-side) is the backstop for any remaining race.
- **No `flock` dependency in core.** An optional local-only file lock may be offered as a convenience for laptop runs.

## 7. Provider-Side Data Lifecycle and Cleanup

Batch input/output data is stored on provider servers (or, for Vertex, in the caller's own GCS bucket). Leftover data costs money or quota on the two paths the caller actually uses. See §10 for the research basis.

### 7.1 Completion cleanup

- `Job` / `Pipeline` accept `cleanup_provider_files` (**default `True`**).
- After a batch completes, the result is **first durably persisted to local state**, and **only then** are the provider-side input/output files deleted. A crash between the two can never lose data (next poll re-reads the still-present output; cleanup retries).
- Per provider (in the provider layer, §9):
  - **OpenAI** — delete input/output/error file objects (`DELETE /v1/files/{id}`); avoids the 2.5 TB project quota.
  - **Vertex AI** — delete the GCS input/output objects; avoids ongoing GCS storage billing.
  - **Gemini API / Anthropic** — no-op (data auto-expires; storage is free).
- Set `cleanup_provider_files=False` to retain raw provider artifacts for debugging.

### 7.2 Stale-job timeout (safety net for "never completes")

The poller treats any job stuck in `SENT` past `max_batch_age` (default **3 days**, configurable — well beyond every provider's 24h SLA) as dead: mark `FAILED`, attempt provider cleanup, and surface it. This catches the rare case where polling never resolves or the local record outlives a vanished batch. (Providers already mark batches `expired` after their window; `poll_batch` maps that to `FAILED` directly.)

### 7.3 Orphan sweep (optional)

`Poller.sweep_orphans(older_than_days=3)` lists provider-side batches/files and deletes those older than the threshold with no live local state — for the crash-after-submit-before-save case. Best-effort and provider-specific, so it is opt-in, not core.

## 8. Failure Handling

- `FAILED` is **sticky**: `execute()` on a failed job raises `BatchJobFailed` (carrying `batch_job_id` and the provider error) rather than silently retrying.
- Re-submitting is an explicit `job.retry()` (resets to `NOT_SENT`).
- **Bundle policy (MVP):** if any request in a batch errors, the whole job → `FAILED`, with per-`custom_id` detail in `error_message`. Per-request partial success is a noted future extension (§13), not MVP.
- Exit-code protocol (preserved from prototype, for cron): `0` done, `1` pending, `2` failed.

## 9. Provider Layer (litellm)

`providers.py` isolates everything litellm/provider-specific behind a small interface so `Job` stays provider-agnostic:

```python
def submit_batch(model, requests) -> BatchHandle      # build JSONL/inline, upload, create
def poll_batch(handle) -> (status, results | None)    # 'pending'|'completed'|'failed'
def complete(model, messages) -> str                  # interactive path
def cleanup(handle) -> None                           # delete provider-side files (§7)
def provider_of(model) -> str                         # model prefix → custom_llm_provider
```

- JSONL is built one line per request, `custom_id = "req-0".."req-N"` (matches prototype).
- Provider quirks surfaced by research live **only here**: Vertex-only batch for Gemini (the `gemini/` prefix has no litellm batch support), Anthropic's inline submission (no file object), OpenAI's file-upload flow.
- `provider_of` maps e.g. `vertex_ai/gemini-2.5-flash → "vertex_ai"`, `gpt-4o → "openai"`, `claude-* → "anthropic"`.

## 10. Provider Research Summary (basis for §4, §7)

Full detail in `docs/research/{openai,gemini,claude}-batch.md`.

| Provider | Batch discount | Tied to batch size? | Data storage | Auto-delete | Cleanup needed |
|---|---|---|---|---|---|
| OpenAI | ~50% flat per-token | No | OpenAI Files (free, 2.5 TB quota) | output ~30d | **Yes** (quota) |
| Gemini API (`gemini/`) | ~50% flat per-token | No | Google-managed files (free) | 48h | No — *but no litellm batch support* |
| Vertex AI (`vertex_ai/`) | ~50% flat per-token | No | **Caller's GCS bucket (billed)** | **Never** | **Yes** (cost) |
| Anthropic | ~50% flat per-token | No | Inline (free) | 29d | No |

Key conclusions: (1) the discount is a flat per-token reduction independent of batch size, so the library must not impose grouping; (2) the two usable batch paths (Vertex, OpenAI) accumulate billable/quota-limited data, justifying default-on cleanup.

## 11. CLI

```bash
llm-batch poll ./data/state            # poll all SENT jobs, advance via rerun_cmd
llm-batch poll ./data/state --sweep-orphans 3   # also sweep provider orphans >3d
```

## 12. Module Structure

```
llm_batch/
  __init__.py        # Job, Pipeline, Poller, FileStorage, PostgresStorage
  job.py             # Job + execute() state machine
  pipeline.py        # Pipeline + pipeline.json meta (pipeline_started_at, config)
  providers.py       # litellm batch/interactive, model→provider, cleanup
  storage/
    base.py          # Storage protocol
    file.py          # FileStorage (atomic, deterministic JSON)
    postgres.py      # PostgresStorage
  poller.py          # Poller + rerun_cmd dispatch, stale timeout, orphan sweep
  cli.py             # `llm-batch poll`
  errors.py          # BatchJobFailed, etc.
tests/
```

## 13. Testing Strategy

- **TDD with litellm fully mocked** — no network in unit tests.
- Coverage: state-machine transitions (`not_sent→sent→completed`, `→failed`), `execute()` idempotency (two calls = one submit), interactive path, deadline-fallback math, single-vs-bundle results, `FileStorage` atomic + deterministic round-trip, poller de-dup + `rerun_cmd` dispatch (subprocess mocked), stale-job timeout, completion cleanup ordering (persist before delete), failure → `FAILED`.
- **Live provider tests** are opt-in via an env var and skipped by default.

## 14. Interface Required by `pdf-to-text` (downstream consumer)

- `Job` / `Pipeline.job()` with `execute() → str | list[str] | None`.
- Deadline-aware batch→interactive fallback.
- Per-job idempotent file state (one file per page or per bundled step).
- `job.set_completed(text)` to mark a job pre-completed without an LLM call (e.g. when `fitz` extracts text directly and no LLM is needed).

## 15. Out of Scope

- Model cascade / confidence-based escalation (lives in the caller, e.g. pdf-to-text).
- Job priorities or queues.
- Parallel job execution (cron handles this by re-running the pipeline script).
- Library-managed git (caller owns pull/commit/push).

## 16. Future Extensions (noted, not MVP)

- Per-request partial success for bundles (return `list[str | None]` with per-`custom_id` status).
- A `GitStorage` backend, if a caller ever wants library-managed git despite the commit-chatter cost.
- Provider-side cleanup beyond Vertex/OpenAI if new providers expose billable batch storage.
