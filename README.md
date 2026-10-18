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

### Fallback strategies

`fallback_strategy` (on `Pipeline`/`Job`) controls how a near-deadline batch job avoids missing its deadline:

- `"conservative"` (default) — decide once at submit time: if `deadline − now < max_batch_turnaround_hours`, run interactively instead of batching. Safe, but forgoes the batch discount within a day or two of the deadline.
- `"speculative"` — submit the batch optimistically and only cancel + re-run interactively at the last responsible moment, dropping the effective margin to roughly `poll_interval + cancel_margin + interactive_runtime_estimate`. Captures the discount in the common (fast) case.

Speculative cancellation is delegated to litellm and supported for **OpenAI, Azure, and Vertex AI** (requires `litellm>=1.83.0`). For **Anthropic**, which litellm cannot cancel, a speculative job transparently degrades to `conservative` (with a one-time warning).

Tuning knobs (all hours): `poll_interval_hours`, `interactive_runtime_estimate_hours`, `cancel_margin_hours` (provider-aware default), `cancel_resolution_timeout_hours` (provider-aware backstop for a cancel that never settles).

### Per-request results

`Job.items` (after completion) lists one dict per request, in request order, for both
single and bundle jobs:
`{"text", "finish_reason", "usage": {"prompt_tokens", "completion_tokens", "total_tokens",
"cached_tokens"}, "cost": {"charged_usd", "interactive_usd", "saved_usd"}}`.
`cached_tokens` is the provider-reported cached prompt count
(`prompt_tokens_details.cached_tokens`). How cached tokens are priced inside batch jobs is set
by `providers.BATCH_CACHE_POLICY` (`batch_only` | `stack` | `cache_precedence`).

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

## Vertex AI setup

Vertex AI is the recommended provider for trying out batch mode: it's one of
the few providers litellm can both submit **and** cancel batch jobs for
(alongside OpenAI/Azure), which makes it the best target for exercising
`fallback_strategy="speculative"`.

Prerequisite: the GCP project must already have billing enabled
(`gcloud billing projects describe "$VERTEXAI_PROJECT"` to check;
`gcloud billing projects link` to fix).

```bash
export GOOGLE_APPLICATION_CREDENTIALS=~/llm-batch-test-key.json
export VERTEXAI_PROJECT=playgrounds-1234567890123
export VERTEXAI_LOCATION=us-central1
export GCS_BUCKET_NAME=llm-batch-test-bucket   # required: litellm stages batch I/O here
VERTEXAI_USER=llm-batch-tester

gcloud auth login
gcloud config set project "$VERTEXAI_PROJECT"
gcloud services enable aiplatform.googleapis.com storage.googleapis.com

gcloud iam service-accounts create "$VERTEXAI_USER" \
    --display-name="llm-batch testing"

gcloud projects add-iam-policy-binding "$VERTEXAI_PROJECT" \
  --member="serviceAccount:$VERTEXAI_USER@$VERTEXAI_PROJECT.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding "$VERTEXAI_PROJECT" \
  --member="serviceAccount:$VERTEXAI_USER@$VERTEXAI_PROJECT.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"

gcloud iam service-accounts keys create "$GOOGLE_APPLICATION_CREDENTIALS" \
  --iam-account="$VERTEXAI_USER@$VERTEXAI_PROJECT.iam.gserviceaccount.com"

gcloud storage buckets create "gs://$GCS_BUCKET_NAME" \
  --location="$VERTEXAI_LOCATION" --project="$VERTEXAI_PROJECT"

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install git+https://github.com/michalfapso/llm-batch-processor.git
pip install google-cloud-aiplatform google-cloud-storage
```

The downloaded service account key is a live credential — keep it out of git
(`.gitignore` it, never commit it).

### Known litellm issue: `Message` not fully defined

As of `litellm==1.97.0` (the latest release at time of writing), `litellm/types/utils.py`
is missing an import of `ChatCompletionReasoningSummaryTextBlock`, which breaks
`Message` under current `pydantic` releases with:

```
pydantic.errors.PydanticUserError: `Message` is not fully defined; you should
define all referenced types, then call `Message.model_rebuild()`.
```

This is unrelated to your GCP/pydantic/openai package versions — it reproduces
identically across pydantic 2.10–2.13. A fix is already up as
[litellm PR #36387](https://github.com/BerriAI/litellm/pull/36387), not yet
merged/released. Until it ships, work around it by running this once at the
top of any script that imports `llm_batch` or `litellm`:

```python
import litellm.types.utils as _litellm_utils
from litellm.types.llms.openai import ChatCompletionReasoningSummaryTextBlock as _RSTB
_litellm_utils.ChatCompletionReasoningSummaryTextBlock = _RSTB
_litellm_utils.Message.model_rebuild(force=True)
```

Remove this workaround once you're on a litellm release that includes the fix.

### Smoke test

```python
from llm_batch import Pipeline

pipeline = Pipeline(state_dir="./state/smoke_test/", mode="batch")
job = pipeline.job("smoke", messages=[{"role": "user", "content": "Say hi in one word."}],
                    model="vertex_ai/gemini-2.5-flash")
result = job.execute()
print(result)
if result is None:
    print("Batch submitted, not yet complete — re-run this script to poll.")
    raise SystemExit(1)   # don't call pipeline.finish() until the job is actually done

print("Done. Cost summary:", pipeline.cost_summary())
pipeline.finish()
```

Re-run the script to poll; Vertex batch jobs target a 24h SLO but small jobs
often finish in minutes. `charged_usd` in the cost summary should come out to
roughly half of `interactive_usd`, confirming the batch discount applied.
