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

Speculative cancellation is delegated to litellm and supported for **OpenAI, Azure, and Vertex AI** (requires `litellm>=1.83.0`). For **Anthropic**, which litellm cannot cancel, and **OpenRouter**, whose Batch API has no cancel endpoint, a speculative job transparently degrades to `conservative` (with a one-time warning).

Tuning knobs (all hours): `poll_interval_hours`, `interactive_runtime_estimate_hours`, `cancel_margin_hours` (provider-aware default), `cancel_resolution_timeout_hours` (provider-aware backstop for a cancel that never settles).

### Abandoning jobs

`Job.abandon(reason)` cancels a live batch the caller no longer wants, and `Pipeline.abandon_all(reason)` does so for every job in the pipeline (returning the keys a provider cancel was issued for). Per state:

- `SENT` — asks the provider to cancel (OpenAI, Azure, Vertex AI) and marks the job `FAILED` locally with `reason`; returns `True`. For providers that cannot cancel (Anthropic, OpenRouter) it only marks the job `FAILED` and returns `False`.
- `CANCEL_REQUESTED` — the cancel was already issued; marks the job `FAILED` and returns `True`.
- `NOT_SENT`, `COMPLETED`, `FAILED` — no-op, returns `False`. A `COMPLETED` job's result is never altered.

This is best-effort and fire-and-forget: the provider may still finish (and bill) a batch if the cancel does not take effect, and Anthropic batches cannot be cancelled at all. If the provider cancel raises (e.g. `TransientProviderError`), the exception propagates and the job stays `SENT` so the call can be retried.

### Per-request results

`Job.items` (after completion) lists one dict per request, in request order, for both
single and bundle jobs:
`{"text", "finish_reason", "usage": {"prompt_tokens", "completion_tokens", "total_tokens",
"cached_tokens"}, "cost": {"charged_usd", "interactive_usd", "saved_usd"}}`.
`cached_tokens` is the provider-reported cached prompt count
(`prompt_tokens_details.cached_tokens`). How cached tokens are priced inside batch jobs is set
by `providers.BATCH_CACHE_POLICY` (`batch_only` | `stack` | `cache_precedence`); an unknown value raises `ValueError`.
With `batch_only` and heavy caching, `saved_usd` can be negative: the interactive reference
price is cache-discounted while the batch charge is not.

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
default (`cleanup_provider_files=True`): OpenAI file objects (quota), Vertex
GCS objects (billing) and OpenRouter batches (stored requests and results).
Gemini-API and Anthropic auto-expire (no-op).

See `examples/government_contracts/` for a full GitHub Actions deployment.

## OpenRouter batch

`model="openrouter/<vendor>/<model>"` with `mode="batch"` submits through
OpenRouter's own Batch API (direct HTTP; litellm has no OpenRouter batch
support). Use the model's normal slug, e.g. `openrouter/anthropic/claude-haiku-5.5`,
so the same string also works for the interactive deadline fallback. The model
needs a batch variant ([list](https://openrouter.ai/models?variant=batch)); one
without it fails at submit with `400`. Research and probe results are in
`docs/research/openrouter-batch.md`.

- **Env:** `OPENROUTER_API_KEY` (required). `OPENROUTER_API_BASE` is optional and
  defaults to `https://openrouter.ai/api/v1`.
- **Cost:** `cost_source="provider"`. `charged_usd` is what OpenRouter billed
  (the batch's `usage.cost`), split across requests by token share. `interactive_usd`
  is estimated as `charged / providers.BATCH_RATES["openrouter"]` (0.5, OpenRouter's
  "typically 50%"; a few models differ).
- **No cancel:** `speculative` degrades to `conservative`, and `abandon()` cannot stop
  a running batch, which is still billed.
- **No inline images:** base64/data-URI images are rejected (the batch ends
  `failed`). Send public `https` image URLs, which the upstream provider fetches
  when it runs the batch (possibly hours later), or pass an `image_host` (below).
- **Limits:** one model per batch, 200 MB body (`413`). Per-account request and
  in-progress caps are not published (`413`/`429`, retried as transient).
  `completion_window` is always `24h`, but Google Vertex can take up to 96h,
  which is longer than the poller's default `max_batch_age_days=3`.
- **Errors:** `429`/`5xx`/network are transient. `401`/`402` (bad key, not enough
  credits) raise and leave the job's state unchanged; top up and re-run. A `404`
  on poll is retried (a just-created batch can briefly 404). `410` (past the
  30-day retention) fails the job.
- `llm-batch poll --sweep-orphans N` deletes terminal OpenRouter batches older
  than N days that have no local state, when the state tree has OpenRouter jobs.
  Every key in a workspace sees the same batches, so this can delete batches
  that other tools in the workspace submitted.
- **Image hosting:** with `image_host=` on `Pipeline`, `Job` or `Poller`, an
  OpenRouter batch submit stores every data-URI `image_url` on the host and sends
  its URL instead. The interactive path and other providers keep the inline
  images. The images are deleted once the batch has completed or failed (also
  when a failed submit leaves the job unsent). Ids are random and unguessable,
  and the URL is the only access control. The image ids are in the job state
  (`hosted_images`); `Poller.sweep_orphans` also deletes old images that no
  running job needs (for example ones left by a crash before submit).

  ```python
  from llm_batch import MongoImageHost, S3ImageHost

  # MongoDB (`pip install "llm-batch[mongo]"`): documents {_id, content_type, data, created_at}
  # in `llm_images`. Your server must serve GET <public_base_url>/<id> from them.
  host = MongoImageHost(pymongo.MongoClient(uri).wd, "https://example.org/api/v1/llm-images")
  host.ensure_indexes(ttl_days=10)    # TTL backstop; keep it above the batch turnaround

  # A publicly readable S3 bucket (`pip install "llm-batch[s3]"`), e.g. MinIO:
  host = S3ImageHost(boto3.client("s3", endpoint_url=...), "bucket", "https://asset.example.org")

  pipeline = Pipeline("./state", mode="batch", image_host=host)
  ```

  The URL has to be reachable from the internet. `localhost` in development is not.
- Live test (opt-in, ~$0.000002): `LLM_BATCH_LIVE=1 OPENROUTER_API_KEY=… pytest tests/test_openrouter_live.py`.

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
