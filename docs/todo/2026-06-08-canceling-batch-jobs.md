# TODO: Speculative Batch with Late Cancellation (Phase 2)

**Status:** Deferred to a phase after the MVP. The MVP ships the `conservative` fallback
strategy only (decide once at submit time using `max_batch_turnaround_hours`); this document
captures the design for the `speculative` strategy so it can be implemented later.

**Related:** `docs/superpowers/specs/2026-06-07-llm-interactive-batch-lib-design.md` (§4 fallback,
§8 failure handling, §9 provider layer). Provider research with a "Batch Cancellation" section
each: `docs/research/{openai,gemini,claude}-batch.md`.

## Motivation

The MVP's `conservative` strategy falls back from batch to interactive whenever
`deadline − now < max_batch_turnaround_hours`. Because that margin must cover the *worst-case*
batch turnaround (up to 24–48h), it throws away the batch discount on every job submitted within
a day or two of the deadline — even though **most batches actually finish in far less than 24h**.

The `speculative` strategy submits the batch optimistically and only falls back at the last
responsible moment, letting the effective deadline drop to roughly
`poll_interval + cancel_margin + interactive_runtime_estimate` (≈1.5–2h with hourly polling)
instead of 24–48h. This captures the discount in the common (fast) case and pays full interactive
price only on the slow tail.

## Cancellation Algorithm (as specified)

At each poll of a `SENT` batch job under the `speculative` strategy:

1. If the batch is **completed** → use the result, mark `COMPLETED` (no cancellation needed).
2. Otherwise, evaluate the time budget. If we can still afford to wait for the next poll
   (`deadline − now ≥ poll_interval + cancel_margin + interactive_runtime_estimate + safety`),
   keep waiting → return pending.
3. If we **cannot** afford to wait, cancel:
   a. **Initiate cancel** at the provider.
   b. **Poll the provider job state** until it returns `canceled` **or** `finished`/`completed`.
   c. If it **finished** before the cancel took effect → **continue as if no cancellation**:
      use the returned result, mark `COMPLETED`. (We already paid the batch rate for it; the
      result is valid, so use it.)
   d. If it is **canceled** → **re-issue the request as an interactive call**, store that result,
      set `actual_mode = "interactive"`, mark `COMPLETED`.

This "initiate, then poll until canceled-or-finished" loop is the crux of why this is deferred:
it needs careful state handling and real testing against each provider's cancellation behavior.

## Cost Findings (already researched — see research files)

Submit-then-cancel is **essentially free** on all three providers:
- **No cancellation fee** anywhere.
- Billed **only for requests that already completed/succeeded** before the cancel took effect,
  at the 50% batch rate. A single-request job not yet complete → ~$0.
- Storage during the wait is negligible: OpenAI free (2.5 TB quota), Vertex GCS ~sub-cent for
  MB-scale JSONL over a day, Anthropic none (inline submission).

## Caveats / Risks to Handle

1. **Cancellation latency, no uniform SLA.**
   - OpenAI: up to ~10 min (`cancelling` → `cancelled`).
   - Anthropic: seconds–minutes; in-flight requests are allowed to finish.
   - **Vertex AI: best-effort, no documented bound** — the job may even complete despite the
     cancel. This is why step 3b polls until a terminal state and step 3c salvages a finished job.
   - ⇒ `cancel_margin` must be conservative and provider-aware (e.g. default ≥15–30 min).

2. **litellm cannot cancel Vertex or Anthropic batches.**
   - `litellm.cancel_batch` / `acancel_batch` support **OpenAI/Azure only**.
   - Vertex must be cancelled via the Vertex SDK
     (`aiplatform.gapic.JobServiceClient.cancel_batch_prediction_job`) or REST
     (`projects.locations.batchPredictionJobs.cancel`).
   - Anthropic must be cancelled via the native SDK (`client.messages.batches.cancel(batch_id)`)
     or `POST /v1/messages/batches/{id}/cancel`.
   - ⇒ The provider layer's `cancel()` must dispatch per provider; only OpenAI can delegate to
     litellm. This is the main reason for the extra implementation + testing.

## New Configuration Required

- `fallback_strategy: "conservative" | "speculative"` (MVP default `"conservative"`).
- `poll_interval` — expected time between poller runs (e.g. 1h); the speculative decision needs it.
- `interactive_runtime_estimate` — estimated time to run this job interactively (larger for big
  bundles); caller-configurable with a default.
- `cancel_margin` — conservative upper bound on cancellation latency; provider-aware default.

## Provider Layer Additions

- Add `cancel(handle) -> None` and reuse `poll_batch(handle)` to watch for `canceled`/`completed`.
- `cancel()` dispatch: OpenAI via litellm; Vertex via Vertex SDK; Anthropic via native SDK.
- Map provider cancel states to the algorithm's `canceled` / `finished` outcomes.

## Testing Notes

- Unit-test the speculative decision math (wait vs. cancel) with mocked clocks and `poll_interval`.
- Mock the "finished-before-cancel" race (step 3c) and the "canceled → interactive re-issue" path
  (step 3d).
- Live tests (opt-in) should exercise real cancellation latency per provider, especially Vertex's
  unbounded case.
