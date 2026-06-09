# TODO: Speculative Batch with Late Cancellation (Phase 2)

**Status:** Deferred to a phase after the MVP. The MVP ships the `conservative` fallback
strategy only (decide once at submit time using `max_batch_turnaround_hours`); this document
captures the design for the `speculative` strategy so it can be implemented later.

**Related:** `docs/superpowers/specs/2026-06-07-llm-interactive-batch-lib-design.md` (§4 fallback,
§8 failure handling, §9 provider layer). Provider research with a "Batch Cancellation" section
each: `docs/research/{openai,gemini,claude}-batch.md`.

## Architecture Decision: Cancellation via litellm (2026-06-09)

Cancellation is **delegated entirely to litellm**; we add **no native provider-SDK cancel code**
to this repo. `providers.py` stays litellm-only.

- litellm's `cancel_batch` / `acancel_batch` now route **OpenAI, Azure, and Vertex AI**. Vertex was
  added by merged PR https://github.com/BerriAI/litellm/pull/23957 (merge commit `8d843fd`). This
  closes the Vertex gap the original draft of this doc called out.
- **Anthropic batch cancellation is not supported in litellm** and nobody upstream is working on it
  (issues + PRs checked 2026-06-09: `cancel_batch` routes only OpenAI/Azure/Vertex; Anthropic's
  handler implements `retrieve_batch` only, no cancel). We are **not** contributing it upstream as
  part of this work.
- ⇒ The `speculative` strategy is offered **only for cancellable providers**
  (`_CANCELLABLE = {"openai", "azure", "vertex_ai"}`). For Anthropic, a job configured with
  `fallback_strategy="speculative"` **transparently degrades to `conservative`** (the existing
  24h-window fallback) and emits a **one-time WARNING** the first time a downgrade happens, so the
  user knows speculative is not taking effect for that provider. Mixed-provider runs therefore work
  uncomplaining: speculative applies where supported, conservative elsewhere.

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
(The original "per-provider cancel dispatch" reason is now gone — litellm handles dispatch for
OpenAI/Azure/Vertex; see Architecture Decision. What remains is the state-machine + race handling.)

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
   - **Vertex AI: best-effort, no documented bound** — the job may even complete despite the
     cancel. This is why step 3b polls until a terminal state and step 3c salvages a finished job.
   - Anthropic: not applicable — we never cancel Anthropic (it uses conservative). Latency recorded
     in research only: seconds–minutes, in-flight requests allowed to finish.
   - ⇒ `cancel_margin` must be conservative and provider-aware (e.g. default ≥15–30 min).

2. **Anthropic cannot be cancelled via litellm (so speculative is unavailable there).**
   - litellm's `cancel_batch` / `acancel_batch` route **OpenAI, Azure, and Vertex AI** only.
   - Cancelling Anthropic would require native-SDK code (`client.messages.batches.cancel(batch_id)`
     / `POST /v1/messages/batches/{id}/cancel`), which we have decided **not** to add (see
     Architecture Decision above).
   - ⇒ No per-provider dispatch in our `cancel()`: it is a thin `litellm.acancel_batch` wrapper, and
     Anthropic jobs simply never enter the speculative path.

## New Configuration Required

- `fallback_strategy: "conservative" | "speculative"` (MVP default `"conservative"`). Global knob,
  not per-call. When `"speculative"` is set, it only takes effect for cancellable providers; a job
  on a non-cancellable provider (Anthropic) is downgraded to `"conservative"` with a one-time
  WARNING (see Architecture Decision).
- `poll_interval` — expected time between poller runs (e.g. 1h); the speculative decision needs it.
- `interactive_runtime_estimate` — estimated time to run this job interactively (larger for big
  bundles); caller-configurable with a default.
- `cancel_margin` — conservative upper bound on cancellation latency; provider-aware default, and
  only consulted for cancellable providers. OpenAI ~10 min; Vertex conservative (≥15–30 min) given
  its best-effort/unbounded cancel.

## Provider Layer Additions

- Add `provider_supports_cancel(provider) -> bool` backed by
  `_CANCELLABLE = {"openai", "azure", "vertex_ai"}`. The strategy selector consults this to gate
  the speculative path (and to trigger the one-time Anthropic→conservative downgrade warning).
- Add `cancel(handle) -> None` that delegates to
  `litellm.acancel_batch(batch_id=handle.batch_job_id, custom_llm_provider=handle.provider)` —
  **no per-provider SDK branching**. Reuse the existing `poll_batch(handle)` to watch for
  `canceled`/`completed`. (`poll_batch` already maps `cancelled`/`canceling` into `_FAILED_STATUSES`;
  the speculative loop reads that plus `completed` to drive steps 3b–3d.)
- Map provider cancel states to the algorithm's `canceled` / `finished` outcomes. Vertex's cancel is
  best-effort with no bound (it may even complete despite the cancel), so the "poll until
  canceled-or-finished" loop and the finished-before-cancel salvage (algorithm step 3c) remain
  essential — delegation to litellm does not remove this requirement.

## Dependency: litellm Vertex-cancel capability

The installed litellm must include PR #23957 (Vertex `cancel_batch` support). Rather than pin a
brittle version, verify the capability at startup/CI: assert `litellm.acancel_batch` accepts
`custom_llm_provider="vertex_ai"` without raising "unsupported provider". If it does not, treat
Vertex like Anthropic (no speculative; conservative fallback) until litellm is upgraded.

## Testing Notes

- Unit-test the speculative decision math (wait vs. cancel) with mocked clocks and `poll_interval`.
- Mock the "finished-before-cancel" race (step 3c) and the "canceled → interactive re-issue" path
  (step 3d).
- Live tests (opt-in) should exercise real cancellation latency per provider, especially Vertex's
  unbounded case.
- Unit-test the provider gating: `provider_supports_cancel` returns the right set, an Anthropic job
  under `speculative` is downgraded to `conservative`, and the downgrade WARNING is emitted exactly
  once (not per poll).
- Unit-test that `cancel()` calls `litellm.acancel_batch` with the right `custom_llm_provider` and
  does not import any native provider SDK.
