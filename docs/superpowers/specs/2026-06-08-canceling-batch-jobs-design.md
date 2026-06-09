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

## Cancellation Algorithm — non-blocking, scheduled-poll state machine

**Key constraint: never busy-wait.** The poller runs on a schedule (e.g. hourly via GitHub
Actions). Initiating a cancel and then looping in-process until the provider reports a terminal
state would burn Actions minutes idling. Instead, cancellation is **persisted as job state** and
resolved across *subsequent scheduled poll runs* — each poll run does bounded work and exits.

This adds one job state, `CANCEL_REQUESTED` (between `SENT` and a terminal state), plus a
`cancel_requested_at` timestamp. The state machine, evaluated once per poll run per job:

**A. Job is `SENT` (no cancel requested yet).**
1. Poll the provider. If **completed** → use the result, mark `COMPLETED`. Done.
2. Otherwise evaluate the time budget. If we can still afford to wait for the next scheduled poll
   (`deadline − now ≥ poll_interval + cancel_margin + interactive_runtime_estimate + safety`),
   return pending — re-check next run.
3. If we **cannot** afford to wait: **initiate cancel** via `litellm.acancel_batch`, set state to
   `CANCEL_REQUESTED`, stamp `cancel_requested_at`, and **return pending immediately** (do *not*
   wait for the cancel to take effect). The outcome is read on the next scheduled poll.

**B. Job is `CANCEL_REQUESTED` (cancel already initiated on an earlier run).** Poll the provider and
branch on its state:
- **completed** → the batch finished before the cancel took effect. **Salvage it**: use the result,
  mark `COMPLETED`. (We already paid the batch rate; the result is valid.)
- **cancelled** → **re-issue the request as an interactive call**, store that result, set
  `actual_mode = "interactive"`, mark `COMPLETED`.
- **still queued / in-progress / cancelling** (cancel not yet effective) → **return pending and wait
  for the next scheduled poll.** Do not re-issue interactively yet — the batch may still complete or
  finish cancelling, and re-issuing now risks a double charge / lost result. Seeing "queued" here
  usually means cancellation hasn't propagated yet (more likely when `poll_interval` is short, e.g.
  5 min; with hourly polling a terminal state is the common case).

So the three provider states observed on a `CANCEL_REQUESTED` job map directly to: *queued* → keep
waiting; *cancelled* → interactive re-issue; *completed* → salvage.

**Why this is deferred:** it needs careful state handling (the new `CANCEL_REQUESTED` state + the
finished-before-cancel race) and real testing against each provider's cancellation behavior. The
original "per-provider cancel dispatch" reason is now gone — litellm handles dispatch for
OpenAI/Azure/Vertex (see Architecture Decision).

**Bounding the wait.** A `CANCEL_REQUESTED` job that never reaches a terminal state would otherwise
poll forever. Bound it: if `now − cancel_requested_at` exceeds a `cancel_resolution_timeout`
(provider-aware; generous for Vertex's unbounded cancel), stop waiting on the cancel and re-issue
interactively so the caller still gets a result before/near the deadline. (If the batch later
completes, `cleanup()` still removes its artifacts.)

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
     cancel. This is why a `CANCEL_REQUESTED` job is resolved across scheduled polls (state B) and
     the completed branch salvages a finished job, with `cancel_resolution_timeout` as the backstop.
   - Anthropic: not applicable — we never cancel Anthropic (it uses conservative). Latency recorded
     in research only: seconds–minutes, in-flight requests allowed to finish.
   - ⇒ `cancel_margin` must be conservative and provider-aware (e.g. default ≥15–30 min). Because we
     never busy-wait, the practical resolution granularity is also bounded below by `poll_interval`.

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
- `cancel_resolution_timeout` — provider-aware backstop for a `CANCEL_REQUESTED` job that never
  reaches a terminal state. Once `now − cancel_requested_at` exceeds it, stop waiting on the cancel
  and re-issue interactively (state B). Generous for Vertex (unbounded cancel); tighter for OpenAI.

## Provider Layer Additions

- Add `provider_supports_cancel(provider) -> bool` backed by
  `_CANCELLABLE = {"openai", "azure", "vertex_ai"}`. The strategy selector consults this to gate
  the speculative path (and to trigger the one-time Anthropic→conservative downgrade warning).
- Add `cancel(handle) -> None` that **only initiates** the cancel via
  `litellm.acancel_batch(batch_id=handle.batch_job_id, custom_llm_provider=handle.provider)` and
  returns — **no per-provider SDK branching, and no wait loop**. The outcome is read by a later
  scheduled `poll_batch`.
- **`poll_batch` must distinguish `canceled` from `failed`.** Today it folds `cancelled`/`canceling`
  into `_FAILED_STATUSES` (returns `failed`), which is correct for a non-speculative job but wrong
  for a `CANCEL_REQUESTED` one, where `canceled` is the *expected* outcome that triggers interactive
  re-issue. Add a distinct `canceled` poll status (or a speculative-aware poll) so state B can branch
  `queued → wait`, `canceled → re-issue`, `completed → salvage`. `canceling`/`queued`/`in_progress`
  stay non-terminal (keep waiting).
- Vertex's cancel is best-effort with no bound (it may even complete despite the cancel), so the
  cross-poll resolution (state B) and the finished-before-cancel salvage remain essential — and the
  `cancel_resolution_timeout` backstop bounds the pathological never-terminal case.

## Dependency: litellm Vertex-cancel capability

Vertex `cancel_batch` support (PR #23957) first shipped in **litellm 1.83.0** (PyPI, 2026-03-31);
all 1.82.x and earlier lack it. The dependency floor in `pyproject.toml` is therefore
`litellm>=1.83.0`. As defence-in-depth, the speculative path may still verify the capability at
startup — assert `litellm.acancel_batch` accepts `custom_llm_provider="vertex_ai"` — and, if a
patched/older litellm is somehow present, treat Vertex like Anthropic (no speculative; conservative
fallback) rather than erroring.

## Testing Notes

- Unit-test the state-A decision math (wait vs. cancel) with mocked clocks and `poll_interval`.
- Unit-test the state-B branching across separate poll runs: `completed` → salvage; `cancelled` →
  interactive re-issue (`actual_mode="interactive"`); `queued`/`canceling` → stays pending and waits
  for the next run.
- Assert **no busy-wait**: `cancel()` initiates and returns; a single poll run does bounded work and
  does not loop until terminal. (e.g. spy on the provider client — at most one state read per job
  per run.)
- Unit-test the `cancel_resolution_timeout` backstop: a `CANCEL_REQUESTED` job stuck non-terminal
  past the timeout re-issues interactively rather than waiting forever.
- Unit-test that `poll_batch` reports `canceled` distinctly from `failed` for a cancelled batch.
- Live tests (opt-in) should exercise real cancellation latency per provider, especially Vertex's
  unbounded case.
- Unit-test the provider gating: `provider_supports_cancel` returns the right set, an Anthropic job
  under `speculative` is downgraded to `conservative`, and the downgrade WARNING is emitted exactly
  once (not per poll).
- Unit-test that `cancel()` calls `litellm.acancel_batch` with the right `custom_llm_provider` and
  does not import any native provider SDK.
