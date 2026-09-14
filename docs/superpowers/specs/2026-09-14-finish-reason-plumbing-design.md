# Design: surface provider finish_reason for batch & interactive completions

**Date:** 2026-09-14
**Status:** Approved

## Summary

A consumer of this library (pdfmux, see its own spec:
`pdfmux/docs/superpowers/specs/2026-09-14-llm-batch-output-cap-design.md`) needs to detect
when a vision-to-markdown generation was cut off by the provider's own output-token limit,
so it can flag the affected page rather than silently accepting truncated text.

This library already lets callers cap generation via arbitrary litellm-style kwargs
(`max_tokens`, `thinking`, etc.) forwarded end-to-end from `Pipeline.job()` →
`Job.__init__(**litellm_kwargs)` → `providers.submit_batch()`/`providers.complete()` — no
change needed there (Section 1 confirms this and adds regression coverage). What's missing
is visibility into the provider's `finish_reason` on the response, which tells the caller
whether generation completed normally (`"stop"`) or was truncated by the cap (`"length"`).

This spec adds `finish_reason` plumbing through `providers.PollResult`/`CompletionResult`
and `Job`, mirroring the existing `.usage`/`.cost` pattern exactly. It changes no runtime
behavior — no retries, no new failure states, no new required kwargs — it only makes an
existing provider signal readable by callers.

**Explicitly out of scope**: any pdfmux-specific behavior (hardcoded `max_tokens`/`thinking`
values, `PageResult.truncated`, `.log.json` sidecar fields, CLI output). Those live entirely
in pdfmux's own spec and plan; this library stays provider-agnostic and pdfmux-unaware.

---

## Section 1: kwarg passthrough already exists — no production change needed

`Pipeline.job(key, *, messages=None, requests=None, model=None, **litellm_kwargs)` and
`Job.__init__(..., **litellm_kwargs)` already accept and persist arbitrary litellm-style
generation kwargs in job state (`_PERSISTED_FIELDS` → `litellm_kwargs`, restored on
resume). `providers.submit_batch()`/`_build_jsonl()` already forward those kwargs into
each request's body, and `providers.complete()` already forwards them directly into
`litellm.completion(...)`.

A caller wanting to cap generation (e.g. a consumer passing `max_tokens=4000,
thinking={"type": "disabled"}` into `Pipeline.job(...)`) needs no library change to do so
today. This spec adds a regression test proving that end-to-end (Section 3) but touches no
production code for this part — it exists purely so this behavior doesn't silently regress
while `finish_reason` plumbing (Section 2) is being added nearby.

## Section 2: PollResult / CompletionResult / Job gain finish_reason

- `PollResult` gains `finish_reason: str | None = None`. A batch job through this library is
  always single-request per every known caller's usage today, so a scalar suffices;
  bundle/multi-request jobs elsewhere in the library are unaffected by this field — it is
  simply `None` for a bundle response until/unless a future consumer needs per-item values.
  Populated in `poll_batch()` from the completed batch's response body
  (`body["choices"][0].get("finish_reason")`) only when there is exactly one result in the
  batch; `None` for zero or multiple results.
- `CompletionResult` gains `finish_reason: str | None = None`, populated the same way from
  the interactive completion response's `choices[0].finish_reason` (via `getattr`, since
  some providers/litellm versions may not report one).
- `Job` gains a `.finish_reason` property, mirroring the existing `.usage`/`.cost` pattern
  exactly: reads `self._state.get('finish_reason')`, returns `None` when absent or the job
  hasn't completed (including old/partial persisted state written before this field
  existed). Set in every completion path:
  - `_finish_completed()` (the batch-poll completion path) — from `PollResult.finish_reason`.
  - `_submit_interactive()` (mode="interactive") — from the last `CompletionResult`'s
    `finish_reason`, or `None` when the job is a bundle (multiple requests), since the field
    is scalar-only.
  - `_reissue_interactive()` (the speculative-cancel fallback path) — same rule.
  - `_base_state()` initializes `"finish_reason": None` so a SENT-but-not-yet-completed job
    reads `None` rather than a missing key.

## Section 3: testing

`tests/test_providers.py`:
- `PollResult`/`CompletionResult` default `finish_reason` to `None`.
- `complete()` returns `finish_reason` from the provider response; `None` when the response
  doesn't report one.
- `poll_batch()` on a completed single-request batch returns the request's `finish_reason`.
- `poll_batch()` on a completed multi-request (bundle) batch returns `finish_reason=None`.
- `poll_batch()` returns `finish_reason=None` when the response body has no
  `"finish_reason"` key at all (older/partial provider response shape).

`tests/test_job.py` (mirroring the existing `.usage`/`.cost` tests):
- `Job.finish_reason` is `None` before completion.
- `Job.finish_reason` returns the stored value after a completed batch poll.
- `Job.finish_reason` returns the stored value after a completed interactive completion
  (non-bundle).
- `Job.finish_reason` is `None` after a completed bundle (multi-request) job.
- `Job.finish_reason` is `None` when the state has no `finish_reason` key (older/partial
  state, or a provider that never reported one).
- A `max_tokens`/`thinking` kwarg passed to `Job(...)` — and via `Pipeline.job(...)` —
  appears in the request body the fake `submit_batch`/`complete` receives (Section 1's
  regression coverage; extends `fake_providers` to capture kwargs).

All new/changed tests follow the existing RED→GREEN TDD discipline used throughout this
library.
