# Speculative Batch with Late Cancellation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `speculative` fallback strategy that submits a batch optimistically and, only when the deadline forces it, cancels via litellm and re-runs the request interactively — captured across scheduled poll runs so the poller never busy-waits.

**Architecture:** A new persisted `Job` state `CANCEL_REQUESTED` sits between `SENT` and a terminal state. State-A (`SENT` under speculative) decides each poll whether it can still afford to wait; when it can't, it initiates `litellm.cancel_batch` and returns immediately. State-B (`CANCEL_REQUESTED`) resolves the outcome on later poll runs: `completed` → salvage the batch result, `cancelled` → re-issue interactively, still-queued → keep waiting (bounded by `cancel_resolution_timeout`). Cancellation is delegated entirely to litellm (OpenAI/Azure/Vertex); Anthropic isn't cancellable, so a speculative Anthropic job degrades to `conservative` with a one-time warning. Speculative batch jobs persist their request payload + litellm kwargs so a later poll run can reconstruct and re-issue them.

**Tech Stack:** Python 3.10+, litellm>=1.83.0 (first release with Vertex `cancel_batch`), pytest. The provider layer (`llm_batch/providers.py`) is the only module importing litellm.

**Spec:** `docs/superpowers/specs/2026-06-08-canceling-batch-jobs-design.md`

---

## File Structure

- **Modify `llm_batch/providers.py`** — add `_CANCELLABLE` + `provider_supports_cancel()`; reclassify poll statuses so `cancelled` is reported distinctly from `failed` and `canceling` is treated as non-terminal; add `cancel(handle)` delegating to `litellm.cancel_batch`.
- **Modify `llm_batch/job.py`** — add `CANCEL_REQUESTED` state, speculative config params, new persisted fields (`payload`, `litellm_kwargs`, `cancel_requested_at`, `fallback_strategy`), provider-aware timing defaults, the speculative submit decision + one-time downgrade warning, the state-A cancel decision, and the state-B resolution.
- **Modify `llm_batch/poller.py`** — advance `CANCEL_REQUESTED` jobs too, but apply the staleness check only to `SENT` jobs (cancel-requested jobs resolve via `cancel_resolution_timeout`, not the staleness sweep).
- **Modify `llm_batch/pipeline.py`** — thread the five speculative config knobs from `Pipeline` into each `Job`.
- **Modify `tests/conftest.py`** — add a `cancel_batch` stub to the fake litellm namespace.
- **Create `tests/test_job_speculative.py`** — the state machine: submit decision, degrade-with-warning, state-A cancel, state-B salvage/re-issue/timeout, payload reconstruction.
- **Modify `tests/test_providers.py`** — `provider_supports_cancel`, the `canceled` poll status, `cancel()`.
- **Modify `tests/test_poller.py`** — poller advances `CANCEL_REQUESTED` jobs.
- **Modify `tests/test_pipeline.py`** — config threading.

A note on units: all new time knobs are **hours** (matching the existing `max_batch_turnaround_hours`).

---

## Task 1: Provider cancellability predicate

**Files:**
- Modify: `llm_batch/providers.py`
- Test: `tests/test_providers.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_providers.py`:

```python
def test_provider_supports_cancel():
    from llm_batch import providers
    assert providers.provider_supports_cancel("openai") is True
    assert providers.provider_supports_cancel("azure") is True
    assert providers.provider_supports_cancel("vertex_ai") is True
    assert providers.provider_supports_cancel("anthropic") is False
    assert providers.provider_supports_cancel("gemini") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_providers.py::test_provider_supports_cancel -v`
Expected: FAIL with `AttributeError: module 'llm_batch.providers' has no attribute 'provider_supports_cancel'`

- [ ] **Step 3: Write minimal implementation**

In `llm_batch/providers.py`, after the `BATCH_RATE` constant (around line 20), add:

```python
# Providers whose batch jobs litellm can cancel (litellm>=1.83.0 adds Vertex).
# Anthropic and gemini are absent: they fall back to the conservative strategy.
_CANCELLABLE = {"openai", "azure", "vertex_ai"}


def provider_supports_cancel(provider: str) -> bool:
    """True iff litellm.cancel_batch can cancel a batch for this provider."""
    return provider in _CANCELLABLE
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_providers.py::test_provider_supports_cancel -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add llm_batch/providers.py tests/test_providers.py
git commit -m "feat(providers): add provider_supports_cancel predicate"
```

---

## Task 2: Report `cancelled` distinctly from `failed` in poll_batch

The current `poll_batch` folds `cancelled`/`canceling` into `_FAILED_STATUSES` and returns `"failed"`. The speculative state machine needs `cancelled` reported as a distinct terminal outcome (it triggers interactive re-issue), and `canceling` treated as still-working (keep waiting).

**Files:**
- Modify: `llm_batch/providers.py:192-227` (the status-set constants and the early `poll_batch` branches)
- Test: `tests/test_providers.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_providers.py`:

```python
def test_poll_batch_cancelled_is_distinct_status(litellm_mock):
    from llm_batch import providers
    from llm_batch.providers import BatchHandle
    litellm_mock.retrieve_batch = lambda **k: make_obj(status="cancelled")
    handle = BatchHandle(provider="openai", model="gpt-4o", batch_job_id="b1")
    result = providers.poll_batch(handle)
    assert result.status == "canceled"


def test_poll_batch_canceling_is_pending(litellm_mock):
    from llm_batch import providers
    from llm_batch.providers import BatchHandle
    litellm_mock.retrieve_batch = lambda **k: make_obj(status="cancelling")
    handle = BatchHandle(provider="openai", model="gpt-4o", batch_job_id="b1")
    result = providers.poll_batch(handle)
    assert result.status == "pending"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_providers.py::test_poll_batch_cancelled_is_distinct_status tests/test_providers.py::test_poll_batch_canceling_is_pending -v`
Expected: FAIL — the first asserts `"canceled"` but gets `"failed"`; the second asserts `"pending"` but gets `"failed"`.

- [ ] **Step 3: Implement the reclassification**

In `llm_batch/providers.py`, replace the two status-set constants (currently around lines 192-195):

```python
# Provider batch statuses that mean "still working".
_PENDING_STATUSES = {"validating", "in_progress", "finalizing"}
# Provider batch statuses that mean "terminally dead".
_FAILED_STATUSES = {"failed", "expired", "cancelled", "canceling", "cancelling"}
```

with:

```python
# Provider batch statuses that mean "still working" — including an in-progress
# cancellation, which is non-terminal (we keep polling until it settles).
_PENDING_STATUSES = {"validating", "in_progress", "finalizing", "canceling", "cancelling"}
# Provider batch statuses that mean "the job was cancelled" (terminal). Reported
# distinctly from failure so the speculative state machine can re-issue interactively.
_CANCELED_STATUSES = {"cancelled", "canceled"}
# Provider batch statuses that mean "terminally dead".
_FAILED_STATUSES = {"failed", "expired"}
```

Then in `poll_batch`, immediately after the `if status in _PENDING_STATUSES:` block (currently around lines 215-216), add a new branch before the `_FAILED_STATUSES` check:

```python
    if status in _PENDING_STATUSES:
        return PollResult(status="pending")
    if status in _CANCELED_STATUSES:
        return PollResult(
            status="canceled",
            error_message=f"batch {handle.batch_job_id} reported status={status}",
        )
    if status in _FAILED_STATUSES:
```

(Leave the rest of `poll_batch` unchanged.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_providers.py -v -k "cancelled or canceling or poll_batch"`
Expected: PASS, including the existing `test_poll_batch_provider_failed_is_terminal` and `test_poll_batch_expired_is_terminal` (still `"failed"`).

- [ ] **Step 5: Commit**

```bash
git add llm_batch/providers.py tests/test_providers.py
git commit -m "feat(providers): poll_batch reports cancelled distinctly; canceling is pending"
```

---

## Task 3: `providers.cancel()` delegating to litellm

**Files:**
- Modify: `llm_batch/providers.py` (add `cancel`), `tests/conftest.py` (add `cancel_batch` stub)
- Test: `tests/test_providers.py`

- [ ] **Step 1: Add the `cancel_batch` stub to the fake litellm**

In `tests/conftest.py`, in the `litellm_mock` fixture, add one line next to the other stubs (after `fake.file_content = _not_configured`):

```python
    fake.cancel_batch = _not_configured
```

- [ ] **Step 2: Write the failing tests**

Add to `tests/test_providers.py`:

```python
def test_cancel_delegates_to_litellm_with_provider(litellm_mock):
    from llm_batch import providers
    from llm_batch.providers import BatchHandle
    seen = {}

    def cancel_batch(*, batch_id, custom_llm_provider, **k):
        seen["batch_id"] = batch_id
        seen["provider"] = custom_llm_provider
        return make_obj(id=batch_id, status="cancelling")

    litellm_mock.cancel_batch = cancel_batch
    handle = BatchHandle(provider="vertex_ai", model="vertex_ai/gemini-1.5-pro", batch_job_id="b9")
    assert providers.cancel(handle) is None
    assert seen == {"batch_id": "b9", "provider": "vertex_ai"}


def test_cancel_wraps_transient_errors(litellm_mock):
    from llm_batch import providers

    class Boom(Exception):
        status_code = 503

    def cancel_batch(**k):
        raise Boom("service unavailable")

    litellm_mock.cancel_batch = cancel_batch
    handle = providers.BatchHandle(provider="openai", model="gpt-4o", batch_job_id="b1")
    with pytest.raises(providers.TransientProviderError):
        providers.cancel(handle)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_providers.py::test_cancel_delegates_to_litellm_with_provider tests/test_providers.py::test_cancel_wraps_transient_errors -v`
Expected: FAIL with `AttributeError: module 'llm_batch.providers' has no attribute 'cancel'`

- [ ] **Step 4: Implement `cancel`**

In `llm_batch/providers.py`, add after `submit_batch` (around line 157):

```python
def cancel(handle: BatchHandle) -> None:
    """Initiate cancellation of a batch job and return immediately.

    Delegates to litellm for all providers (OpenAI/Azure/Vertex). Does NOT wait
    for the cancel to take effect — the outcome is observed by a later poll_batch
    (the job transitions through CANCEL_REQUESTED). Transient errors are wrapped so
    the caller can leave the job in place and retry on the next scheduled poll.
    """
    try:
        litellm.cancel_batch(
            batch_id=handle.batch_job_id,
            custom_llm_provider=handle.provider,
        )
    except Exception as exc:
        if _is_transient(exc):
            raise TransientProviderError(str(exc)) from exc
        raise
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_providers.py -v -k cancel`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add llm_batch/providers.py tests/conftest.py tests/test_providers.py
git commit -m "feat(providers): add cancel() delegating to litellm.cancel_batch"
```

---

## Task 4: Job — new state, config params, and persisted-field reconstruction

This task adds the `CANCEL_REQUESTED` state constant, the five speculative config params, and persistence/reconstruction of the request payload + litellm kwargs + cancel timestamp + effective strategy. No behavior change yet (submit/poll still take the conservative path) — this task only wires storage so later tasks can rely on it.

**Files:**
- Modify: `llm_batch/job.py`
- Test: `tests/test_job_speculative.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_job_speculative.py`:

```python
import logging

import pytest

from llm_batch import job as job_module
from llm_batch.job import Job, _now
from llm_batch.providers import BatchHandle, CompletionResult, PollResult
from llm_batch.storage.file import FileStorage
from datetime import timedelta


@pytest.fixture(autouse=True)
def _reset_warn_state():
    job_module._SPECULATIVE_DOWNGRADE_WARNED.clear()
    yield
    job_module._SPECULATIVE_DOWNGRADE_WARNED.clear()


def _spec_job(tmp_path, key="x", *, deadline=None, model="gpt-4o", mode="batch",
              requests=None, messages=None, **over):
    if requests is None and messages is None:
        messages = [{"role": "user", "content": "hi"}]
    kwargs = dict(
        fallback_strategy="speculative", deadline=deadline, model=model, mode=mode,
        poll_interval_hours=1.0, interactive_runtime_estimate_hours=0.25,
    )
    kwargs.update(over)
    return Job(FileStorage(str(tmp_path)).for_job(key), key,
               messages=messages, requests=requests, **kwargs)


def test_speculative_batch_persists_payload_and_strategy(tmp_path, monkeypatch):
    monkeypatch.setattr(job_module.providers, "provider_of", lambda m: "openai")
    monkeypatch.setattr(job_module.providers, "provider_supports_cancel", lambda p: True)
    monkeypatch.setattr(
        job_module.providers, "submit_batch",
        lambda model, requests, **kw: BatchHandle(
            provider="openai", model=model, batch_job_id="b1", input_file_id="f1"))

    job = _spec_job(tmp_path, deadline=(_now() + timedelta(hours=72)).isoformat(),
                    temperature=0.2)
    assert job.execute() is None
    snap = job.snapshot()
    assert snap["state"] == Job.SENT
    assert snap["fallback_strategy"] == "speculative"
    assert snap["payload"] == [[{"role": "user", "content": "hi"}]]
    assert snap["litellm_kwargs"] == {"temperature": 0.2}

    # Reconstruct from storage (as the poller does: no messages=) and confirm
    # the payload + kwargs come back.
    reloaded = Job(FileStorage(str(tmp_path)).for_job("x"))
    assert reloaded._payload == [[{"role": "user", "content": "hi"}]]
    assert reloaded._kw == {"temperature": 0.2}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_job_speculative.py::test_speculative_batch_persists_payload_and_strategy -v`
Expected: FAIL — `Job.__init__` rejects unknown kwarg `fallback_strategy` (TypeError), or `_SPECULATIVE_DOWNGRADE_WARNED` doesn't exist (AttributeError).

- [ ] **Step 3: Add the state constant, config params, persistence, and reconstruction**

In `llm_batch/job.py`:

(a) Add `import logging` to the top imports.

(b) Add a module-level warn-tracking set and logger after the imports (around line 9):

```python
_LOG = logging.getLogger("llm_batch")
# Providers already warned about a speculative→conservative downgrade (process-wide,
# so the warning is emitted once, not once per poll).
_SPECULATIVE_DOWNGRADE_WARNED: set[str] = set()
```

(c) Add `CANCEL_REQUESTED` next to the other state constants (after `SENT = "sent"`):

```python
    CANCEL_REQUESTED = "cancel_requested"
```

(d) Extend `_PERSISTED_FIELDS` (around line 35) with the new fields:

```python
_PERSISTED_FIELDS = (
    "key", "state", "mode", "model", "n_requests", "bundle", "batch_job_id",
    "input_file_id", "output_file_id", "submitted_at", "completed_at", "deadline",
    "actual_mode", "response", "usage", "cost", "error_message", "poll_attempts",
    "rerun_cmd", "cleanup_provider_files",
    "fallback_strategy", "payload", "litellm_kwargs", "cancel_requested_at",
)
```

(e) Add the five new params to `__init__` (extend the signature; keep existing params):

```python
    def __init__(self, storage, key=None, *, messages=None, requests=None,
                 model=None, mode="batch", deadline=None,
                 max_batch_turnaround_hours=24, rerun_cmd=None,
                 cleanup_provider_files=True,
                 fallback_strategy="conservative", poll_interval_hours=1.0,
                 interactive_runtime_estimate_hours=0.25,
                 cancel_margin_hours=None, cancel_resolution_timeout_hours=None,
                 **litellm_kwargs):
```

(f) Store them just after `self._kw = litellm_kwargs` (around line 84):

```python
        self._fallback_strategy = fallback_strategy
        self._poll_interval_h = poll_interval_hours
        self._interactive_estimate_h = interactive_runtime_estimate_hours
        self._cancel_margin_h = cancel_margin_hours
        self._cancel_resolution_timeout_h = cancel_resolution_timeout_hours
```

(g) In the resume block (inside `if self._storage.load():`, after the existing reconstruction lines, around line 99), reconstruct payload + kwargs when not passed in-memory:

```python
            if self._payload is None and self._state.get("payload") is not None:
                self._payload = self._state.get("payload")
                self._is_bundle = bool(self._state.get("bundle"))
            if not self._kw and self._state.get("litellm_kwargs"):
                self._kw = dict(self._state.get("litellm_kwargs"))
```

(h) In `_base_state` (around line 169) add the new fields to the returned dict (set defaults; `_submit_batch` fills payload/kwargs/strategy for speculative):

```python
            "poll_attempts": 0,
            "fallback_strategy": "conservative",
            "payload": None,
            "litellm_kwargs": None,
            "cancel_requested_at": None,
        }
```

- [ ] **Step 4: Run test to verify it fails differently**

Run: `pytest tests/test_job_speculative.py::test_speculative_batch_persists_payload_and_strategy -v`
Expected: still FAIL — `Job.__init__` now accepts the kwargs, but `snapshot()["fallback_strategy"]` is `"conservative"` and `payload` is `None` because `_submit_batch` doesn't yet store the speculative payload. This is fixed in Task 6. Leave this test failing for now and rerun the rest of the suite to confirm no regressions:

Run: `pytest tests/test_job.py tests/test_job_fallback.py tests/test_poller.py -v`
Expected: PASS (existing behavior unchanged — new fields default to conservative/None).

- [ ] **Step 5: Commit**

```bash
git add llm_batch/job.py tests/test_job_speculative.py
git commit -m "feat(job): add CANCEL_REQUESTED state, speculative config params, payload persistence scaffolding"
```

---

## Task 5: Job — provider-aware timing defaults and the "can afford to wait" decision

**Files:**
- Modify: `llm_batch/job.py`
- Test: `tests/test_job_speculative.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_job_speculative.py`:

```python
def test_can_afford_to_wait_decision(tmp_path, monkeypatch):
    monkeypatch.setattr(job_module.providers, "provider_of", lambda m: "openai")
    # window = poll_interval(1.0) + cancel_margin(openai default 0.25)
    #          + interactive_estimate(0.25) + safety(0.1) = 1.6h
    job = _spec_job(tmp_path, deadline=(_now() + timedelta(hours=3)).isoformat())
    assert job._can_afford_to_wait() is True

    tight = _spec_job(tmp_path, key="y",
                      deadline=(_now() + timedelta(hours=1)).isoformat())
    assert tight._can_afford_to_wait() is False


def test_can_afford_to_wait_no_deadline_is_true(tmp_path, monkeypatch):
    monkeypatch.setattr(job_module.providers, "provider_of", lambda m: "openai")
    job = _spec_job(tmp_path, deadline=None)
    assert job._can_afford_to_wait() is True


def test_cancel_margin_default_is_provider_aware(tmp_path, monkeypatch):
    monkeypatch.setattr(job_module.providers, "provider_of", lambda m: "vertex_ai")
    job = _spec_job(tmp_path, model="vertex_ai/gemini-1.5-pro")
    # vertex default margin 0.5 → window = 1.0 + 0.5 + 0.25 + 0.1 = 1.85h
    assert job._speculative_window_hours() == pytest.approx(1.85)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_job_speculative.py -v -k "afford or margin_default"`
Expected: FAIL with `AttributeError: 'Job' object has no attribute '_can_afford_to_wait'`

- [ ] **Step 3: Implement the timing helpers**

In `llm_batch/job.py`, add module-level defaults and helpers after `_ZERO_COST` (around line 32):

```python
# Provider-aware timing defaults (hours), used when the caller leaves the knob None.
_SAFETY_HOURS = 0.1
_DEFAULT_CANCEL_MARGIN_HOURS = {"openai": 0.25, "azure": 0.25, "vertex_ai": 0.5}
_FALLBACK_CANCEL_MARGIN_HOURS = 0.5
_DEFAULT_CANCEL_RESOLUTION_HOURS = {"openai": 0.5, "azure": 0.5, "vertex_ai": 2.0}
_FALLBACK_CANCEL_RESOLUTION_HOURS = 2.0


def _default_cancel_margin(provider: str) -> float:
    return _DEFAULT_CANCEL_MARGIN_HOURS.get(provider, _FALLBACK_CANCEL_MARGIN_HOURS)


def _default_cancel_resolution_timeout(provider: str) -> float:
    return _DEFAULT_CANCEL_RESOLUTION_HOURS.get(provider, _FALLBACK_CANCEL_RESOLUTION_HOURS)
```

Then add these methods to `Job` (place them just before `_should_use_interactive`, around line 157):

```python
    def _provider(self) -> str:
        return providers.provider_of(self._model)

    def _speculative_window_hours(self) -> float:
        """Minimum slack the deadline must have to risk waiting one more poll."""
        margin = self._cancel_margin_h
        if margin is None:
            margin = _default_cancel_margin(self._provider())
        return (self._poll_interval_h + margin
                + self._interactive_estimate_h + _SAFETY_HOURS)

    def _hours_remaining(self) -> float | None:
        if self._deadline is None:
            return None
        deadline_dt = _parse_iso(self._deadline)
        if deadline_dt is None:
            return None
        return (deadline_dt - _now()).total_seconds() / 3600.0

    def _can_afford_to_wait(self) -> bool:
        """True if the deadline leaves room to wait for the next scheduled poll."""
        remaining = self._hours_remaining()
        if remaining is None:
            return True
        return remaining >= self._speculative_window_hours()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_job_speculative.py -v -k "afford or margin_default"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add llm_batch/job.py tests/test_job_speculative.py
git commit -m "feat(job): provider-aware cancel timing defaults and can-afford-to-wait decision"
```

---

## Task 6: Job — speculative submit decision + one-time downgrade warning

This makes `execute()` on a `NOT_SENT` speculative job: submit optimistically when there's slack; go interactive immediately when the deadline is already too tight; and degrade to conservative (with a one-time warning) on a non-cancellable provider. It also stores payload/kwargs/strategy for speculative batch submissions.

**Files:**
- Modify: `llm_batch/job.py` (`_submit`, `_submit_batch`, add `_effective_strategy` + `_warn_*`)
- Test: `tests/test_job_speculative.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_job_speculative.py`:

```python
def _patch_providers(monkeypatch, *, provider="openai", supports=True):
    monkeypatch.setattr(job_module.providers, "provider_of", lambda m: provider)
    monkeypatch.setattr(job_module.providers, "provider_supports_cancel", lambda p: supports)
    calls = {"submit": 0, "complete": 0}

    def submit_batch(model, requests, **kw):
        calls["submit"] += 1
        return BatchHandle(provider=provider, model=model, batch_job_id="b1", input_file_id="f1")

    def complete(model, messages, **kw):
        calls["complete"] += 1
        return CompletionResult(text="now", usage={}, cost={"charged_usd": 0.0, "interactive_usd": 0.0, "saved_usd": 0.0})

    monkeypatch.setattr(job_module.providers, "submit_batch", submit_batch)
    monkeypatch.setattr(job_module.providers, "complete", complete)
    return calls


def test_speculative_ample_time_submits_batch(tmp_path, monkeypatch):
    calls = _patch_providers(monkeypatch)
    job = _spec_job(tmp_path, deadline=(_now() + timedelta(hours=72)).isoformat())
    assert job.execute() is None
    assert calls["submit"] == 1 and calls["complete"] == 0
    snap = job.snapshot()
    assert snap["state"] == Job.SENT
    assert snap["fallback_strategy"] == "speculative"
    assert snap["payload"] == [[{"role": "user", "content": "hi"}]]


def test_speculative_too_tight_goes_interactive_immediately(tmp_path, monkeypatch):
    calls = _patch_providers(monkeypatch)
    # 1h remaining < ~1.6h speculative window → no point submitting a batch.
    job = _spec_job(tmp_path, deadline=(_now() + timedelta(hours=1)).isoformat())
    assert job.execute() == "now"
    assert calls["submit"] == 0 and calls["complete"] == 1
    assert job.snapshot()["actual_mode"] == "interactive"


def test_speculative_anthropic_degrades_to_conservative_with_one_warning(tmp_path, monkeypatch, caplog):
    calls = _patch_providers(monkeypatch, provider="anthropic", supports=False)
    deadline = (_now() + timedelta(hours=72)).isoformat()  # ample for conservative too
    with caplog.at_level(logging.WARNING, logger="llm_batch"):
        j1 = _spec_job(tmp_path, key="a", model="claude-3-5-sonnet", deadline=deadline)
        assert j1.execute() is None
        j2 = _spec_job(tmp_path, key="b", model="claude-3-5-sonnet", deadline=deadline)
        assert j2.execute() is None
    # Conservative path under ample time → batch submitted, stored as conservative.
    assert calls["submit"] == 2
    assert j1.snapshot()["fallback_strategy"] == "conservative"
    warnings = [r for r in caplog.records if "speculative" in r.getMessage().lower()]
    assert len(warnings) == 1  # one-time, not per job/poll
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_job_speculative.py -v -k "ample_time_submits or too_tight or degrades"`
Expected: FAIL — currently `_submit` ignores the strategy, so `fallback_strategy` stays `"conservative"`/`payload` stays `None`, the tight-deadline job still submits a batch, and no warning is emitted.

- [ ] **Step 3: Implement the submit decision**

In `llm_batch/job.py`, add the warning helper and strategy resolver as methods on `Job` (near `_can_afford_to_wait`):

```python
    def _warn_speculative_downgrade(self, provider: str) -> None:
        if provider in _SPECULATIVE_DOWNGRADE_WARNED:
            return
        _SPECULATIVE_DOWNGRADE_WARNED.add(provider)
        _LOG.warning(
            "speculative fallback_strategy is unavailable for provider %r "
            "(litellm cannot cancel its batches); falling back to conservative.",
            provider)

    def _effective_strategy(self) -> str:
        """Resolve the strategy actually used, degrading speculative when needed."""
        if self._fallback_strategy != "speculative":
            return "conservative"
        provider = self._provider()
        if not providers.provider_supports_cancel(provider):
            self._warn_speculative_downgrade(provider)
            return "conservative"
        return "speculative"
```

Replace `_submit` (currently around lines 194-200):

```python
    def _submit(self):
        if self._payload is None:
            raise RuntimeError(f"job {self._key!r} has no request payload to submit")

        if self._mode == "interactive":
            return self._submit_interactive()

        if self._effective_strategy() == "speculative":
            # Submit optimistically unless the deadline is already inside the
            # cancel+re-issue window, in which case batch is pointless.
            if not self._can_afford_to_wait():
                return self._submit_interactive()
            return self._submit_batch(strategy="speculative")

        # conservative: decide once, here, using the worst-case turnaround margin.
        if self._should_use_interactive():
            return self._submit_interactive()
        return self._submit_batch(strategy="conservative")
```

Replace `_submit_batch` (currently around lines 224-236) to accept and persist the strategy, payload, and kwargs:

```python
    def _submit_batch(self, *, strategy="conservative"):
        try:
            handle = providers.submit_batch(self._model, self._payload, **self._kw)
        except providers.TransientProviderError:
            return None  # leave state NOT_SENT; retried next run

        state = self._base_state()
        state.update(state=self.SENT, actual_mode="batch",
                     batch_job_id=handle.batch_job_id,
                     input_file_id=handle.input_file_id,
                     fallback_strategy=strategy)
        if strategy == "speculative":
            # Persist enough to reconstruct and re-issue interactively from a
            # later scheduled poll (the poller rebuilds Job with no payload/kwargs).
            state["payload"] = self._payload
            state["litellm_kwargs"] = dict(self._kw)
        self._state = state
        self._save()
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_job_speculative.py -v -k "ample_time_submits or too_tight or degrades or persists_payload"`
Expected: PASS (including `test_speculative_batch_persists_payload_and_strategy` from Task 4, now satisfied).

- [ ] **Step 5: Run the full suite to confirm no regressions**

Run: `pytest tests/test_job.py tests/test_job_fallback.py -v`
Expected: PASS — conservative submissions still store `fallback_strategy="conservative"` and no payload.

- [ ] **Step 6: Commit**

```bash
git add llm_batch/job.py tests/test_job_speculative.py
git commit -m "feat(job): speculative submit decision with one-time anthropic downgrade warning"
```

---

## Task 7: Job — state-A poll (decide to wait or cancel)

A `SENT` speculative job, polled, either completes, keeps waiting, or — when it can no longer afford to wait — initiates the cancel and transitions to `CANCEL_REQUESTED`. This task splits `_poll` by strategy and extracts a shared completed-handler.

**Files:**
- Modify: `llm_batch/job.py` (`execute`, `_poll` → split, add `_poll_speculative` + `_finish_completed`)
- Test: `tests/test_job_speculative.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_job_speculative.py`:

```python
def _seed_speculative_sent(tmp_path, key="x", *, deadline, provider="openai"):
    """Write a SENT speculative job directly (as it would be after submit)."""
    handle = FileStorage(str(tmp_path)).for_job(key)
    handle.save({
        "key": key, "state": "sent", "mode": "batch", "model": "gpt-4o",
        "n_requests": 1, "bundle": False, "batch_job_id": "b1",
        "input_file_id": "f-in", "output_file_id": None,
        "submitted_at": _now().isoformat(), "completed_at": None,
        "deadline": deadline, "actual_mode": "batch", "response": None,
        "usage": {}, "cost": {}, "error_message": None, "poll_attempts": 0,
        "rerun_cmd": None, "cleanup_provider_files": False,
        "fallback_strategy": "speculative",
        "payload": [[{"role": "user", "content": "hi"}]],
        "litellm_kwargs": {}, "cancel_requested_at": None,
    })
    return handle


def test_state_a_keeps_waiting_when_time_remains(tmp_path, monkeypatch):
    monkeypatch.setattr(job_module.providers, "provider_of", lambda m: "openai")
    monkeypatch.setattr(job_module.providers, "poll_batch",
                        lambda h: PollResult(status="pending"))
    cancel_called = {"n": 0}
    monkeypatch.setattr(job_module.providers, "cancel",
                        lambda h: cancel_called.__setitem__("n", cancel_called["n"] + 1))
    handle = _seed_speculative_sent(tmp_path, deadline=(_now() + timedelta(hours=72)).isoformat())
    job = Job(handle)
    assert job.execute() is None
    assert job.state == Job.SENT
    assert cancel_called["n"] == 0


def test_state_a_initiates_cancel_when_out_of_time(tmp_path, monkeypatch):
    monkeypatch.setattr(job_module.providers, "provider_of", lambda m: "openai")
    monkeypatch.setattr(job_module.providers, "poll_batch",
                        lambda h: PollResult(status="pending"))
    cancel_called = {"n": 0}
    monkeypatch.setattr(job_module.providers, "cancel",
                        lambda h: cancel_called.__setitem__("n", cancel_called["n"] + 1))
    handle = _seed_speculative_sent(tmp_path, deadline=(_now() + timedelta(hours=1)).isoformat())
    job = Job(handle)
    assert job.execute() is None
    assert job.state == Job.CANCEL_REQUESTED
    assert cancel_called["n"] == 1
    assert job.snapshot()["cancel_requested_at"] is not None


def test_state_a_completed_returns_result(tmp_path, monkeypatch):
    monkeypatch.setattr(job_module.providers, "provider_of", lambda m: "openai")
    monkeypatch.setattr(job_module.providers, "cleanup", lambda h: None)
    monkeypatch.setattr(
        job_module.providers, "poll_batch",
        lambda h: PollResult(status="completed", texts=["done"],
                             usage={}, cost={}, output_file_id="of1"))
    handle = _seed_speculative_sent(tmp_path, deadline=(_now() + timedelta(hours=1)).isoformat())
    job = Job(handle)
    assert job.execute() == "done"
    assert job.state == Job.COMPLETED


def test_state_a_cancel_transient_stays_sent(tmp_path, monkeypatch):
    monkeypatch.setattr(job_module.providers, "provider_of", lambda m: "openai")
    monkeypatch.setattr(job_module.providers, "poll_batch",
                        lambda h: PollResult(status="pending"))

    def boom(h):
        raise job_module.providers.TransientProviderError("rate limited")

    monkeypatch.setattr(job_module.providers, "cancel", boom)
    handle = _seed_speculative_sent(tmp_path, deadline=(_now() + timedelta(hours=1)).isoformat())
    job = Job(handle)
    assert job.execute() is None
    assert job.state == Job.SENT  # cancel failed transiently; retry next poll
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_job_speculative.py -v -k state_a`
Expected: FAIL — `_poll` ignores strategy, so the out-of-time job stays `SENT` and `cancel` is never called; `CANCEL_REQUESTED` transitions don't happen.

- [ ] **Step 3: Implement the state-A poll**

In `llm_batch/job.py`:

(a) Extend `execute` to route the new state (add a branch before `return self._submit()`):

```python
        if state == self.SENT:
            return self._poll()
        if state == self.CANCEL_REQUESTED:
            return self._poll_cancel()
        return self._submit()  # NOT_SENT
```

(b) Replace the current `_poll` (lines ~248-277) with a strategy dispatcher plus the conservative body kept intact, and add `_finish_completed` + `_poll_speculative`:

```python
    def _poll(self):
        handle = self._batch_handle()
        if self._state.get("fallback_strategy") == "speculative":
            return self._poll_speculative(handle)
        return self._poll_conservative(handle)

    def _poll_conservative(self, handle):
        result = providers.poll_batch(handle)

        if result.status in ("pending", "transient_error"):
            self._state["poll_attempts"] = self._state.get("poll_attempts", 0) + 1
            if result.status == "transient_error":
                self._state["error_message"] = result.error_message
            self._save()
            return None

        if result.status in ("failed", "canceled"):
            self._mark_failed(result.error_message, handle, result.output_file_id)
            raise BatchJobFailed(result.error_message or "batch failed",
                                 handle.batch_job_id)

        return self._finish_completed(handle, result)

    def _finish_completed(self, handle, result):
        response = result.texts if self._is_bundle else result.texts[0]
        self._state.update(
            state=self.COMPLETED, response=response, completed_at=_now_iso(),
            usage=result.usage, cost=result.cost,
            output_file_id=result.output_file_id, error_message=None)
        self._save()  # persist BEFORE provider-side cleanup (spec §7.1)
        if self._cleanup:
            handle.output_file_id = result.output_file_id
            try:
                providers.cleanup(handle)
            except Exception:
                pass  # best-effort; result already durably saved
        return response

    def _poll_speculative(self, handle):
        result = providers.poll_batch(handle)

        if result.status == "completed":
            return self._finish_completed(handle, result)

        if result.status in ("failed", "canceled"):
            # We have not requested a cancel yet, so 'canceled' means it was
            # cancelled out of band → treat as a failure.
            self._mark_failed(result.error_message, handle, result.output_file_id)
            raise BatchJobFailed(result.error_message or "batch failed",
                                 handle.batch_job_id)

        # pending or transient_error → still working
        self._state["poll_attempts"] = self._state.get("poll_attempts", 0) + 1
        if result.status == "transient_error":
            self._state["error_message"] = result.error_message
            self._save()
            return None

        if self._can_afford_to_wait():
            self._save()
            return None

        # Out of time: initiate cancel and hand off to CANCEL_REQUESTED. Do NOT
        # wait for it to take effect — the next scheduled poll resolves it.
        try:
            providers.cancel(handle)
        except providers.TransientProviderError:
            self._save()  # cancel failed transiently; retry next poll, stay SENT
            return None
        self._state["state"] = self.CANCEL_REQUESTED
        self._state["cancel_requested_at"] = _now_iso()
        self._save()
        return None
```

Note: the existing `_poll` body becomes `_poll_conservative` verbatim except it now also treats `"canceled"` as a failure (an out-of-band cancel on a conservative job) and delegates the completed branch to `_finish_completed`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_job_speculative.py -v -k state_a`
Expected: PASS

- [ ] **Step 5: Run the full suite to confirm the conservative path is unchanged**

Run: `pytest tests/test_job.py tests/test_poller.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add llm_batch/job.py tests/test_job_speculative.py
git commit -m "feat(job): state-A speculative poll initiates cancel when out of time"
```

---

## Task 8: Job — state-B resolution (salvage / re-issue / timeout)

A `CANCEL_REQUESTED` job, polled on a later run, resolves: `completed` → salvage; `canceled` → re-issue interactively; still-pending → keep waiting unless `cancel_resolution_timeout` has elapsed, in which case re-issue interactively.

**Files:**
- Modify: `llm_batch/job.py` (add `_poll_cancel`, `_reissue_interactive`, `_cancel_resolution_expired`)
- Test: `tests/test_job_speculative.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_job_speculative.py`:

```python
def _seed_cancel_requested(tmp_path, key="x", *, cancel_requested_at, provider="openai",
                           model="gpt-4o"):
    handle = FileStorage(str(tmp_path)).for_job(key)
    handle.save({
        "key": key, "state": "cancel_requested", "mode": "batch", "model": model,
        "n_requests": 1, "bundle": False, "batch_job_id": "b1",
        "input_file_id": "f-in", "output_file_id": None,
        "submitted_at": _now().isoformat(), "completed_at": None, "deadline": None,
        "actual_mode": "batch", "response": None, "usage": {}, "cost": {},
        "error_message": None, "poll_attempts": 1, "rerun_cmd": None,
        "cleanup_provider_files": False, "fallback_strategy": "speculative",
        "payload": [[{"role": "user", "content": "hi"}]],
        "litellm_kwargs": {}, "cancel_requested_at": cancel_requested_at,
    })
    return handle


def test_state_b_completed_salvages_batch_result(tmp_path, monkeypatch):
    monkeypatch.setattr(job_module.providers, "provider_of", lambda m: "openai")
    monkeypatch.setattr(job_module.providers, "cleanup", lambda h: None)
    monkeypatch.setattr(
        job_module.providers, "poll_batch",
        lambda h: PollResult(status="completed", texts=["batched"],
                             usage={}, cost={}, output_file_id="of1"))
    handle = _seed_cancel_requested(tmp_path, cancel_requested_at=_now().isoformat())
    job = Job(handle)
    assert job.execute() == "batched"
    assert job.state == Job.COMPLETED
    assert job.snapshot()["actual_mode"] == "batch"


def test_state_b_cancelled_reissues_interactively(tmp_path, monkeypatch):
    monkeypatch.setattr(job_module.providers, "provider_of", lambda m: "openai")
    monkeypatch.setattr(job_module.providers, "cleanup", lambda h: None)
    monkeypatch.setattr(job_module.providers, "poll_batch",
                        lambda h: PollResult(status="canceled"))
    monkeypatch.setattr(
        job_module.providers, "complete",
        lambda model, messages, **kw: CompletionResult(
            text="interactive-now", usage={}, cost={"charged_usd": 0.02, "interactive_usd": 0.02, "saved_usd": 0.0}))
    handle = _seed_cancel_requested(tmp_path, cancel_requested_at=_now().isoformat())
    job = Job(handle)
    assert job.execute() == "interactive-now"
    assert job.state == Job.COMPLETED
    assert job.snapshot()["actual_mode"] == "interactive"


def test_state_b_still_queued_keeps_waiting(tmp_path, monkeypatch):
    monkeypatch.setattr(job_module.providers, "provider_of", lambda m: "openai")
    monkeypatch.setattr(job_module.providers, "poll_batch",
                        lambda h: PollResult(status="pending"))
    handle = _seed_cancel_requested(tmp_path, cancel_requested_at=_now().isoformat())
    job = Job(handle)
    assert job.execute() is None
    assert job.state == Job.CANCEL_REQUESTED


def test_state_b_timeout_forces_interactive(tmp_path, monkeypatch):
    monkeypatch.setattr(job_module.providers, "provider_of", lambda m: "openai")
    monkeypatch.setattr(job_module.providers, "cleanup", lambda h: None)
    monkeypatch.setattr(job_module.providers, "poll_batch",
                        lambda h: PollResult(status="pending"))
    monkeypatch.setattr(
        job_module.providers, "complete",
        lambda model, messages, **kw: CompletionResult(
            text="forced", usage={}, cost={"charged_usd": 0.0, "interactive_usd": 0.0, "saved_usd": 0.0}))
    # openai resolution timeout default 0.5h; stamp the request 2h ago → expired.
    stamped = (_now() - timedelta(hours=2)).isoformat()
    handle = _seed_cancel_requested(tmp_path, cancel_requested_at=stamped)
    job = Job(handle)
    assert job.execute() == "forced"
    assert job.state == Job.COMPLETED
    assert job.snapshot()["actual_mode"] == "interactive"


def test_state_b_reissue_transient_stays_cancel_requested(tmp_path, monkeypatch):
    monkeypatch.setattr(job_module.providers, "provider_of", lambda m: "openai")
    monkeypatch.setattr(job_module.providers, "poll_batch",
                        lambda h: PollResult(status="canceled"))

    def boom(model, messages, **kw):
        raise job_module.providers.TransientProviderError("rate limited")

    monkeypatch.setattr(job_module.providers, "complete", boom)
    handle = _seed_cancel_requested(tmp_path, cancel_requested_at=_now().isoformat())
    job = Job(handle)
    assert job.execute() is None
    assert job.state == Job.CANCEL_REQUESTED  # retry interactive next poll
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_job_speculative.py -v -k state_b`
Expected: FAIL with `AttributeError: 'Job' object has no attribute '_poll_cancel'`

- [ ] **Step 3: Implement state-B resolution**

In `llm_batch/job.py`, add these methods to `Job` (after `_poll_speculative`):

```python
    def _cancel_resolution_expired(self) -> bool:
        stamped = _parse_iso(self._state.get("cancel_requested_at"))
        if stamped is None:
            return False
        timeout_h = self._cancel_resolution_timeout_h
        if timeout_h is None:
            timeout_h = _default_cancel_resolution_timeout(self._provider())
        elapsed_h = (_now() - stamped).total_seconds() / 3600.0
        return elapsed_h >= timeout_h

    def _poll_cancel(self):
        handle = self._batch_handle()
        result = providers.poll_batch(handle)

        if result.status == "completed":
            # Finished before the cancel took effect → salvage the batch result.
            return self._finish_completed(handle, result)

        if result.status == "canceled":
            return self._reissue_interactive(handle)

        if result.status == "failed":
            self._mark_failed(result.error_message, handle, result.output_file_id)
            raise BatchJobFailed(result.error_message or "batch failed",
                                 handle.batch_job_id)

        # pending / transient_error: cancel not yet effective.
        self._state["poll_attempts"] = self._state.get("poll_attempts", 0) + 1
        if result.status == "transient_error":
            self._state["error_message"] = result.error_message
        if self._cancel_resolution_expired():
            # Stop waiting on a cancel that won't settle (e.g. Vertex best-effort).
            return self._reissue_interactive(handle)
        self._save()
        return None

    def _reissue_interactive(self, handle):
        """Re-run the persisted request interactively and mark COMPLETED."""
        texts = []
        usage = dict(_ZERO_USAGE)
        cost = dict(_ZERO_COST)
        try:
            for messages in self._payload:
                cr = providers.complete(self._model, messages, **self._kw)
                texts.append(cr.text)
                _accumulate(usage, cr.usage)
                _accumulate_cost(cost, cr.cost)
        except providers.TransientProviderError:
            self._save()  # stay CANCEL_REQUESTED; retry interactive next poll
            return None

        response = texts if self._is_bundle else texts[0]
        self._state.update(
            state=self.COMPLETED, actual_mode="interactive",
            completed_at=_now_iso(), response=response, usage=usage, cost=cost,
            error_message=None)
        self._save()
        if self._cleanup:
            try:
                providers.cleanup(handle)  # remove the (cancelled) batch artifacts
            except Exception:
                pass
        return response
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_job_speculative.py -v -k state_b`
Expected: PASS

- [ ] **Step 5: Run the whole speculative + job suite**

Run: `pytest tests/test_job_speculative.py tests/test_job.py tests/test_job_fallback.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add llm_batch/job.py tests/test_job_speculative.py
git commit -m "feat(job): state-B resolves cancel — salvage, re-issue interactively, or timeout"
```

---

## Task 9: Poller — advance CANCEL_REQUESTED jobs

The poller currently skips any job whose state isn't `SENT`. It must also advance `CANCEL_REQUESTED` jobs, while keeping the staleness sweep limited to `SENT` (a cancel-requested job is bounded by `cancel_resolution_timeout`, not staleness).

**Files:**
- Modify: `llm_batch/poller.py:31-44`
- Test: `tests/test_poller.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_poller.py`:

```python
def test_poller_advances_cancel_requested_jobs(tmp_path, fake_providers, monkeypatch):
    from llm_batch.providers import PollResult, CompletionResult
    # A CANCEL_REQUESTED job whose batch has now settled as cancelled.
    handle = FileStorage(str(tmp_path)).for_job("c")
    handle.save({
        "key": "c", "state": "cancel_requested", "mode": "batch", "model": "gpt-4o",
        "n_requests": 1, "bundle": False, "batch_job_id": "b1",
        "input_file_id": "f-in", "output_file_id": None,
        "submitted_at": _now().isoformat(), "completed_at": None, "deadline": None,
        "actual_mode": "batch", "response": None, "usage": {}, "cost": {},
        "error_message": None, "poll_attempts": 1, "rerun_cmd": None,
        "cleanup_provider_files": False, "fallback_strategy": "speculative",
        "payload": [[{"role": "user", "content": "hi"}]],
        "litellm_kwargs": {}, "cancel_requested_at": _now().isoformat(),
    })
    monkeypatch.setattr(job_module.providers, "poll_batch",
                        lambda h: PollResult(status="canceled"))
    monkeypatch.setattr(
        job_module.providers, "complete",
        lambda model, messages, **kw: CompletionResult(
            text="reissued", usage={}, cost={"charged_usd": 0.0, "interactive_usd": 0.0, "saved_usd": 0.0}))

    Poller(FileStorage(str(tmp_path))).poll_all()

    reloaded = Job(FileStorage(str(tmp_path)).for_job("c"))
    assert reloaded.state == Job.COMPLETED
    assert reloaded.snapshot()["actual_mode"] == "interactive"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_poller.py::test_poller_advances_cancel_requested_jobs -v`
Expected: FAIL — the poller skips the job (state isn't `SENT`), so it stays `cancel_requested`.

- [ ] **Step 3: Update the poller filter and staleness scope**

In `llm_batch/poller.py`, replace the top of the `for handle in self._backend.iter_jobs():` loop (currently lines 31-40):

```python
        for handle in self._backend.iter_jobs():
            if handle.get("state") != Job.SENT:
                continue

            if self._is_stale(handle):
                job = Job(handle)
                job.mark_failed(f"stale: no completion within {self._max_age.days}d")
                continue

            job = Job(handle)
```

with:

```python
        for handle in self._backend.iter_jobs():
            state = handle.get("state")
            if state not in (Job.SENT, Job.CANCEL_REQUESTED):
                continue

            # Staleness only force-fails SENT jobs; a CANCEL_REQUESTED job is
            # bounded instead by its cancel_resolution_timeout (re-issues interactively).
            if state == Job.SENT and self._is_stale(handle):
                job = Job(handle)
                job.mark_failed(f"stale: no completion within {self._max_age.days}d")
                continue

            job = Job(handle)
```

(The rest of the loop body — `job.execute()`, the `BatchJobFailed` handler, and the rerun-cmd collection — is unchanged.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_poller.py::test_poller_advances_cancel_requested_jobs -v`
Expected: PASS

- [ ] **Step 5: Run the full poller suite**

Run: `pytest tests/test_poller.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add llm_batch/poller.py tests/test_poller.py
git commit -m "feat(poller): advance CANCEL_REQUESTED jobs; limit staleness to SENT"
```

---

## Task 10: Pipeline — thread speculative config to jobs

**Files:**
- Modify: `llm_batch/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_pipeline.py`:

```python
def test_pipeline_threads_speculative_config_to_jobs(tmp_path):
    from llm_batch.pipeline import Pipeline
    p = Pipeline(str(tmp_path), fallback_strategy="speculative",
                 poll_interval_hours=2.0, interactive_runtime_estimate_hours=0.5,
                 cancel_margin_hours=0.4, cancel_resolution_timeout_hours=3.0)
    job = p.job("k", messages=[{"role": "user", "content": "hi"}], model="gpt-4o")
    assert job._fallback_strategy == "speculative"
    assert job._poll_interval_h == 2.0
    assert job._interactive_estimate_h == 0.5
    assert job._cancel_margin_h == 0.4
    assert job._cancel_resolution_timeout_h == 3.0


def test_pipeline_defaults_to_conservative(tmp_path):
    from llm_batch.pipeline import Pipeline
    p = Pipeline(str(tmp_path))
    job = p.job("k", messages=[{"role": "user", "content": "hi"}], model="gpt-4o")
    assert job._fallback_strategy == "conservative"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pipeline.py -v -k "speculative_config or defaults_to_conservative"`
Expected: FAIL — `Pipeline.__init__` rejects `fallback_strategy` (TypeError).

- [ ] **Step 3: Add the params to Pipeline and forward them**

In `llm_batch/pipeline.py`, extend `Pipeline.__init__` (the signature around lines 18-20) by adding the five params:

```python
    def __init__(self, state_dir, *, mode="batch", deadline=None,
                 time_budget_hours=None, max_batch_turnaround_hours=24,
                 rerun_cmd=None, storage=None, cleanup_provider_files=True,
                 fallback_strategy="conservative", poll_interval_hours=1.0,
                 interactive_runtime_estimate_hours=0.25,
                 cancel_margin_hours=None, cancel_resolution_timeout_hours=None):
```

Store them just after `self._cleanup = cleanup_provider_files` (around line 25):

```python
        self._fallback_strategy = fallback_strategy
        self._poll_interval_h = poll_interval_hours
        self._interactive_estimate_h = interactive_runtime_estimate_hours
        self._cancel_margin_h = cancel_margin_hours
        self._cancel_resolution_timeout_h = cancel_resolution_timeout_hours
```

Forward them in `job()` (extend the `Job(...)` call around lines 59-65):

```python
    def job(self, key, *, messages=None, requests=None, model=None, **litellm_kwargs) -> Job:
        return Job(
            self._backend.for_job(key), key,
            messages=messages, requests=requests, model=model,
            mode=self._mode, deadline=self._deadline,
            max_batch_turnaround_hours=self._max_turnaround,
            rerun_cmd=self._rerun_cmd, cleanup_provider_files=self._cleanup,
            fallback_strategy=self._fallback_strategy,
            poll_interval_hours=self._poll_interval_h,
            interactive_runtime_estimate_hours=self._interactive_estimate_h,
            cancel_margin_hours=self._cancel_margin_h,
            cancel_resolution_timeout_hours=self._cancel_resolution_timeout_h,
            **litellm_kwargs)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pipeline.py -v -k "speculative_config or defaults_to_conservative"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add llm_batch/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): thread speculative fallback config through to jobs"
```

---

## Task 11: End-to-end speculative cancel → re-issue test

A single test that drives the full lifecycle through `Job.execute()` across simulated poll runs: submit → wait → out-of-time cancel → cancelled → interactive re-issue. This guards the state transitions as an integration, not just per-method.

**Files:**
- Test: `tests/test_job_speculative.py`

- [ ] **Step 1: Write the test**

Add to `tests/test_job_speculative.py`:

```python
def test_full_speculative_lifecycle_cancel_then_reissue(tmp_path, monkeypatch):
    monkeypatch.setattr(job_module.providers, "provider_of", lambda m: "openai")
    monkeypatch.setattr(job_module.providers, "provider_supports_cancel", lambda p: True)
    monkeypatch.setattr(job_module.providers, "cleanup", lambda h: None)
    monkeypatch.setattr(
        job_module.providers, "submit_batch",
        lambda model, requests, **kw: BatchHandle(
            provider="openai", model=model, batch_job_id="b1", input_file_id="f1"))

    # Provider state evolves across polls: pending, then cancelled after we ask.
    state = {"phase": "pending"}
    monkeypatch.setattr(
        job_module.providers, "poll_batch",
        lambda h: PollResult(status=state["phase"]))
    monkeypatch.setattr(job_module.providers, "cancel",
                        lambda h: state.__setitem__("phase", "canceled"))
    monkeypatch.setattr(
        job_module.providers, "complete",
        lambda model, messages, **kw: CompletionResult(
            text="final", usage={}, cost={"charged_usd": 0.02, "interactive_usd": 0.04, "saved_usd": 0.0}))

    storage = FileStorage(str(tmp_path)).for_job("x")

    # Poll run 1 — submit with ample time (deadline far out).
    j = _spec_job(tmp_path, deadline=(_now() + timedelta(hours=72)).isoformat())
    assert j.execute() is None
    assert j.state == Job.SENT

    # Poll run 2 — reconstruct from storage with a now-tight deadline; out of time
    # → initiate cancel, transition to CANCEL_REQUESTED. (Rewrite the deadline to
    # simulate time passing close to the deadline.)
    snap = storage.snapshot()
    snap["deadline"] = (_now() + timedelta(hours=1)).isoformat()
    storage.save(snap)
    j2 = Job(storage)
    assert j2.execute() is None
    assert j2.state == Job.CANCEL_REQUESTED

    # Poll run 3 — provider now reports cancelled → re-issue interactively.
    j3 = Job(FileStorage(str(tmp_path)).for_job("x"))
    assert j3.execute() == "final"
    assert j3.state == Job.COMPLETED
    assert j3.snapshot()["actual_mode"] == "interactive"
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_job_speculative.py::test_full_speculative_lifecycle_cancel_then_reissue -v`
Expected: PASS

- [ ] **Step 3: Run the entire test suite**

Run: `pytest -v`
Expected: PASS (all existing + new tests).

- [ ] **Step 4: Commit**

```bash
git add tests/test_job_speculative.py
git commit -m "test: end-to-end speculative cancel-then-reissue lifecycle"
```

---

## Task 12: Documentation — README + public exports

Surface the new strategy in the README and confirm the public surface.

**Files:**
- Modify: `README.md`
- Check: `llm_batch/__init__.py` (no new public symbol is required — `fallback_strategy` is a `Pipeline`/`Job` constructor arg; confirm `Pipeline` and `Job` are already exported)

- [ ] **Step 1: Confirm the exports**

Run: `grep -n "Pipeline\|Job\|fallback" llm_batch/__init__.py`
Expected: `Pipeline` and `Job` already appear in `__all__` / imports. If `Pipeline` is exported, no code change is needed here — the new behavior is reached via its `fallback_strategy=` argument.

- [ ] **Step 2: Add a README subsection**

In `README.md`, under the fallback/strategy documentation, add:

```markdown
### Fallback strategies

`fallback_strategy` (on `Pipeline`/`Job`) controls how a near-deadline batch job avoids missing its deadline:

- `"conservative"` (default) — decide once at submit time: if `deadline − now < max_batch_turnaround_hours`, run interactively instead of batching. Safe, but forgoes the batch discount within a day or two of the deadline.
- `"speculative"` — submit the batch optimistically and only cancel + re-run interactively at the last responsible moment, dropping the effective margin to roughly `poll_interval + cancel_margin + interactive_runtime_estimate`. Captures the discount in the common (fast) case.

Speculative cancellation is delegated to litellm and supported for **OpenAI, Azure, and Vertex AI** (requires `litellm>=1.83.0`). For **Anthropic**, which litellm cannot cancel, a speculative job transparently degrades to `conservative` (with a one-time warning).

Tuning knobs (all hours): `poll_interval_hours`, `interactive_runtime_estimate_hours`, `cancel_margin_hours` (provider-aware default), `cancel_resolution_timeout_hours` (provider-aware backstop for a cancel that never settles).
```

- [ ] **Step 3: Commit**

```bash
git add README.md llm_batch/__init__.py
git commit -m "docs: document speculative fallback strategy in README"
```

---

## Self-Review Checklist (completed during plan authoring)

- **Spec coverage:** Architecture decision (litellm delegation) → Tasks 1-3. `_CANCELLABLE` gating + one-time downgrade → Tasks 1, 6. Non-blocking state machine (states A/B, no busy-wait) → Tasks 7-8. `cancel_resolution_timeout` backstop → Task 8. `poll_batch` distinguishes `canceled` from `failed` → Task 2. State persistence for re-issue (payload + kwargs) → Tasks 4, 6, 8. Config knobs → Tasks 4, 10. Poller advancing the new state → Task 9. Tests called out in the spec → distributed across Tasks 2, 6, 7, 8, 9, 11.
- **No placeholders:** every code step shows complete code; commands include expected output.
- **Type/name consistency:** `PollResult.status` adds `"canceled"` (Task 2) and is consumed in Tasks 7-9. `fallback_strategy`, `payload`, `litellm_kwargs`, `cancel_requested_at` are added to `_PERSISTED_FIELDS` (Task 4) and read/written in Tasks 6-8. `_finish_completed`, `_poll_conservative`, `_poll_speculative`, `_poll_cancel`, `_reissue_interactive`, `_can_afford_to_wait`, `_speculative_window_hours`, `_effective_strategy`, `_cancel_resolution_expired` are defined once and referenced consistently. Hours-suffixed knobs match across `Job` and `Pipeline`.
