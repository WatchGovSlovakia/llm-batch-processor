"""Job — the unit of resumable work and its execute() state machine."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from llm_batch import providers
from llm_batch.errors import BatchJobFailed
from llm_batch.providers import BatchHandle

_LOG = logging.getLogger("llm_batch")
# Providers already warned about a speculative→conservative downgrade (process-wide,
# so the warning is emitted once, not once per poll).
_SPECULATIVE_DOWNGRADE_WARNED: set[str] = set()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now().isoformat()


def _parse_iso(value: str | None) -> datetime | None:
    if value is None:
        return None
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


_ZERO_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
_ZERO_COST = {"charged_usd": 0.0, "interactive_usd": 0.0, "saved_usd": 0.0}

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


# Fields reconstructed from storage when resuming a job.
_PERSISTED_FIELDS = (
    "key", "state", "mode", "model", "n_requests", "bundle", "batch_job_id",
    "input_file_id", "output_file_id", "submitted_at", "completed_at", "deadline",
    "actual_mode", "response", "usage", "cost", "finish_reason", "error_message",
    "poll_attempts", "rerun_cmd", "cleanup_provider_files",
    "fallback_strategy", "payload", "litellm_kwargs", "cancel_requested_at",
)


def _accumulate(target: dict, src: dict) -> None:
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        target[k] = target.get(k, 0) + (src or {}).get(k, 0)


def _accumulate_cost(target: dict, src: dict) -> None:
    for k in ("charged_usd", "interactive_usd", "saved_usd"):
        target[k] = target.get(k, 0.0) + (src or {}).get(k, 0.0)


class Job:
    """One or many requests, a model, a mode, and a storage handle."""

    NOT_SENT = "not_sent"
    SENT = "sent"
    CANCEL_REQUESTED = "cancel_requested"
    COMPLETED = "completed"
    FAILED = "failed"

    def __init__(self, storage, key=None, *, messages=None, requests=None,
                 model=None, mode="batch", deadline=None,
                 max_batch_turnaround_hours=24, rerun_cmd=None,
                 cleanup_provider_files=True,
                 fallback_strategy="conservative", poll_interval_hours=1.0,
                 interactive_runtime_estimate_hours=0.25,
                 cancel_margin_hours=None, cancel_resolution_timeout_hours=None,
                 **litellm_kwargs):
        if messages is not None and requests is not None:
            raise ValueError("pass either messages= or requests=, not both")

        self._storage = storage
        self._key = key
        self._is_bundle = requests is not None
        if requests is not None:
            self._payload = list(requests)
        elif messages is not None:
            self._payload = [messages]
        else:
            self._payload = None  # poll-only reconstruction

        self._model = model
        self._mode = mode
        self._deadline = deadline
        self._max_turnaround = max_batch_turnaround_hours
        self._rerun_cmd = rerun_cmd
        self._cleanup = cleanup_provider_files
        self._kw = litellm_kwargs
        self._fallback_strategy = fallback_strategy
        self._poll_interval_h = poll_interval_hours
        self._interactive_estimate_h = interactive_runtime_estimate_hours
        self._cancel_margin_h = cancel_margin_hours
        self._cancel_resolution_timeout_h = cancel_resolution_timeout_hours
        self._state: dict = {}

        if self._storage.load():
            self._state = {f: self._storage.get(f) for f in _PERSISTED_FIELDS}
            # Persisted values win for resumption.
            self._key = self._state.get("key", self._key)
            self._model = self._state.get("model") or self._model
            self._mode = self._state.get("mode") or self._mode
            self._deadline = self._state.get("deadline", self._deadline)
            if self._state.get("bundle") is not None:
                self._is_bundle = bool(self._state.get("bundle"))
            if self._state.get("rerun_cmd"):
                self._rerun_cmd = self._state.get("rerun_cmd")
            if self._state.get("cleanup_provider_files") is not None:
                self._cleanup = self._state.get("cleanup_provider_files")
            if self._payload is None and self._state.get("payload") is not None:
                self._payload = self._state.get("payload")
                self._is_bundle = bool(self._state.get("bundle"))
            if not self._kw and self._state.get("litellm_kwargs"):
                self._kw = dict(self._state.get("litellm_kwargs"))

    # --- public API -----------------------------------------------------

    @property
    def state(self) -> str:
        return self._state.get("state", self.NOT_SENT)

    @property
    def result(self):
        if self.state != self.COMPLETED:
            return None
        return self._state.get("response")

    @property
    def usage(self) -> dict:
        return dict(self._state.get('usage') or _ZERO_USAGE)

    @property
    def finish_reason(self) -> str | None:
        return self._state.get('finish_reason')

    def snapshot(self) -> dict:
        return dict(self._state)

    def execute(self):
        state = self.state
        if state == self.COMPLETED:
            return self._state.get("response")
        if state == self.FAILED:
            raise BatchJobFailed(
                self._state.get("error_message") or "job previously failed",
                self._state.get("batch_job_id"))
        if state == self.SENT:
            return self._poll()
        if state == self.CANCEL_REQUESTED:
            return self._poll_cancel()
        return self._submit()  # NOT_SENT

    def set_completed(self, text) -> None:
        """Mark this job COMPLETED with a preset result and no LLM call.

        Used when the caller already has the answer (e.g. fitz extracted the
        PDF text directly). `text` is stored verbatim as the response.
        """
        if self.state == self.COMPLETED:
            return
        state = self._base_state()
        state.update(state=self.COMPLETED, actual_mode="preset",
                     completed_at=_now_iso(), response=text)
        self._state = state
        self._save()

    def retry(self) -> None:
        """Reset a FAILED job to NOT_SENT so the next execute() resubmits."""
        if self.state != self.FAILED:
            return
        self._state["state"] = self.NOT_SENT
        self._state["error_message"] = None
        self._save()

    def mark_failed(self, message: str) -> None:
        """Force this job to FAILED and attempt provider cleanup (poller use)."""
        self._mark_failed(message, self._batch_handle(),
                          self._state.get("output_file_id"))

    # --- internals ------------------------------------------------------

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

    def _should_use_interactive(self) -> bool:
        if self._mode == "interactive":
            return True
        if self._deadline is None:
            return False
        deadline_dt = _parse_iso(self._deadline)
        if deadline_dt is None:
            return False
        hours_remaining = (deadline_dt - _now()).total_seconds() / 3600.0
        # Too little time for another worst-case batch round → go interactive now.
        return hours_remaining < self._max_turnaround

    def _base_state(self) -> dict:
        return {
            "key": self._key,
            "model": self._model,
            "mode": self._mode,
            "n_requests": len(self._payload) if self._payload else 0,
            "bundle": self._is_bundle,
            "deadline": self._deadline,
            "rerun_cmd": self._rerun_cmd,
            "cleanup_provider_files": self._cleanup,
            "submitted_at": _now_iso(),
            "completed_at": None,
            "batch_job_id": None,
            "input_file_id": None,
            "output_file_id": None,
            "response": None,
            "usage": dict(_ZERO_USAGE),
            "cost": dict(_ZERO_COST),
            "finish_reason": None,
            "error_message": None,
            "poll_attempts": 0,
            "fallback_strategy": "conservative",
            "payload": None,
            "litellm_kwargs": None,
            "cancel_requested_at": None,
        }

    def _save(self) -> None:
        self._storage.save(self._state)

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

    def _submit_interactive(self):
        texts = []
        usage = dict(_ZERO_USAGE)
        cost = dict(_ZERO_COST)
        last_finish_reason = None
        try:
            for messages in self._payload:
                cr = providers.complete(self._model, messages, **self._kw)
                texts.append(cr.text)
                _accumulate(usage, cr.usage)
                _accumulate_cost(cost, cr.cost)
                last_finish_reason = cr.finish_reason
        except providers.TransientProviderError:
            return None  # leave state NOT_SENT; retried next run

        response = texts if self._is_bundle else texts[0]
        state = self._base_state()
        state.update(state=self.COMPLETED, actual_mode="interactive",
                     completed_at=_now_iso(), response=response,
                     usage=usage, cost=cost,
                     finish_reason=None if self._is_bundle else last_finish_reason)
        self._state = state
        self._save()
        return response

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

    def _batch_handle(self) -> BatchHandle:
        return BatchHandle(
            provider=providers.provider_of(self._model),
            model=self._model,
            batch_job_id=self._state.get("batch_job_id"),
            input_file_id=self._state.get("input_file_id"),
            output_file_id=self._state.get("output_file_id"),
            n_requests=self._state.get("n_requests"),
        )

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
            finish_reason=None if self._is_bundle else result.finish_reason,
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
        # Safe to re-cancel if we crash before the save below: a repeat cancel_batch on an already-cancelling job is idempotent (no fee).
        try:
            providers.cancel(handle)
        except providers.TransientProviderError:
            self._save()  # cancel failed transiently; retry next poll, stay SENT
            return None
        self._state["state"] = self.CANCEL_REQUESTED
        self._state["cancel_requested_at"] = _now_iso()
        self._save()
        return None

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
        last_finish_reason = None
        try:
            for messages in self._payload:
                cr = providers.complete(self._model, messages, **self._kw)
                texts.append(cr.text)
                _accumulate(usage, cr.usage)
                _accumulate_cost(cost, cr.cost)
                last_finish_reason = cr.finish_reason
        except providers.TransientProviderError:
            self._save()  # stay CANCEL_REQUESTED; retry interactive next poll
            return None

        response = texts if self._is_bundle else texts[0]
        self._state.update(
            state=self.COMPLETED, actual_mode="interactive",
            completed_at=_now_iso(), response=response, usage=usage, cost=cost,
            finish_reason=None if self._is_bundle else last_finish_reason,
            error_message=None)
        self._save()
        if self._cleanup:
            try:
                providers.cleanup(handle)  # remove the (cancelled) batch artifacts
            except Exception:
                pass
        return response

    def _mark_failed(self, message, handle=None, output_file_id=None):
        self._state["state"] = self.FAILED
        self._state["error_message"] = message
        self._state["completed_at"] = _now_iso()
        if output_file_id:
            self._state["output_file_id"] = output_file_id
        self._save()
        if handle is not None and self._cleanup:
            if output_file_id:
                handle.output_file_id = output_file_id
            try:
                providers.cleanup(handle)
            except Exception:
                pass
