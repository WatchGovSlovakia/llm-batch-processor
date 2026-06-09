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

# Fields reconstructed from storage when resuming a job.
_PERSISTED_FIELDS = (
    "key", "state", "mode", "model", "n_requests", "bundle", "batch_job_id",
    "input_file_id", "output_file_id", "submitted_at", "completed_at", "deadline",
    "actual_mode", "response", "usage", "cost", "error_message", "poll_attempts",
    "rerun_cmd", "cleanup_provider_files",
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

        if self._should_use_interactive():
            return self._submit_interactive()
        return self._submit_batch()

    def _submit_interactive(self):
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
            return None  # leave state NOT_SENT; retried next run

        response = texts if self._is_bundle else texts[0]
        state = self._base_state()
        state.update(state=self.COMPLETED, actual_mode="interactive",
                     completed_at=_now_iso(), response=response,
                     usage=usage, cost=cost)
        self._state = state
        self._save()
        return response

    def _submit_batch(self):
        try:
            handle = providers.submit_batch(self._model, self._payload, **self._kw)
        except providers.TransientProviderError:
            return None  # leave state NOT_SENT; retried next run

        state = self._base_state()
        state.update(state=self.SENT, actual_mode="batch",
                     batch_job_id=handle.batch_job_id,
                     input_file_id=handle.input_file_id)
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
        result = providers.poll_batch(handle)

        if result.status in ("pending", "transient_error"):
            self._state["poll_attempts"] = self._state.get("poll_attempts", 0) + 1
            if result.status == "transient_error":
                self._state["error_message"] = result.error_message
            self._save()
            return None

        if result.status == "failed":
            self._mark_failed(result.error_message, handle, result.output_file_id)
            raise BatchJobFailed(result.error_message or "batch failed",
                                 handle.batch_job_id)

        # completed
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
