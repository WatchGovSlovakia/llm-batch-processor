"""Pipeline — shared state dir + config across a multi-step job sequence."""

from __future__ import annotations

from datetime import timedelta

from llm_batch.job import Job, _now, _now_iso
from llm_batch.storage.file import FileStorage


class Pipeline:
    """Convenience wrapper that mints namespaced Jobs over one state dir.

    A bare Job works standalone; Pipeline just removes boilerplate and records
    the deadline anchor (`pipeline_started_at`) once in pipeline.json.
    """

    def __init__(self, state_dir, *, mode="batch", deadline=None,
                 time_budget_hours=None, max_batch_turnaround_hours=24,
                 rerun_cmd=None, storage=None, cleanup_provider_files=True,
                 fallback_strategy="conservative", poll_interval_hours=1.0,
                 interactive_runtime_estimate_hours=0.25,
                 cancel_margin_hours=None, cancel_resolution_timeout_hours=None):
        self._state_dir = state_dir
        self._mode = mode
        self._max_turnaround = max_batch_turnaround_hours
        self._rerun_cmd = rerun_cmd
        self._cleanup = cleanup_provider_files
        self._fallback_strategy = fallback_strategy
        self._poll_interval_h = poll_interval_hours
        self._interactive_estimate_h = interactive_runtime_estimate_hours
        self._cancel_margin_h = cancel_margin_hours
        self._cancel_resolution_timeout_h = cancel_resolution_timeout_hours
        self._backend = storage if storage is not None else FileStorage(state_dir)
        self._meta = self._load_or_init_meta(deadline, time_budget_hours)
        self._deadline = self._meta.get("deadline")

    def _load_or_init_meta(self, deadline, time_budget_hours) -> dict:
        handle = self._backend.for_meta()
        if handle.load():
            return handle.snapshot()
        started = _now()
        resolved_deadline = deadline
        if resolved_deadline is None and time_budget_hours is not None:
            resolved_deadline = (started + timedelta(hours=time_budget_hours)).isoformat()
        meta = {
            "pipeline_started_at": started.isoformat(),
            "deadline": resolved_deadline,
            "mode": self._mode,
            "rerun_cmd": self._rerun_cmd,
        }
        handle.save(meta)
        return meta

    def meta(self) -> dict:
        return dict(self._meta)

    def save_meta(self, extra: dict) -> None:
        """Merge extra fields into pipeline.json (e.g. pdf_url, domain deadline)."""
        self._meta.update(extra)
        # Keep the active deadline in sync so jobs minted after save_meta()
        # inherit the updated deadline (otherwise the fallback uses a stale one).
        self._deadline = self._meta.get("deadline")
        self._backend.for_meta().save(self._meta)

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

    def cost_summary(self, job_keys=None) -> dict:
        """Cost triple across this pipeline's jobs.

        job_keys restricts the total to just those keys (e.g. one
        reprocessing pass's own jobs) instead of every job ever persisted
        to this state dir.
        """
        wanted = set(job_keys) if job_keys is not None else None
        total = {"charged_usd": 0.0, "interactive_usd": 0.0, "saved_usd": 0.0}
        for handle in self._backend.iter_jobs():
            if wanted is not None and handle.get("key") not in wanted:
                continue
            cost = handle.get("cost") or {}
            for k in total:
                total[k] += cost.get(k, 0.0)
        return total

    def usage_summary(self, job_keys=None) -> dict:
        """Usage totals across this pipeline's jobs, optionally filtered to job_keys."""
        wanted = set(job_keys) if job_keys is not None else None
        total = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'requests': 0}
        for handle in self._backend.iter_jobs():
            if wanted is not None and handle.get("key") not in wanted:
                continue
            usage = handle.get('usage') or {}
            if not usage:
                continue
            for k in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                total[k] += usage.get(k, 0)
            total['requests'] += 1
        return total

    def job_keys(self) -> list[str]:
        """Every job key persisted in this pipeline's state dir (any state)."""
        return [handle.get("key") for handle in self._backend.iter_jobs()]

    def abandon_all(self, reason: str = "abandoned by caller") -> list[str]:
        """Abandon every job in this pipeline; return keys a provider cancel was issued for.

        Stops at the first exception from Job.abandon (which propagates).
        """
        cancelled = []
        for key in self.job_keys():
            if key is None:
                continue  # malformed state file with no key
            if self.job(key).abandon(reason):
                cancelled.append(key)
        return cancelled

    def finish(self) -> None:
        """Delete this pipeline's local state files."""
        for handle in list(self._backend.iter_jobs()):
            handle.finish()
        self._backend.for_meta().finish()
