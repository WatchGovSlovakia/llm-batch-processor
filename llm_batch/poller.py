"""Poller — central advancement of SENT and CANCEL_REQUESTED jobs across a state tree."""

from __future__ import annotations

import shlex
import subprocess
from datetime import timedelta

from llm_batch import providers
from llm_batch.errors import BatchJobFailed
from llm_batch.job import Job, _now, _parse_iso


class Poller:
    """Scans SENT and CANCEL_REQUESTED jobs, advances completed pipelines, fails stale jobs.

    `backend` is a directory/namespace-level storage object exposing
    `iter_jobs()` (and `for_job`). `run_cmd` is injectable for testing; it
    defaults to running the command list with subprocess.
    """

    def __init__(self, backend, *, max_batch_age_days=3, run_cmd=None):
        self._backend = backend
        self._max_age = timedelta(days=max_batch_age_days)
        self._run = run_cmd if run_cmd is not None else _default_run

    def poll_all(self) -> dict:
        rerun_cmds: list[str] = []
        seen_cmds: set[str] = set()

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
            try:
                job.execute()  # polls once
            except BatchJobFailed:
                continue       # terminal failure → no rerun

            if job.state == Job.COMPLETED:
                cmd = handle.get("rerun_cmd")
                if cmd and cmd not in seen_cmds:
                    seen_cmds.add(cmd)
                    rerun_cmds.append(cmd)

        for cmd in rerun_cmds:
            self._run(shlex.split(cmd))

        return {"advanced": rerun_cmds}

    def _is_stale(self, handle) -> bool:
        submitted = handle.get("submitted_at")
        if not submitted:
            return False
        submitted_dt = _parse_iso(submitted)
        return (_now() - submitted_dt) > self._max_age

    def sweep_orphans(self, older_than_days: int = 3) -> dict:
        """Best-effort deletion of provider-side files with no live local state."""
        live_ids: set[str] = set()
        provider_set: set[str] = set()
        for handle in self._backend.iter_jobs():
            for field in ("batch_job_id", "input_file_id", "output_file_id"):
                value = handle.get(field)
                if value:
                    live_ids.add(value)
            model = handle.get("model")
            if model:
                provider_set.add(providers.provider_of(model))

        provider_set.add("openai")  # the one provider with a concrete sweep
        deleted: dict[str, list] = {}
        for provider in provider_set:
            deleted[provider] = providers.cleanup_orphans(
                provider, live_ids, older_than_days)
        return deleted


def _default_run(argv: list[str]) -> None:
    subprocess.run(argv, check=False)
