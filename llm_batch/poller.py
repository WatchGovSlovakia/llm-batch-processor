"""Poller — central advancement of SENT and CANCEL_REQUESTED jobs across a state tree."""

from __future__ import annotations

import logging
import shlex
import subprocess
from datetime import timedelta

from llm_batch import providers
from llm_batch.errors import BatchJobFailed
from llm_batch.job import Job, _now, _parse_iso

_LOG = logging.getLogger("llm_batch")


class Poller:
    """Scans SENT and CANCEL_REQUESTED jobs, advances completed pipelines, fails stale jobs.

    `backend` is a directory/namespace-level storage object exposing
    `iter_jobs()` (and `for_job`). `run_cmd` is injectable for testing; it
    defaults to running the command list with subprocess. `image_host` (the
    one the jobs were submitted with) lets the poller delete a finished job's
    hosted images and sweep orphaned ones.

    `credentials` is either one {provider: ProviderCredentials} map for every job, or a
    callable `credentials_ref -> {provider: ProviderCredentials} | None`, called per job
    with the `credentials_ref` the job was created with (None for jobs created without
    one). The callable lets one poller serve jobs submitted under different keys.
    """

    def __init__(self, backend, *, max_batch_age_days=3, run_cmd=None, image_host=None,
                 credentials=None):
        self._backend = backend
        self._image_host = image_host
        self._credentials = credentials
        self._max_age = timedelta(days=max_batch_age_days)
        self._run = run_cmd if run_cmd is not None else _default_run

    def _credentials_for(self, credentials_ref):
        if callable(self._credentials):
            return self._credentials(credentials_ref)
        return self._credentials

    def poll_all(self) -> dict:
        """Poll every SENT / CANCEL_REQUESTED job once, then run the reruns of completed ones.

        Returns {"advanced": [rerun_cmd, ...], "errors": [{"key", "rerun_cmd", "error"}, ...]}.
        A job that raises (a missing or revoked key, an unexpected provider answer) is
        logged and listed under "errors" and left as it is, to be retried next run; the
        other jobs are still polled.
        """
        rerun_cmds: list[str] = []
        seen_cmds: set[str] = set()
        errors: list[dict] = []

        for handle in self._backend.iter_jobs():
            state = handle.get("state")
            if state not in (Job.SENT, Job.CANCEL_REQUESTED):
                continue

            try:
                job = Job(handle, image_host=self._image_host,
                          credentials=self._credentials_for(handle.get("credentials_ref")))
                # Staleness only force-fails SENT jobs; a CANCEL_REQUESTED job is
                # bounded instead by its cancel_resolution_timeout (re-issues interactively).
                if state == Job.SENT and self._is_stale(handle):
                    job.mark_failed(f"stale: no completion within {self._max_age.days}d")
                    continue
                job.execute()  # polls once
            except BatchJobFailed:
                continue       # terminal failure → no rerun
            except Exception as exc:
                _LOG.warning("poll of job %r failed, retried next run: %s: %s",
                             handle.get("key"), type(exc).__name__, exc)
                errors.append({"key": handle.get("key"), "rerun_cmd": handle.get("rerun_cmd"),
                               "error": f"{type(exc).__name__}: {exc}"})
                continue

            if job.state == Job.COMPLETED:
                cmd = handle.get("rerun_cmd")
                if cmd and cmd not in seen_cmds:
                    seen_cmds.add(cmd)
                    rerun_cmds.append(cmd)

        for cmd in rerun_cmds:
            self._run(shlex.split(cmd))

        return {"advanced": rerun_cmds, "errors": errors}

    def _is_stale(self, handle) -> bool:
        submitted = handle.get("submitted_at")
        if not submitted:
            return False
        submitted_dt = _parse_iso(submitted)
        return (_now() - submitted_dt) > self._max_age

    def sweep_orphans(self, older_than_days: int = 3) -> dict:
        """Best-effort deletion of provider-side files with no live local state,
        and of hosted images (key "images") no unfinished job still needs."""
        live_ids: set[str] = set()
        live_images: set[str] = set()
        provider_set: set[str] = set()
        refs: set = set()
        for handle in self._backend.iter_jobs():
            refs.add(handle.get("credentials_ref"))
            for field in ("batch_job_id", "input_file_id", "output_file_id"):
                value = handle.get(field)
                if value:
                    live_ids.add(value)
            if handle.get("state") in (Job.SENT, Job.CANCEL_REQUESTED):
                live_images.update(handle.get("hosted_images") or ())
            model = handle.get("model")
            if model:
                provider_set.add(providers.provider_of(model))

        provider_set.add("openai")  # the one provider with a concrete sweep
        deleted: dict[str, list] = {provider: [] for provider in provider_set}
        for credentials in self._sweep_credentials(refs):
            for provider in provider_set:
                deleted[provider] += providers.cleanup_orphans(
                    provider, live_ids, older_than_days, credentials=credentials)
        if self._image_host is not None:
            try:
                deleted["images"] = self._image_host.delete_orphans(live_images, older_than_days)
            except Exception:
                deleted["images"] = []
        return deleted

    def _sweep_credentials(self, refs: set) -> list:
        """Distinct credential maps to sweep with: the fixed one, or one per known credentials_ref.

        With a callable, only workspaces that still have a job in this backend are swept.
        """
        if not callable(self._credentials):
            return [self._credentials]
        unique: dict = {}
        for ref in sorted(refs, key=lambda r: (r is not None, str(r))):
            try:
                credentials = self._credentials(ref)
            except Exception as exc:
                _LOG.warning("orphan sweep: no credentials for ref %r: %s", ref, exc)
                continue
            identity = tuple(sorted((p, c.api_key, c.api_base) for p, c in (credentials or {}).items()))
            unique.setdefault(identity, credentials)
        return list(unique.values())


def _default_run(argv: list[str]) -> None:
    subprocess.run(argv, check=False)
