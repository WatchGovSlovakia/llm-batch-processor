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
