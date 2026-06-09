import pytest

from llm_batch import job as job_module
from llm_batch.job import Job, _now
from llm_batch.providers import BatchHandle, CompletionResult
from llm_batch.storage.file import FileStorage
from datetime import timedelta


@pytest.fixture
def fake_providers(monkeypatch):
    calls = {"submit": 0, "complete": 0}

    def submit_batch(model, requests, **kw):
        calls["submit"] += 1
        return BatchHandle(provider="openai", model=model, batch_job_id="b1", input_file_id="f1")

    def complete(model, messages, **kw):
        calls["complete"] += 1
        return CompletionResult(text="now", usage={}, cost={"charged_usd": 0.02, "interactive_usd": 0.02, "saved_usd": 0.0})

    monkeypatch.setattr(job_module.providers, "submit_batch", submit_batch)
    monkeypatch.setattr(job_module.providers, "complete", complete)
    monkeypatch.setattr(job_module.providers, "provider_of", lambda m: "openai")
    return calls


def _job(tmp_path, deadline, mode="batch", max_h=24):
    return Job(FileStorage(str(tmp_path)).for_job("x"), "x",
               messages=[{"role": "user", "content": "hi"}],
               model="gpt-4o", mode=mode, deadline=deadline,
               max_batch_turnaround_hours=max_h)


def test_no_deadline_stays_batch(tmp_path, fake_providers):
    job = _job(tmp_path, deadline=None)
    assert job.execute() is None
    assert fake_providers["submit"] == 1
    assert job.snapshot()["actual_mode"] == "batch"


def test_ample_time_stays_batch(tmp_path, fake_providers):
    deadline = (_now() + timedelta(hours=72)).isoformat()
    job = _job(tmp_path, deadline=deadline)
    assert job.execute() is None
    assert fake_providers["submit"] == 1
    assert job.snapshot()["actual_mode"] == "batch"


def test_tight_deadline_falls_back_to_interactive(tmp_path, fake_providers):
    deadline = (_now() + timedelta(hours=3)).isoformat()   # < 24h margin
    job = _job(tmp_path, deadline=deadline)
    result = job.execute()
    assert result == "now"
    assert fake_providers["submit"] == 0
    assert fake_providers["complete"] == 1
    assert job.snapshot()["actual_mode"] == "interactive"
    assert job.state == Job.COMPLETED


def test_custom_margin_respected(tmp_path, fake_providers):
    deadline = (_now() + timedelta(hours=10)).isoformat()
    job = _job(tmp_path, deadline=deadline, max_h=6)   # 10h > 6h margin → still batch
    assert job.execute() is None
    assert fake_providers["submit"] == 1


def test_interactive_mode_ignores_deadline(tmp_path, fake_providers):
    deadline = (_now() + timedelta(hours=72)).isoformat()
    job = _job(tmp_path, deadline=deadline, mode="interactive")
    assert job.execute() == "now"
    assert fake_providers["complete"] == 1
