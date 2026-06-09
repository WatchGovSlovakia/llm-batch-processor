import pytest

from llm_batch import job as job_module
from llm_batch.job import Job
from llm_batch.errors import BatchJobFailed
from llm_batch.providers import BatchHandle, PollResult
from llm_batch.storage.file import FileStorage


@pytest.fixture
def fake_providers(monkeypatch):
    calls = {"submit": 0}

    def submit_batch(model, requests, **kw):
        calls["submit"] += 1
        return BatchHandle(provider="openai", model=model, batch_job_id="b1", input_file_id="f1")

    monkeypatch.setattr(job_module.providers, "submit_batch", submit_batch)
    monkeypatch.setattr(job_module.providers, "provider_of", lambda m: "openai")
    monkeypatch.setattr(job_module.providers, "cleanup", lambda h: None)
    return calls


def test_set_completed_marks_done_without_llm(tmp_path, fake_providers):
    job = Job(FileStorage(str(tmp_path)).for_job("ocr"), "ocr",
              requests=[[{"role": "user", "content": "p1"}]], model="gpt-4o", mode="batch")
    job.set_completed("already extracted text")
    assert job.state == Job.COMPLETED
    assert job.execute() == "already extracted text"     # no submit
    assert fake_providers["submit"] == 0
    assert job.snapshot()["actual_mode"] == "preset"


def test_set_completed_persists_across_reload(tmp_path, fake_providers):
    Job(FileStorage(str(tmp_path)).for_job("ocr"), "ocr",
        requests=[[{"role": "user", "content": "p1"}]], model="gpt-4o").set_completed("text")
    reloaded = Job(FileStorage(str(tmp_path)).for_job("ocr"), "ocr")
    assert reloaded.execute() == "text"


def test_set_completed_noop_if_already_completed(tmp_path, fake_providers):
    handle = FileStorage(str(tmp_path)).for_job("ocr")
    job = Job(handle, "ocr", messages=[{"role": "user", "content": "p1"}], model="gpt-4o")
    job.set_completed("first")
    job.set_completed("second")            # ignored
    assert job.execute() == "first"


def test_retry_resets_failed_to_not_sent(tmp_path, fake_providers, monkeypatch):
    monkeypatch.setattr(job_module.providers, "poll_batch",
                        lambda h: PollResult(status="failed", error_message="expired"))
    job = Job(FileStorage(str(tmp_path)).for_job("x"), "x",
              messages=[{"role": "user", "content": "hi"}], model="gpt-4o", mode="batch")
    job.execute()                                   # SENT
    with pytest.raises(BatchJobFailed):
        job.execute()                               # FAILED
    job.retry()
    assert job.state == Job.NOT_SENT
    assert job.snapshot()["error_message"] is None
    assert job.execute() is None                    # resubmits
    assert fake_providers["submit"] == 2


def test_retry_noop_if_not_failed(tmp_path, fake_providers):
    job = Job(FileStorage(str(tmp_path)).for_job("x"), "x",
              messages=[{"role": "user", "content": "hi"}], model="gpt-4o", mode="batch")
    job.execute()                                   # SENT
    job.retry()                                     # ignored (not FAILED)
    assert job.state == Job.SENT
