import pytest

from llm_batch import job as job_module
from llm_batch import poller as poller_module
from llm_batch.job import Job, _now
from llm_batch.poller import Poller
from llm_batch.providers import BatchHandle, PollResult
from llm_batch.storage.file import FileStorage
from datetime import timedelta


@pytest.fixture
def fake_providers(monkeypatch):
    monkeypatch.setattr(job_module.providers, "provider_of", lambda m: "openai")
    monkeypatch.setattr(job_module.providers, "cleanup", lambda h: None)
    return monkeypatch


def _seed_sent_job(dir_path, key, rerun_cmd, submitted_at=None):
    """Write a SENT job state file directly."""
    handle = FileStorage(dir_path).for_job(key)
    handle.save({
        "key": key, "state": "sent", "mode": "batch", "model": "gpt-4o",
        "n_requests": 1, "bundle": False, "batch_job_id": "b1",
        "input_file_id": "f-in", "output_file_id": None,
        "submitted_at": submitted_at or _now().isoformat(),
        "completed_at": None, "deadline": None, "actual_mode": "batch",
        "response": None, "usage": {}, "cost": {}, "error_message": None,
        "poll_attempts": 0, "rerun_cmd": rerun_cmd, "cleanup_provider_files": True,
    })


def test_poll_all_advances_completed_pipeline_once(tmp_path, fake_providers):
    # two jobs in the SAME pipeline (same rerun_cmd), both complete this run
    p = str(tmp_path / "doc1")
    _seed_sent_job(p, "ocr", "python go.py doc1")
    _seed_sent_job(p, "extract", "python go.py doc1")
    fake_providers.setattr(job_module.providers, "poll_batch", lambda h: PollResult(
        status="completed", texts=["t"], usage={}, cost={}, output_file_id="o"))

    ran = []
    poller = Poller(FileStorage(str(tmp_path)), run_cmd=lambda argv: ran.append(argv))
    poller.poll_all()

    assert ran == [["python", "go.py", "doc1"]]      # de-duped to one rerun


def test_poll_all_advances_distinct_pipelines(tmp_path, fake_providers):
    _seed_sent_job(str(tmp_path / "doc1"), "ocr", "python go.py doc1")
    _seed_sent_job(str(tmp_path / "doc2"), "ocr", "python go.py doc2")
    fake_providers.setattr(job_module.providers, "poll_batch", lambda h: PollResult(
        status="completed", texts=["t"], usage={}, cost={}, output_file_id="o"))

    ran = []
    Poller(FileStorage(str(tmp_path)), run_cmd=lambda a: ran.append(a)).poll_all()
    assert sorted(ran) == [["python", "go.py", "doc1"], ["python", "go.py", "doc2"]]


def test_poll_all_no_rerun_when_still_pending(tmp_path, fake_providers):
    _seed_sent_job(str(tmp_path / "doc1"), "ocr", "python go.py doc1")
    fake_providers.setattr(job_module.providers, "poll_batch",
                           lambda h: PollResult(status="pending"))
    ran = []
    Poller(FileStorage(str(tmp_path)), run_cmd=lambda a: ran.append(a)).poll_all()
    assert ran == []


def test_poll_all_no_rerun_on_failure(tmp_path, fake_providers):
    _seed_sent_job(str(tmp_path / "doc1"), "ocr", "python go.py doc1")
    fake_providers.setattr(job_module.providers, "poll_batch",
                           lambda h: PollResult(status="failed", error_message="expired"))
    ran = []
    Poller(FileStorage(str(tmp_path)), run_cmd=lambda a: ran.append(a)).poll_all()
    assert ran == []
    # the job is now FAILED on disk
    assert FileStorage(str(tmp_path / "doc1")).for_job("ocr").load()
    assert FileStorage(str(tmp_path / "doc1")).for_job("ocr").get("state") == "failed"


def test_stale_job_marked_failed_and_cleaned(tmp_path, fake_providers):
    old = (_now() - timedelta(days=5)).isoformat()
    _seed_sent_job(str(tmp_path / "doc1"), "ocr", "python go.py doc1", submitted_at=old)
    cleaned = []
    fake_providers.setattr(job_module.providers, "cleanup",
                           lambda h: cleaned.append(h.batch_job_id))
    # poll_batch must NOT be called for a stale job
    fake_providers.setattr(job_module.providers, "poll_batch",
                           lambda h: (_ for _ in ()).throw(AssertionError("polled a stale job")))
    ran = []
    Poller(FileStorage(str(tmp_path)), max_batch_age_days=3,
           run_cmd=lambda a: ran.append(a)).poll_all()
    handle = FileStorage(str(tmp_path / "doc1")).for_job("ocr")
    handle.load()
    assert handle.get("state") == "failed"
    assert "stale" in handle.get("error_message")
    assert cleaned == ["b1"]
    assert ran == []


def test_sweep_orphans_delegates_per_provider(tmp_path, fake_providers):
    _seed_sent_job(str(tmp_path / "doc1"), "ocr", "python go.py doc1")
    seen = {}
    def fake_cleanup_orphans(provider, live_ids, older_than_days):
        seen[provider] = (set(live_ids), older_than_days)
        return []
    fake_providers.setattr(poller_module.providers, "cleanup_orphans", fake_cleanup_orphans)
    fake_providers.setattr(poller_module.providers, "provider_of", lambda m: "openai")

    Poller(FileStorage(str(tmp_path))).sweep_orphans(older_than_days=2)
    assert "openai" in seen
    live_ids, days = seen["openai"]
    assert days == 2
    assert "f-in" in live_ids        # live input file id excluded from deletion


def test_poller_advances_cancel_requested_jobs(tmp_path, fake_providers, monkeypatch):
    from llm_batch.providers import CompletionResult
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
