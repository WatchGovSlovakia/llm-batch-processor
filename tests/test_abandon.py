import pytest

from llm_batch import job as job_module
from llm_batch.job import Job
from llm_batch.pipeline import Pipeline
from llm_batch.providers import BatchHandle, TransientProviderError
from llm_batch.storage.file import FileStorage


@pytest.fixture
def prov(monkeypatch):
    """Mock providers: records cancel/cleanup calls; provider taken from the model prefix."""
    calls = {"cancel": [], "cleanup": []}
    monkeypatch.setattr(job_module.providers, "provider_of", lambda m: m.split("/")[0])
    monkeypatch.setattr(job_module.providers, "provider_supports_cancel",
                        lambda p: p in ("openai", "azure", "vertex_ai"))
    monkeypatch.setattr(job_module.providers, "cancel", lambda h: calls["cancel"].append(h))
    monkeypatch.setattr(job_module.providers, "cleanup", lambda h: calls["cleanup"].append(h))
    monkeypatch.setattr(
        job_module.providers, "submit_batch",
        lambda model, requests, **kw: BatchHandle(
            provider=model.split("/")[0], model=model, batch_job_id="b-" + model,
            input_file_id="f1"))
    return calls


def _sent_job(tmp_path, key="x", model="openai/gpt-4o"):
    job = Job(FileStorage(str(tmp_path)).for_job(key), key,
              messages=[{"role": "user", "content": "hi"}], model=model)
    assert job.execute() is None
    assert job.state == Job.SENT
    return job


def test_abandon_sent_cancellable_cancels_and_fails(tmp_path, prov):
    job = _sent_job(tmp_path)
    assert job.abandon("config changed") is True
    assert len(prov["cancel"]) == 1
    assert prov["cancel"][0].batch_job_id == "b-openai/gpt-4o"
    assert job.state == Job.FAILED
    assert job.snapshot()["error_message"] == "config changed"


def test_abandon_sent_non_cancellable_fails_without_cancel(tmp_path, prov):
    job = _sent_job(tmp_path, model="anthropic/claude")
    assert job.abandon() is False
    assert prov["cancel"] == []
    assert job.state == Job.FAILED
    assert job.snapshot()["error_message"] == "abandoned by caller"


def test_abandon_cancel_requested_does_not_recancel(tmp_path, prov):
    job = _sent_job(tmp_path)
    job._state["state"] = Job.CANCEL_REQUESTED
    assert job.abandon("gone") is True
    assert prov["cancel"] == []
    assert job.state == Job.FAILED
    assert job.snapshot()["error_message"] == "gone"


def test_abandon_noop_for_not_sent_completed_failed(tmp_path, prov):
    storage = FileStorage(str(tmp_path))
    fresh = Job(storage.for_job("n"), "n", messages=[{"role": "user", "content": "a"}],
                model="openai/gpt-4o")
    assert fresh.abandon() is False
    assert fresh.state == Job.NOT_SENT

    done = Job(storage.for_job("c"), "c", model="openai/gpt-4o")
    done.set_completed("answer")
    assert done.abandon() is False
    assert done.state == Job.COMPLETED
    assert done.result == "answer"

    failed = _sent_job(tmp_path, key="f")
    failed.mark_failed("boom")
    prov["cancel"].clear()
    assert failed.abandon("later") is False
    assert failed.snapshot()["error_message"] == "boom"
    assert prov["cancel"] == []


def test_abandon_cancel_error_propagates_and_keeps_state(tmp_path, prov, monkeypatch):
    job = _sent_job(tmp_path)

    def boom(handle):
        raise TransientProviderError("rate limited")
    monkeypatch.setattr(job_module.providers, "cancel", boom)
    with pytest.raises(TransientProviderError):
        job.abandon()
    assert job.state == Job.SENT
    reloaded = Job(FileStorage(str(tmp_path)).for_job("x"), "x")
    assert reloaded.state == Job.SENT


def test_abandon_all_returns_only_cancelled_keys(tmp_path, prov):
    p = Pipeline(state_dir=str(tmp_path), mode="batch")
    for key, model in (("live", "openai/gpt-4o"), ("anth", "anthropic/claude")):
        p.job(key, messages=[{"role": "user", "content": "hi"}], model=model).execute()
    p.job("done").set_completed("ok")
    p.job("idle", messages=[{"role": "user", "content": "hi"}], model="openai/gpt-4o")
    # a never-executed job leaves no persisted state, so it is not enumerated

    assert sorted(p.abandon_all("cfg")) == ["live"]
    assert len(prov["cancel"]) == 1

    assert p.job("live").state == Job.FAILED
    assert p.job("live").snapshot()["error_message"] == "cfg"
    assert p.job("anth").state == Job.FAILED
    assert p.job("done").state == Job.COMPLETED
    assert p.job("done").result == "ok"
