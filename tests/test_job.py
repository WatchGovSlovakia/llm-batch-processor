import pytest

from llm_batch import job as job_module
from llm_batch.job import Job
from llm_batch.errors import BatchJobFailed
from llm_batch.providers import BatchHandle, PollResult, CompletionResult
from llm_batch.storage.file import FileStorage


@pytest.fixture
def fake_providers(monkeypatch):
    """Stub the provider layer for Job tests; configurable per test."""
    calls = {"submit": 0, "poll": 0, "complete": 0, "cleanup": [],
             "submit_kwargs": None, "complete_kwargs": None}

    def submit_batch(model, requests, **kw):
        calls["submit"] += 1
        calls["submit_kwargs"] = kw
        return BatchHandle(provider="openai", model=model,
                           batch_job_id="batches/job-1", input_file_id="files/in-1")

    def complete(model, messages, **kw):
        calls["complete"] += 1
        calls["complete_kwargs"] = kw
        return CompletionResult(text="interactive answer",
                                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                                cost={"charged_usd": 0.02, "interactive_usd": 0.02, "saved_usd": 0.0})

    def cleanup(handle):
        calls["cleanup"].append(handle.batch_job_id)

    monkeypatch.setattr(job_module.providers, "submit_batch", submit_batch)
    monkeypatch.setattr(job_module.providers, "complete", complete)
    monkeypatch.setattr(job_module.providers, "cleanup", cleanup)
    monkeypatch.setattr(job_module.providers, "provider_of", lambda m: "openai")
    return calls


def _handle(tmp_path, key="x"):
    return FileStorage(str(tmp_path)).for_job(key)


def test_interactive_single_completes_immediately(tmp_path, fake_providers):
    job = Job(_handle(tmp_path), "x", messages=[{"role": "user", "content": "hi"}],
              model="gpt-4o", mode="interactive")
    result = job.execute()
    assert result == "interactive answer"
    assert job.state == Job.COMPLETED
    assert fake_providers["complete"] == 1
    assert fake_providers["submit"] == 0


def test_interactive_bundle_returns_list(tmp_path, fake_providers):
    job = Job(_handle(tmp_path), "x",
              requests=[[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]],
              model="gpt-4o", mode="interactive")
    result = job.execute()
    assert result == ["interactive answer", "interactive answer"]
    assert fake_providers["complete"] == 2


def test_batch_submit_returns_none_and_persists_sent(tmp_path, fake_providers):
    job = Job(_handle(tmp_path), "x", messages=[{"role": "user", "content": "hi"}],
              model="gpt-4o", mode="batch")
    assert job.execute() is None
    assert job.state == Job.SENT
    assert job.snapshot()["batch_job_id"] == "batches/job-1"
    assert job.snapshot()["actual_mode"] == "batch"
    assert job.snapshot()["n_requests"] == 1


def test_execute_is_idempotent_one_submit(tmp_path, fake_providers, monkeypatch):
    monkeypatch.setattr(job_module.providers, "poll_batch",
                        lambda h: PollResult(status="pending"))
    job = Job(_handle(tmp_path), "x", messages=[{"role": "user", "content": "hi"}],
              model="gpt-4o", mode="batch")
    assert job.execute() is None      # submits
    assert job.execute() is None      # polls (pending), does NOT resubmit
    assert fake_providers["submit"] == 1
    assert job.snapshot()["poll_attempts"] == 1


def test_batch_poll_completed_single(tmp_path, fake_providers, monkeypatch):
    monkeypatch.setattr(job_module.providers, "poll_batch", lambda h: PollResult(
        status="completed", texts=["the text"],
        usage={"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        cost={"charged_usd": 0.01, "interactive_usd": 0.02, "saved_usd": 0.01},
        output_file_id="files/out-1"))
    job = Job(_handle(tmp_path), "x", messages=[{"role": "user", "content": "hi"}],
              model="gpt-4o", mode="batch")
    job.execute()                       # submit -> SENT
    result = job.execute()              # poll -> COMPLETED
    assert result == "the text"
    assert job.state == Job.COMPLETED
    assert job.snapshot()["response"] == "the text"
    assert job.snapshot()["cost"]["saved_usd"] == 0.01
    assert fake_providers["cleanup"] == ["batches/job-1"]   # cleanup ran after persist


def test_batch_poll_completed_bundle_returns_list(tmp_path, fake_providers, monkeypatch):
    monkeypatch.setattr(job_module.providers, "poll_batch", lambda h: PollResult(
        status="completed", texts=["p1", "p2"],
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        cost={"charged_usd": 0.0, "interactive_usd": 0.0, "saved_usd": 0.0},
        output_file_id="files/out-1"))
    job = Job(_handle(tmp_path), "x",
              requests=[[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]],
              model="gpt-4o", mode="batch")
    job.execute()
    assert job.execute() == ["p1", "p2"]


def test_completed_is_terminal_no_api(tmp_path, fake_providers, monkeypatch):
    monkeypatch.setattr(job_module.providers, "poll_batch", lambda h: PollResult(
        status="completed", texts=["done"], usage={}, cost={}, output_file_id="o"))
    handle = _handle(tmp_path)
    job = Job(handle, "x", messages=[{"role": "user", "content": "hi"}], model="gpt-4o", mode="batch")
    job.execute(); job.execute()                      # -> COMPLETED
    # a fresh Job loaded from the same storage returns the stored result, no poll
    def boom(h):
        raise AssertionError("must not poll a completed job")
    monkeypatch.setattr(job_module.providers, "poll_batch", boom)
    reloaded = Job(FileStorage(str(tmp_path)).for_job("x"), "x")
    assert reloaded.execute() == "done"
    assert reloaded.result == "done"


def test_transient_poll_keeps_sent_and_bumps_attempts(tmp_path, fake_providers, monkeypatch):
    monkeypatch.setattr(job_module.providers, "poll_batch", lambda h: PollResult(
        status="transient_error", error_message="429"))
    job = Job(_handle(tmp_path), "x", messages=[{"role": "user", "content": "hi"}],
              model="gpt-4o", mode="batch")
    job.execute()                       # SENT
    assert job.execute() is None        # transient
    assert job.state == Job.SENT        # NOT failed
    assert job.snapshot()["poll_attempts"] == 1


def test_provider_failed_makes_job_failed_and_sticky(tmp_path, fake_providers, monkeypatch):
    monkeypatch.setattr(job_module.providers, "poll_batch", lambda h: PollResult(
        status="failed", error_message="batch expired", output_file_id=None))
    job = Job(_handle(tmp_path), "x", messages=[{"role": "user", "content": "hi"}],
              model="gpt-4o", mode="batch")
    job.execute()                       # SENT
    with pytest.raises(BatchJobFailed):
        job.execute()                   # poll -> FAILED + raise
    assert job.state == Job.FAILED
    # sticky: a reloaded job re-raises without polling
    reloaded = Job(FileStorage(str(tmp_path)).for_job("x"), "x")
    with pytest.raises(BatchJobFailed):
        reloaded.execute()


def test_resume_across_instances(tmp_path, fake_providers, monkeypatch):
    # First process: submit.
    Job(_handle(tmp_path), "x", messages=[{"role": "user", "content": "hi"}],
        model="gpt-4o", mode="batch").execute()
    # Second process: poll-only reconstruction (no messages), completes.
    monkeypatch.setattr(job_module.providers, "poll_batch", lambda h: PollResult(
        status="completed", texts=["resumed"], usage={}, cost={}, output_file_id="o"))
    resumed = Job(FileStorage(str(tmp_path)).for_job("x"), key=None)
    assert resumed.execute() == "resumed"
    assert resumed.snapshot()["model"] == "gpt-4o"      # model recovered from state


def test_persist_happens_before_cleanup(tmp_path, fake_providers, monkeypatch):
    order = []
    handle = _handle(tmp_path)
    monkeypatch.setattr(job_module.providers, "poll_batch", lambda h: PollResult(
        status="completed", texts=["t"], usage={}, cost={}, output_file_id="o"))
    monkeypatch.setattr(job_module.providers, "cleanup",
                        lambda h: order.append("cleanup"))
    job = Job(handle, "x", messages=[{"role": "user", "content": "hi"}],
              model="gpt-4o", mode="batch")
    job.execute()
    # wrap save to record ordering
    real_save = handle.save
    def traced_save(data):
        order.append("save")
        return real_save(data)
    handle.save = traced_save
    job.execute()
    assert order.index("save") < order.index("cleanup")


def test_usage_property_returns_zero_dict_when_not_completed(tmp_path, fake_providers):
    job = Job(_handle(tmp_path), "x", messages=[{"role": "user", "content": "hi"}],
              model="gpt-4o", mode="batch")
    assert job.usage == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def test_usage_property_returns_stored_usage_when_completed(tmp_path, fake_providers, monkeypatch):
    monkeypatch.setattr(job_module.providers, "poll_batch", lambda h: PollResult(
        status="completed", texts=["the text"],
        usage={"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        cost={"charged_usd": 0.01, "interactive_usd": 0.02, "saved_usd": 0.01},
        output_file_id="files/out-1"))
    job = Job(_handle(tmp_path), "x", messages=[{"role": "user", "content": "hi"}],
              model="gpt-4o", mode="batch")
    job.execute()                       # submit -> SENT
    job.execute()                       # poll -> COMPLETED
    assert job.usage == {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}


def test_usage_property_returns_zero_dict_when_no_usage_key_in_state(tmp_path, fake_providers):
    # Build a job and directly manipulate its state to have no 'usage' key
    handle = _handle(tmp_path)
    job = Job(handle, "x", messages=[{"role": "user", "content": "hi"}],
              model="gpt-4o", mode="batch")
    job._state = {"state": "completed", "response": "test"}
    assert job.usage == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def test_finish_reason_property_none_before_completion(tmp_path, fake_providers):
    job = Job(_handle(tmp_path), "x", messages=[{"role": "user", "content": "hi"}],
              model="gpt-4o", mode="batch")
    assert job.finish_reason is None


def test_finish_reason_property_returns_stored_value_after_batch_completion(
    tmp_path, fake_providers, monkeypatch
):
    monkeypatch.setattr(job_module.providers, "poll_batch", lambda h: PollResult(
        status="completed", texts=["the text"],
        usage={"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        cost={"charged_usd": 0.01, "interactive_usd": 0.02, "saved_usd": 0.01},
        output_file_id="files/out-1", finish_reason="length"))
    job = Job(_handle(tmp_path), "x", messages=[{"role": "user", "content": "hi"}],
              model="gpt-4o", mode="batch")
    job.execute()                       # submit -> SENT
    job.execute()                       # poll -> COMPLETED
    assert job.finish_reason == "length"


def test_finish_reason_property_none_when_no_key_in_state(tmp_path, fake_providers):
    handle = _handle(tmp_path)
    job = Job(handle, "x", messages=[{"role": "user", "content": "hi"}],
              model="gpt-4o", mode="batch")
    job._state = {"state": "completed", "response": "test"}
    assert job.finish_reason is None


def test_finish_reason_set_from_interactive_completion(tmp_path, fake_providers, monkeypatch):
    def complete(model, messages, **kw):
        return CompletionResult(text="interactive answer",
                                 usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                                 cost={"charged_usd": 0.02, "interactive_usd": 0.02, "saved_usd": 0.0},
                                 finish_reason="stop")
    monkeypatch.setattr(job_module.providers, "complete", complete)
    job = Job(_handle(tmp_path), "x", messages=[{"role": "user", "content": "hi"}],
              model="gpt-4o", mode="interactive")
    job.execute()
    assert job.finish_reason == "stop"


def test_finish_reason_none_for_bundle_response(tmp_path, fake_providers, monkeypatch):
    def complete(model, messages, **kw):
        return CompletionResult(text="interactive answer",
                                 usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                                 cost={"charged_usd": 0.02, "interactive_usd": 0.02, "saved_usd": 0.0},
                                 finish_reason="stop")
    monkeypatch.setattr(job_module.providers, "complete", complete)
    job = Job(_handle(tmp_path), "x",
              requests=[[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]],
              model="gpt-4o", mode="interactive")
    job.execute()
    assert job.finish_reason is None  # bundle: scalar not populated


def test_finish_reason_none_for_one_request_bundle_via_batch_completion(
    tmp_path, fake_providers, monkeypatch
):
    # A bundle of exactly one request (requests=[[msg]]) completing via the batch
    # poll path. providers.poll_batch's count-based rule (n_requests == 1) would
    # populate a scalar finish_reason here, mimicking the provider layer -- but
    # this job is still a bundle (_is_bundle is True), so finish_reason must stay
    # None for consistency with the interactive completion paths.
    monkeypatch.setattr(job_module.providers, "poll_batch", lambda h: PollResult(
        status="completed", texts=["the text"],
        usage={"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        cost={"charged_usd": 0.01, "interactive_usd": 0.02, "saved_usd": 0.01},
        output_file_id="files/out-1", finish_reason="length"))
    job = Job(_handle(tmp_path), "x",
              requests=[[{"role": "user", "content": "a"}]],
              model="gpt-4o", mode="batch")
    job.execute()                       # submit -> SENT
    job.execute()                       # poll -> COMPLETED
    assert job.finish_reason is None


def test_max_tokens_and_thinking_kwargs_reach_batch_submit(tmp_path, fake_providers):
    job = Job(_handle(tmp_path), "x", messages=[{"role": "user", "content": "hi"}],
              model="gpt-4o", mode="batch",
              max_tokens=4000, thinking={"type": "disabled"})
    job.execute()
    assert fake_providers["submit_kwargs"]["max_tokens"] == 4000
    assert fake_providers["submit_kwargs"]["thinking"] == {"type": "disabled"}


def test_max_tokens_and_thinking_kwargs_reach_interactive_complete(tmp_path, fake_providers):
    job = Job(_handle(tmp_path), "x", messages=[{"role": "user", "content": "hi"}],
              model="gpt-4o", mode="interactive",
              max_tokens=4000, thinking={"type": "disabled"})
    job.execute()
    assert fake_providers["complete_kwargs"]["max_tokens"] == 4000
    assert fake_providers["complete_kwargs"]["thinking"] == {"type": "disabled"}


def test_pipeline_job_forwards_max_tokens_and_thinking(tmp_path, fake_providers):
    from llm_batch import Pipeline

    pipeline = Pipeline(str(tmp_path), mode="batch")
    job = pipeline.job("x", messages=[{"role": "user", "content": "hi"}], model="gpt-4o",
                       max_tokens=4000, thinking={"type": "disabled"})
    job.execute()
    assert fake_providers["submit_kwargs"]["max_tokens"] == 4000
    assert fake_providers["submit_kwargs"]["thinking"] == {"type": "disabled"}
