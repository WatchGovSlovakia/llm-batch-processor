import logging

import pytest

from llm_batch import job as job_module
from llm_batch.job import Job, _now
from llm_batch.providers import BatchHandle, CompletionResult, PollResult
from llm_batch.storage.file import FileStorage
from datetime import timedelta, datetime


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


def test_can_afford_to_wait_decision(tmp_path, monkeypatch):
    monkeypatch.setattr(job_module.providers, "provider_of", lambda m: "openai")
    # window = poll_interval(1.0) + cancel_margin(openai default 0.25)
    #          + interactive_estimate(0.25) + safety(0.1) = 1.6h
    job = _spec_job(tmp_path, deadline=(_now() + timedelta(hours=3)).isoformat())
    assert job._can_afford_to_wait() is True

    tight = _spec_job(tmp_path, key="y",
                      deadline=(_now() + timedelta(hours=1)).isoformat())
    assert tight._can_afford_to_wait() is False


def test_can_afford_to_wait_no_deadline_is_true(tmp_path, monkeypatch):
    monkeypatch.setattr(job_module.providers, "provider_of", lambda m: "openai")
    job = _spec_job(tmp_path, deadline=None)
    assert job._can_afford_to_wait() is True


def test_cancel_margin_default_is_provider_aware(tmp_path, monkeypatch):
    monkeypatch.setattr(job_module.providers, "provider_of", lambda m: "vertex_ai")
    job = _spec_job(tmp_path, model="vertex_ai/gemini-1.5-pro")
    # vertex default margin 0.5 → window = 1.0 + 0.5 + 0.25 + 0.1 = 1.85h
    assert job._speculative_window_hours() == pytest.approx(1.85)


def _patch_providers(monkeypatch, *, provider="openai", supports=True):
    monkeypatch.setattr(job_module.providers, "provider_of", lambda m: provider)
    monkeypatch.setattr(job_module.providers, "provider_supports_cancel", lambda p: supports)
    calls = {"submit": 0, "complete": 0}

    def submit_batch(model, requests, **kw):
        calls["submit"] += 1
        return BatchHandle(provider=provider, model=model, batch_job_id="b1", input_file_id="f1")

    def complete(model, messages, **kw):
        calls["complete"] += 1
        return CompletionResult(text="now", usage={}, cost={"charged_usd": 0.0, "interactive_usd": 0.0, "saved_usd": 0.0})

    monkeypatch.setattr(job_module.providers, "submit_batch", submit_batch)
    monkeypatch.setattr(job_module.providers, "complete", complete)
    return calls


def test_speculative_ample_time_submits_batch(tmp_path, monkeypatch):
    calls = _patch_providers(monkeypatch)
    job = _spec_job(tmp_path, deadline=(_now() + timedelta(hours=72)).isoformat())
    assert job.execute() is None
    assert calls["submit"] == 1 and calls["complete"] == 0
    snap = job.snapshot()
    assert snap["state"] == Job.SENT
    assert snap["fallback_strategy"] == "speculative"
    assert snap["payload"] == [[{"role": "user", "content": "hi"}]]


def test_speculative_too_tight_goes_interactive_immediately(tmp_path, monkeypatch):
    calls = _patch_providers(monkeypatch)
    # 1h remaining < ~1.6h speculative window → no point submitting a batch.
    job = _spec_job(tmp_path, deadline=(_now() + timedelta(hours=1)).isoformat())
    assert job.execute() == "now"
    assert calls["submit"] == 0 and calls["complete"] == 1
    assert job.snapshot()["actual_mode"] == "interactive"


def test_speculative_anthropic_degrades_to_conservative_with_one_warning(tmp_path, monkeypatch, caplog):
    calls = _patch_providers(monkeypatch, provider="anthropic", supports=False)
    deadline = (_now() + timedelta(hours=72)).isoformat()  # ample for conservative too
    with caplog.at_level(logging.WARNING, logger="llm_batch"):
        j1 = _spec_job(tmp_path, key="a", model="claude-3-5-sonnet", deadline=deadline)
        assert j1.execute() is None
        j2 = _spec_job(tmp_path, key="b", model="claude-3-5-sonnet", deadline=deadline)
        assert j2.execute() is None
    # Conservative path under ample time → batch submitted, stored as conservative.
    assert calls["submit"] == 2
    assert j1.snapshot()["fallback_strategy"] == "conservative"
    warnings = [r for r in caplog.records if "speculative" in r.getMessage().lower()]
    assert len(warnings) == 1  # one-time, not per job/poll


def _seed_speculative_sent(tmp_path, key="x", *, deadline, provider="openai"):
    """Write a SENT speculative job directly (as it would be after submit)."""
    handle = FileStorage(str(tmp_path)).for_job(key)
    handle.save({
        "key": key, "state": "sent", "mode": "batch", "model": "gpt-4o",
        "n_requests": 1, "bundle": False, "batch_job_id": "b1",
        "input_file_id": "f-in", "output_file_id": None,
        "submitted_at": _now().isoformat(), "completed_at": None,
        "deadline": deadline, "actual_mode": "batch", "response": None,
        "usage": {}, "cost": {}, "error_message": None, "poll_attempts": 0,
        "rerun_cmd": None, "cleanup_provider_files": False,
        "fallback_strategy": "speculative",
        "payload": [[{"role": "user", "content": "hi"}]],
        "litellm_kwargs": {}, "cancel_requested_at": None,
    })
    return handle


def test_state_a_keeps_waiting_when_time_remains(tmp_path, monkeypatch):
    monkeypatch.setattr(job_module.providers, "provider_of", lambda m: "openai")
    monkeypatch.setattr(job_module.providers, "poll_batch",
                        lambda h: PollResult(status="pending"))
    cancel_called = {"n": 0}
    monkeypatch.setattr(job_module.providers, "cancel",
                        lambda h: cancel_called.__setitem__("n", cancel_called["n"] + 1))
    handle = _seed_speculative_sent(tmp_path, deadline=(_now() + timedelta(hours=72)).isoformat())
    job = Job(handle)
    assert job.execute() is None
    assert job.state == Job.SENT
    assert cancel_called["n"] == 0


def test_state_a_initiates_cancel_when_out_of_time(tmp_path, monkeypatch):
    monkeypatch.setattr(job_module.providers, "provider_of", lambda m: "openai")
    monkeypatch.setattr(job_module.providers, "poll_batch",
                        lambda h: PollResult(status="pending"))
    cancel_called = {"n": 0}
    monkeypatch.setattr(job_module.providers, "cancel",
                        lambda h: cancel_called.__setitem__("n", cancel_called["n"] + 1))
    handle = _seed_speculative_sent(tmp_path, deadline=(_now() + timedelta(hours=1)).isoformat())
    job = Job(handle)
    assert job.execute() is None
    assert job.state == Job.CANCEL_REQUESTED
    assert cancel_called["n"] == 1
    assert job.snapshot()["cancel_requested_at"] is not None


def test_state_a_completed_returns_result(tmp_path, monkeypatch):
    monkeypatch.setattr(job_module.providers, "provider_of", lambda m: "openai")
    monkeypatch.setattr(job_module.providers, "cleanup", lambda h: None)
    monkeypatch.setattr(
        job_module.providers, "poll_batch",
        lambda h: PollResult(status="completed", texts=["done"],
                             usage={}, cost={}, output_file_id="of1"))
    handle = _seed_speculative_sent(tmp_path, deadline=(_now() + timedelta(hours=1)).isoformat())
    job = Job(handle)
    assert job.execute() == "done"
    assert job.state == Job.COMPLETED


def test_state_a_cancel_transient_stays_sent(tmp_path, monkeypatch):
    monkeypatch.setattr(job_module.providers, "provider_of", lambda m: "openai")
    monkeypatch.setattr(job_module.providers, "poll_batch",
                        lambda h: PollResult(status="pending"))

    def boom(h):
        raise job_module.providers.TransientProviderError("rate limited")

    monkeypatch.setattr(job_module.providers, "cancel", boom)
    handle = _seed_speculative_sent(tmp_path, deadline=(_now() + timedelta(hours=1)).isoformat())
    job = Job(handle)
    assert job.execute() is None
    assert job.state == Job.SENT  # cancel failed transiently; retry next poll


def _seed_cancel_requested(tmp_path, key="x", *, cancel_requested_at, provider="openai",
                           model="gpt-4o"):
    handle = FileStorage(str(tmp_path)).for_job(key)
    handle.save({
        "key": key, "state": "cancel_requested", "mode": "batch", "model": model,
        "n_requests": 1, "bundle": False, "batch_job_id": "b1",
        "input_file_id": "f-in", "output_file_id": None,
        "submitted_at": _now().isoformat(), "completed_at": None, "deadline": None,
        "actual_mode": "batch", "response": None, "usage": {}, "cost": {},
        "error_message": None, "poll_attempts": 1, "rerun_cmd": None,
        "cleanup_provider_files": False, "fallback_strategy": "speculative",
        "payload": [[{"role": "user", "content": "hi"}]],
        "litellm_kwargs": {}, "cancel_requested_at": cancel_requested_at,
    })
    return handle


def test_state_b_completed_salvages_batch_result(tmp_path, monkeypatch):
    monkeypatch.setattr(job_module.providers, "provider_of", lambda m: "openai")
    monkeypatch.setattr(job_module.providers, "cleanup", lambda h: None)
    monkeypatch.setattr(
        job_module.providers, "poll_batch",
        lambda h: PollResult(status="completed", texts=["batched"],
                             usage={}, cost={}, output_file_id="of1"))
    handle = _seed_cancel_requested(tmp_path, cancel_requested_at=_now().isoformat())
    job = Job(handle)
    assert job.execute() == "batched"
    assert job.state == Job.COMPLETED
    assert job.snapshot()["actual_mode"] == "batch"


def test_state_b_cancelled_reissues_interactively(tmp_path, monkeypatch):
    monkeypatch.setattr(job_module.providers, "provider_of", lambda m: "openai")
    monkeypatch.setattr(job_module.providers, "cleanup", lambda h: None)
    monkeypatch.setattr(job_module.providers, "poll_batch",
                        lambda h: PollResult(status="canceled"))
    monkeypatch.setattr(
        job_module.providers, "complete",
        lambda model, messages, **kw: CompletionResult(
            text="interactive-now", usage={}, cost={"charged_usd": 0.02, "interactive_usd": 0.02, "saved_usd": 0.0}))
    handle = _seed_cancel_requested(tmp_path, cancel_requested_at=_now().isoformat())
    job = Job(handle)
    assert job.execute() == "interactive-now"
    assert job.state == Job.COMPLETED
    assert job.snapshot()["actual_mode"] == "interactive"


def test_state_b_still_queued_keeps_waiting(tmp_path, monkeypatch):
    monkeypatch.setattr(job_module.providers, "provider_of", lambda m: "openai")
    monkeypatch.setattr(job_module.providers, "poll_batch",
                        lambda h: PollResult(status="pending"))
    handle = _seed_cancel_requested(tmp_path, cancel_requested_at=_now().isoformat())
    job = Job(handle)
    assert job.execute() is None
    assert job.state == Job.CANCEL_REQUESTED


def test_state_b_timeout_forces_interactive(tmp_path, monkeypatch):
    monkeypatch.setattr(job_module.providers, "provider_of", lambda m: "openai")
    monkeypatch.setattr(job_module.providers, "cleanup", lambda h: None)
    monkeypatch.setattr(job_module.providers, "poll_batch",
                        lambda h: PollResult(status="pending"))
    monkeypatch.setattr(
        job_module.providers, "complete",
        lambda model, messages, **kw: CompletionResult(
            text="forced", usage={}, cost={"charged_usd": 0.0, "interactive_usd": 0.0, "saved_usd": 0.0}))
    # openai resolution timeout default 0.5h; stamp the request 2h ago → expired.
    stamped = (_now() - timedelta(hours=2)).isoformat()
    handle = _seed_cancel_requested(tmp_path, cancel_requested_at=stamped)
    job = Job(handle)
    assert job.execute() == "forced"
    assert job.state == Job.COMPLETED
    assert job.snapshot()["actual_mode"] == "interactive"


def test_state_b_reissue_transient_stays_cancel_requested(tmp_path, monkeypatch):
    monkeypatch.setattr(job_module.providers, "provider_of", lambda m: "openai")
    monkeypatch.setattr(job_module.providers, "poll_batch",
                        lambda h: PollResult(status="canceled"))

    def boom(model, messages, **kw):
        raise job_module.providers.TransientProviderError("rate limited")

    monkeypatch.setattr(job_module.providers, "complete", boom)
    handle = _seed_cancel_requested(tmp_path, cancel_requested_at=_now().isoformat())
    job = Job(handle)
    assert job.execute() is None
    assert job.state == Job.CANCEL_REQUESTED  # retry interactive next poll


def test_full_speculative_lifecycle_cancel_then_reissue(tmp_path, monkeypatch):
    monkeypatch.setattr(job_module.providers, "provider_of", lambda m: "openai")
    monkeypatch.setattr(job_module.providers, "provider_supports_cancel", lambda p: True)
    monkeypatch.setattr(job_module.providers, "cleanup", lambda h: None)
    monkeypatch.setattr(
        job_module.providers, "submit_batch",
        lambda model, requests, **kw: BatchHandle(
            provider="openai", model=model, batch_job_id="b1", input_file_id="f1"))

    # Provider state evolves across polls: pending, then cancelled after we ask.
    state = {"phase": "pending"}
    monkeypatch.setattr(
        job_module.providers, "poll_batch",
        lambda h: PollResult(status=state["phase"]))
    monkeypatch.setattr(job_module.providers, "cancel",
                        lambda h: state.__setitem__("phase", "canceled"))
    monkeypatch.setattr(
        job_module.providers, "complete",
        lambda model, messages, **kw: CompletionResult(
            text="final", usage={}, cost={"charged_usd": 0.02, "interactive_usd": 0.04, "saved_usd": 0.0}))

    # Poll run 1 — submit with ample time (deadline far out).
    j = _spec_job(tmp_path, deadline=(_now() + timedelta(hours=72)).isoformat())
    assert j.execute() is None
    assert j.state == Job.SENT

    # Poll run 2 — reconstruct from storage with a now-tight deadline; out of time
    # → initiate cancel, transition to CANCEL_REQUESTED. (Rewrite the deadline to
    # simulate time passing close to the deadline.)
    storage = FileStorage(str(tmp_path)).for_job("x")
    snap = storage.snapshot()
    snap["deadline"] = (_now() + timedelta(hours=1)).isoformat()
    storage.save(snap)
    j2 = Job(storage)
    assert j2.execute() is None
    assert j2.state == Job.CANCEL_REQUESTED

    # Poll run 3 — provider now reports cancelled → re-issue interactively.
    j3 = Job(FileStorage(str(tmp_path)).for_job("x"))
    assert j3.execute() == "final"
    assert j3.state == Job.COMPLETED
    assert j3.snapshot()["actual_mode"] == "interactive"
