import pytest

from llm_batch import job as job_module
from llm_batch.pipeline import Pipeline
from llm_batch.job import Job, _now, _parse_iso
from llm_batch.providers import BatchHandle, CompletionResult
from llm_batch.storage.file import FileStorage
from datetime import timedelta


@pytest.fixture
def fake_providers(monkeypatch):
    def complete(model, messages, **kw):
        return CompletionResult(text="ans", usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                                cost={"charged_usd": 0.05, "interactive_usd": 0.05, "saved_usd": 0.0})
    monkeypatch.setattr(job_module.providers, "complete", complete)
    monkeypatch.setattr(job_module.providers, "provider_of", lambda m: "openai")


def test_pipeline_records_started_at_once(tmp_path):
    p1 = Pipeline(state_dir=str(tmp_path), mode="batch")
    started = p1.meta()["pipeline_started_at"]
    # a second Pipeline over the same dir reuses the recorded anchor
    p2 = Pipeline(state_dir=str(tmp_path), mode="batch")
    assert p2.meta()["pipeline_started_at"] == started


def test_time_budget_hours_becomes_deadline(tmp_path):
    p = Pipeline(state_dir=str(tmp_path), mode="batch", time_budget_hours=48)
    deadline = _parse_iso(p.meta()["deadline"])
    started = _parse_iso(p.meta()["pipeline_started_at"])
    assert abs((deadline - started).total_seconds() - 48 * 3600) < 5


def test_explicit_deadline_takes_precedence(tmp_path):
    deadline = (_now() + timedelta(hours=100)).isoformat()
    p = Pipeline(state_dir=str(tmp_path), mode="batch", deadline=deadline)
    assert p.meta()["deadline"] == deadline


def test_job_inherits_pipeline_config(tmp_path, fake_providers):
    p = Pipeline(state_dir=str(tmp_path), mode="interactive", rerun_cmd="python go.py doc1")
    job = p.job("step1", messages=[{"role": "user", "content": "hi"}], model="gpt-4o")
    assert isinstance(job, Job)
    job.execute()
    snap = job.snapshot()
    assert snap["mode"] == "interactive"
    assert snap["rerun_cmd"] == "python go.py doc1"


def test_job_state_file_namespaced_under_pipeline_dir(tmp_path, fake_providers):
    p = Pipeline(state_dir=str(tmp_path), mode="interactive")
    p.job("ocr", messages=[{"role": "user", "content": "hi"}], model="gpt-4o").execute()
    assert (tmp_path / "ocr.json").exists()


def test_cost_summary_aggregates_across_jobs(tmp_path, fake_providers):
    p = Pipeline(state_dir=str(tmp_path), mode="interactive")
    p.job("a", messages=[{"role": "user", "content": "x"}], model="gpt-4o").execute()
    p.job("b", messages=[{"role": "user", "content": "y"}], model="gpt-4o").execute()
    summary = p.cost_summary()
    assert summary["charged_usd"] == pytest.approx(0.10)
    assert summary["interactive_usd"] == pytest.approx(0.10)


def test_usage_summary_aggregates_across_jobs(tmp_path, fake_providers):
    p = Pipeline(state_dir=str(tmp_path), mode="interactive")
    p.job("a", messages=[{"role": "user", "content": "x"}], model="gpt-4o").execute()
    p.job("b", messages=[{"role": "user", "content": "y"}], model="gpt-4o").execute()
    summary = p.usage_summary()
    assert summary["prompt_tokens"] == 2
    assert summary["completion_tokens"] == 2
    assert summary["total_tokens"] == 4
    assert summary["requests"] == 2


def test_cost_summary_filters_by_job_keys(tmp_path, fake_providers):
    p = Pipeline(state_dir=str(tmp_path), mode="interactive")
    p.job("a", messages=[{"role": "user", "content": "x"}], model="gpt-4o").execute()
    p.job("b", messages=[{"role": "user", "content": "y"}], model="gpt-4o").execute()
    summary = p.cost_summary(job_keys=["b"])
    assert summary["charged_usd"] == pytest.approx(0.05)
    assert summary["interactive_usd"] == pytest.approx(0.05)


def test_usage_summary_filters_by_job_keys(tmp_path, fake_providers):
    p = Pipeline(state_dir=str(tmp_path), mode="interactive")
    p.job("a", messages=[{"role": "user", "content": "x"}], model="gpt-4o").execute()
    p.job("b", messages=[{"role": "user", "content": "y"}], model="gpt-4o").execute()
    summary = p.usage_summary(job_keys=["b"])
    assert summary["prompt_tokens"] == 1
    assert summary["completion_tokens"] == 1
    assert summary["total_tokens"] == 2
    assert summary["requests"] == 1


def test_job_keys_empty_before_any_job(tmp_path):
    p = Pipeline(state_dir=str(tmp_path), mode="interactive")
    assert p.job_keys() == []


def test_job_keys_returns_submitted_and_completed_keys(tmp_path, fake_providers):
    p = Pipeline(state_dir=str(tmp_path), mode="interactive")
    p.job("a", messages=[{"role": "user", "content": "x"}], model="gpt-4o").execute()
    p.job("b", messages=[{"role": "user", "content": "y"}], model="gpt-4o").execute()
    assert sorted(p.job_keys()) == ["a", "b"]


def test_job_keys_excludes_pipeline_meta_file(tmp_path, fake_providers):
    p = Pipeline(state_dir=str(tmp_path), mode="interactive")
    p.job("a", messages=[{"role": "user", "content": "x"}], model="gpt-4o").execute()
    assert p.job_keys() == ["a"]
    assert "pipeline" not in p.job_keys()


def test_save_meta_merges_extra_fields(tmp_path):
    p = Pipeline(state_dir=str(tmp_path), mode="batch")
    p.save_meta({"pdf_url": "http://x/doc.pdf", "deadline": "2026-06-15"})
    reloaded = Pipeline(state_dir=str(tmp_path), mode="batch")
    assert reloaded.meta()["pdf_url"] == "http://x/doc.pdf"


def test_finish_removes_local_state(tmp_path, fake_providers):
    p = Pipeline(state_dir=str(tmp_path), mode="interactive")
    p.job("a", messages=[{"role": "user", "content": "x"}], model="gpt-4o").execute()
    p.finish()
    assert not (tmp_path / "a.json").exists()
    assert not (tmp_path / "pipeline.json").exists()


def test_save_meta_deadline_propagates_to_jobs(tmp_path, fake_providers):
    # save_meta sets a tight deadline AFTER init; jobs created afterward must
    # see it and fall back to interactive (not submit a doomed batch).
    p = Pipeline(state_dir=str(tmp_path), mode="batch")
    p.save_meta({"deadline": (_now() + timedelta(hours=2)).isoformat()})
    job = p.job("x", messages=[{"role": "user", "content": "hi"}], model="gpt-4o")
    job.execute()
    assert job.snapshot()["actual_mode"] == "interactive"


def test_pipeline_threads_speculative_config_to_jobs(tmp_path):
    from llm_batch.pipeline import Pipeline
    p = Pipeline(str(tmp_path), fallback_strategy="speculative",
                 poll_interval_hours=2.0, interactive_runtime_estimate_hours=0.5,
                 cancel_margin_hours=0.4, cancel_resolution_timeout_hours=3.0)
    job = p.job("k", messages=[{"role": "user", "content": "hi"}], model="gpt-4o")
    assert job._fallback_strategy == "speculative"
    assert job._poll_interval_h == 2.0
    assert job._interactive_estimate_h == 0.5
    assert job._cancel_margin_h == 0.4
    assert job._cancel_resolution_timeout_h == 3.0


def test_pipeline_defaults_to_conservative(tmp_path):
    from llm_batch.pipeline import Pipeline
    p = Pipeline(str(tmp_path))
    job = p.job("k", messages=[{"role": "user", "content": "hi"}], model="gpt-4o")
    assert job._fallback_strategy == "conservative"
