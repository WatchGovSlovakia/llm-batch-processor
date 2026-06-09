"""Full stack: Pipeline -> Job -> real providers layer, with litellm mocked.

Exercises a two-step batch pipeline that submits, resumes across fresh objects
(as cron would), polls to completion, aggregates cost, and cleans up.
"""

import pytest

from llm_batch import Pipeline, Poller, FileStorage
from tests.conftest import batch_output_jsonl, completion_response, make_obj


@pytest.fixture
def wired_litellm(litellm_mock):
    """Configure the mock litellm so the whole batch flow works end-to-end."""
    litellm_mock.create_file = lambda **k: make_obj(id="files/in-1")
    litellm_mock.create_batch = lambda **k: make_obj(id="batches/job-1")
    litellm_mock.retrieve_batch = lambda **k: make_obj(
        status="completed", output_file_id="files/out-1")
    litellm_mock.cost_per_token = lambda **k: (0.01, 0.02)
    deleted = []
    litellm_mock.file_delete = lambda file_id, custom_llm_provider: deleted.append(file_id)
    litellm_mock._deleted = deleted
    return litellm_mock


def test_two_step_batch_pipeline_end_to_end(tmp_path, wired_litellm):
    state_dir = str(tmp_path / "doc1")

    # ---- Run 1: submit step 1 (a 2-request OCR bundle) ----
    p = Pipeline(state_dir=state_dir, mode="batch", rerun_cmd="python go.py doc1")
    ocr = p.job("ocr", requests=[[{"role": "user", "content": "p1"}],
                                 [{"role": "user", "content": "p2"}]],
                model="gpt-4o")
    assert ocr.execute() is None                      # submitted as batch -> SENT

    # ---- Run 2 (fresh objects, as a later cron tick): poll -> complete ----
    wired_litellm.file_content = lambda **k: batch_output_jsonl(["text one", "text two"])
    p2 = Pipeline(state_dir=state_dir, mode="batch", rerun_cmd="python go.py doc1")
    ocr2 = p2.job("ocr", requests=[[{"role": "user", "content": "p1"}],
                                   [{"role": "user", "content": "p2"}]],
                  model="gpt-4o")
    pages = ocr2.execute()
    assert pages == ["text one", "text two"]
    assert "files/in-1" in wired_litellm._deleted     # cleanup ran after persist

    # ---- step 2: analysis, submit then complete ----
    analysis = p2.job("analysis",
                      messages=[{"role": "user", "content": "\n".join(pages)}],
                      model="gpt-4o")
    assert analysis.execute() is None                 # SENT
    wired_litellm.file_content = lambda **k: batch_output_jsonl(["final analysis"])
    p3 = Pipeline(state_dir=state_dir, mode="batch", rerun_cmd="python go.py doc1")
    analysis3 = p3.job("analysis",
                       messages=[{"role": "user", "content": "ignored on resume"}],
                       model="gpt-4o")
    assert analysis3.execute() == "final analysis"

    # ---- cost aggregation: both steps captured the batch discount ----
    summary = p3.cost_summary()
    assert summary["charged_usd"] > 0
    assert summary["saved_usd"] == pytest.approx(summary["charged_usd"])  # 50% rate

    # ---- finish clears local state ----
    p3.finish()
    assert not (tmp_path / "doc1" / "ocr.json").exists()
    assert not (tmp_path / "doc1" / "analysis.json").exists()


def test_poller_advances_pipeline_end_to_end(tmp_path, wired_litellm):
    state_dir = str(tmp_path)
    # submit one job under a pipeline subdir
    p = Pipeline(state_dir=str(tmp_path / "doc1"), mode="batch", rerun_cmd="python go.py doc1")
    p.job("ocr", messages=[{"role": "user", "content": "p1"}], model="gpt-4o").execute()

    wired_litellm.file_content = lambda **k: batch_output_jsonl(["done"])
    ran = []
    Poller(FileStorage(state_dir), run_cmd=lambda argv: ran.append(argv)).poll_all()
    assert ran == [["python", "go.py", "doc1"]]


def test_deadline_fallback_end_to_end(tmp_path, wired_litellm):
    from datetime import timedelta
    from llm_batch.job import _now
    wired_litellm.completion = lambda **k: completion_response("immediate")

    # deadline 2h away, default 24h margin -> interactive fallback, no batch
    p = Pipeline(state_dir=str(tmp_path), mode="batch",
                 deadline=(_now() + timedelta(hours=2)).isoformat())
    job = p.job("step", messages=[{"role": "user", "content": "hi"}], model="gpt-4o")
    assert job.execute() == "immediate"
    assert job.snapshot()["actual_mode"] == "interactive"
