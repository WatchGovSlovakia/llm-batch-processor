import json
import os

import pytest

from llm_batch.storage.base import Storage, StorageScanner
from llm_batch.storage.file import FileStorage


def test_for_job_round_trip(tmp_path):
    backend = FileStorage(str(tmp_path))
    handle = backend.for_job("ocr")
    assert handle.load() is False                 # nothing persisted yet
    handle.save({"key": "ocr", "state": "sent", "n": 1})
    assert os.path.exists(tmp_path / "ocr.json")

    fresh = FileStorage(str(tmp_path)).for_job("ocr")
    assert fresh.load() is True
    assert fresh.get("state") == "sent"
    assert fresh.snapshot() == {"key": "ocr", "state": "sent", "n": 1}


def test_get_default(tmp_path):
    handle = FileStorage(str(tmp_path)).for_job("missing")
    handle.load()
    assert handle.get("state", "not_sent") == "not_sent"


def test_deterministic_serialization(tmp_path):
    handle = FileStorage(str(tmp_path)).for_job("j")
    handle.save({"b": 2, "a": 1, "nested": {"z": 1, "y": 2}})
    raw = (tmp_path / "j.json").read_text()
    # sorted keys, 2-space indent, trailing newline
    assert raw.endswith("}\n")
    assert raw.index('"a"') < raw.index('"b"')
    assert raw.index('"y"') < raw.index('"z"')


def test_atomic_write_leaves_no_temp_files(tmp_path):
    handle = FileStorage(str(tmp_path)).for_job("j")
    handle.save({"x": 1})
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "j.json"]
    assert leftovers == []


def test_finish_removes_file(tmp_path):
    handle = FileStorage(str(tmp_path)).for_job("j")
    handle.save({"x": 1})
    handle.finish()
    assert not (tmp_path / "j.json").exists()


def test_for_meta_uses_pipeline_json(tmp_path):
    meta = FileStorage(str(tmp_path)).for_meta()
    meta.save({"pipeline_started_at": "2026-06-08T00:00:00+00:00"})
    assert (tmp_path / "pipeline.json").exists()


def test_iter_jobs_skips_meta_and_walks_recursively(tmp_path):
    # two pipelines under one top-level state dir
    FileStorage(str(tmp_path / "doc_a")).for_job("ocr").save({"key": "ocr", "state": "sent"})
    FileStorage(str(tmp_path / "doc_a")).for_meta().save({"pipeline_started_at": "x"})
    FileStorage(str(tmp_path / "doc_b")).for_job("ocr").save({"key": "ocr", "state": "completed"})

    handles = list(FileStorage(str(tmp_path)).iter_jobs())
    states = sorted(h.get("state") for h in handles)
    assert states == ["completed", "sent"]          # pipeline.json excluded


def test_iter_jobs_yields_storage_handles(tmp_path):
    FileStorage(str(tmp_path)).for_job("j").save({"key": "j", "state": "sent"})
    handles = list(FileStorage(str(tmp_path)).iter_jobs())
    assert len(handles) == 1
    assert isinstance(handles[0], Storage)          # runtime_checkable protocol
    # the yielded handle can re-save in place
    handles[0].save({"key": "j", "state": "completed"})
    assert FileStorage(str(tmp_path)).for_job("j").load() and \
        FileStorage(str(tmp_path)).for_job("j").get("state") == "completed"


def test_filestorage_is_scanner(tmp_path):
    assert isinstance(FileStorage(str(tmp_path)), StorageScanner)
