# Transparent Interactive/Batch LLM Library — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `llm_batch`, a Python library that makes LLM calls transparently switchable between interactive and batch modes behind one idempotent `job.execute()` call site, with persistent resumable state and a deadline-aware batch→interactive fallback.

**Architecture:** A `Job` is the unit of resumable work; `execute()` is a state machine (`not_sent → sent → completed | failed`) driven entirely by persisted state, so it is safe to re-run from cron. A `Pipeline` shares one state dir + config across steps. A `providers.py` layer isolates all litellm/provider plumbing behind a tiny interface. Pluggable `Storage` backends (`FileStorage`, `PostgresStorage`) persist per-job state; a `StorageScanner` enumerates jobs for the `Poller`, which advances completed pipelines via their `rerun_cmd`. The library is filesystem-only — the caller owns git.

**Tech Stack:** Python 3.12, `litellm` (provider abstraction), `pytest` (TDD with litellm fully mocked — no network in unit tests), `psycopg`/`psycopg2` (optional, Postgres backend only).

---

## Design Notes (refinements over the spec, locked in here)

These are small, deliberate refinements made while planning. They do not change spec behavior:

1. **`StorageScanner.iter_jobs()` yields loaded per-job `Storage` handles**, not bare dicts. Each handle exposes its dict via `.snapshot()` / `.get()`. The spec wrote `Iterator[dict]`; the poller must *update* state in place, which a bare dict cannot do, so it yields handles instead. Read-only callers (`cost_summary`) just call `.get()`.
2. **A `"bundle"` boolean is added to the persisted state schema** (alongside `n_requests` from §5.4). It records whether the job was created with `requests=` (bundle → result is `list[str]`) vs `messages=` (single → result is `str`), so a resumed/polled job returns the correct shape.
3. **Backend objects expose `for_job(key)`, `for_meta()`, and `iter_jobs()`.** `FileStorage(dir)` and `PostgresStorage(conn, namespace)` are *directory/namespace-level* objects that mint per-job `Storage` handles. This is how the spec's "per-job handle vs cross-job scanner" split (§5.1) is realized concretely.
4. **Time helpers `_now()`, `_now_iso()`, `_parse_iso()` live in `job.py`** and are imported by `pipeline.py` and `poller.py` (both already depend on `job.py`). No new module beyond the spec's §12 list.

---

## File Structure

```
llm_batch/
  __init__.py        # public exports: Job, Pipeline, Poller, FileStorage, PostgresStorage, BatchJobFailed
  errors.py          # BatchJobFailed, TransientProviderError
  providers.py       # litellm batch/interactive, model→provider, pricing, cleanup; BatchHandle/PollResult/CompletionResult
  job.py             # Job + execute() state machine + deadline fallback + time helpers
  pipeline.py        # Pipeline + pipeline.json meta (pipeline_started_at, deadline, config)
  poller.py          # Poller: scan SENT jobs, advance via rerun_cmd, stale timeout, orphan sweep
  cli.py             # `llm-batch poll <state_dir> [--sweep-orphans N]`
  storage/
    __init__.py      # re-exports FileStorage, PostgresStorage
    base.py          # Storage + StorageScanner protocols
    file.py          # FileStorage (atomic, deterministic JSON) + per-job handle
    postgres.py      # PostgresStorage (JSONB upsert, namespaced)
tests/
  conftest.py        # litellm mock fixture + batch-output builder helper
  test_errors.py
  test_storage_file.py
  test_providers.py
  test_job.py
  test_job_fallback.py
  test_job_preset_retry.py
  test_job_cleanup.py
  test_pipeline.py
  test_poller.py
  test_cli.py
  test_storage_postgres.py
  test_end_to_end.py
pyproject.toml
README.md
```

Each file has one clear responsibility: `providers.py` is the only file that imports `litellm`; `job.py` holds the state machine; `storage/` holds persistence; `poller.py`/`cli.py` are the cron surface.

---

## Task 0: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `llm_batch/__init__.py` (placeholder, filled in Task 16)
- Create: `llm_batch/storage/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_smoke.py` (deleted at end of task)

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "llm-batch"
version = "0.1.0"
description = "Transparent, resumable interactive/batch LLM processing on top of litellm"
requires-python = ">=3.10"
dependencies = [
    "litellm>=1.40",
]

[project.optional-dependencies]
postgres = ["psycopg[binary]>=3.1"]
dev = ["pytest>=8.0"]

[project.scripts]
llm-batch = "llm_batch.cli:main"

[tool.setuptools.packages.find]
include = ["llm_batch*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create empty package files**

`llm_batch/__init__.py`:

```python
"""llm_batch — transparent interactive/batch LLM processing."""
```

`llm_batch/storage/__init__.py`:

```python
"""Storage backends for llm_batch."""
```

- [ ] **Step 3: Write the litellm mock fixture in `tests/conftest.py`**

```python
import io
import json
import types

import pytest


@pytest.fixture
def litellm_mock(monkeypatch):
    """A fake litellm module patched into llm_batch.providers.

    Tests configure return values by assigning to the returned object's
    attributes (each is a plain function or a list of canned objects).
    """
    fake = types.SimpleNamespace()

    # Default no-op stubs; individual tests override these.
    def _not_configured(*a, **k):  # pragma: no cover - guard
        raise AssertionError("litellm function called but not configured by the test")

    fake.create_file = _not_configured
    fake.create_batch = _not_configured
    fake.retrieve_batch = _not_configured
    fake.file_content = _not_configured
    fake.completion = _not_configured
    fake.file_delete = lambda *a, **k: None
    fake.cost_per_token = lambda **k: (0.0, 0.0)

    import llm_batch.providers as providers
    monkeypatch.setattr(providers, "litellm", fake)
    return fake


def make_obj(**attrs):
    """Build a simple attribute-bag object (stand-in for litellm response objects)."""
    return types.SimpleNamespace(**attrs)


def batch_output_jsonl(texts, usage=None):
    """Build the raw JSONL body that litellm.file_content returns for a completed batch.

    `texts` is a list aligned to custom_id req-0..req-N.
    """
    usage = usage or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    lines = []
    for i, text in enumerate(texts):
        lines.append(json.dumps({
            "custom_id": f"req-{i}",
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [{"message": {"content": text}}],
                    "usage": usage,
                },
            },
        }))
    body = "\n".join(lines) + "\n"
    return make_obj(text=body, content=body.encode())


def completion_response(text, usage=None):
    """Build a stand-in for a litellm.completion(...) response."""
    usage = usage or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    message = make_obj(content=text)
    choice = make_obj(message=message)
    return make_obj(choices=[choice], usage=usage)
```

- [ ] **Step 4: Add a smoke test to verify pytest + import wiring**

`tests/test_smoke.py`:

```python
import llm_batch


def test_package_imports():
    assert llm_batch is not None
```

- [ ] **Step 5: Install in editable mode and run the smoke test**

Run: `pip install -e ".[dev,postgres]" && pytest tests/test_smoke.py -v`
Expected: PASS (1 passed)

- [ ] **Step 6: Delete the smoke test and commit scaffolding**

```bash
rm tests/test_smoke.py
git add pyproject.toml llm_batch/ tests/conftest.py
git commit -m "chore: project scaffolding, packaging, and litellm test fixture"
```

---

## Task 1: Errors module

**Files:**
- Create: `llm_batch/errors.py`
- Test: `tests/test_errors.py`

- [ ] **Step 1: Write the failing test**

`tests/test_errors.py`:

```python
import pytest

from llm_batch.errors import BatchJobFailed, TransientProviderError


def test_batch_job_failed_carries_batch_id():
    err = BatchJobFailed("provider rejected batch", batch_job_id="batches/abc")
    assert err.batch_job_id == "batches/abc"
    assert "provider rejected batch" in str(err)


def test_batch_job_failed_batch_id_optional():
    err = BatchJobFailed("no id available")
    assert err.batch_job_id is None


def test_transient_provider_error_is_exception():
    assert issubclass(TransientProviderError, Exception)
    with pytest.raises(TransientProviderError):
        raise TransientProviderError("429 rate limited")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_errors.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm_batch.errors'`

- [ ] **Step 3: Write the implementation**

`llm_batch/errors.py`:

```python
"""Exceptions for llm_batch."""


class BatchJobFailed(Exception):
    """Raised when a batch job reaches a terminal failure state.

    `execute()` on a FAILED job re-raises this rather than silently retrying.
    """

    def __init__(self, message: str, batch_job_id: str | None = None):
        super().__init__(message)
        self.batch_job_id = batch_job_id


class TransientProviderError(Exception):
    """A retryable provider error (network blip, timeout, HTTP 429/5xx).

    Raised by the provider layer on submit. The Job leaves its state unchanged
    so the next run retries; nothing is marked FAILED.
    """
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_errors.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add llm_batch/errors.py tests/test_errors.py
git commit -m "feat: add BatchJobFailed and TransientProviderError"
```

---

## Task 2: Storage protocols

**Files:**
- Create: `llm_batch/storage/base.py`
- Test: `tests/test_storage_file.py` (the protocol-conformance check is added with FileStorage in Task 3; this task only defines the protocols)

- [ ] **Step 1: Write the implementation (protocols are structural; no separate failing test)**

`llm_batch/storage/base.py`:

```python
"""Storage protocols.

`Storage` is the per-job handle a Job holds (load/save/get/finish).
`StorageScanner` is the backend-level enumeration the Poller needs.

A backend object (FileStorage, PostgresStorage) is neither, exactly — it is a
*factory* that mints per-job Storage handles via `for_job(key)` / `for_meta()`
and enumerates them via `iter_jobs()`.
"""

from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable


@runtime_checkable
class Storage(Protocol):
    """Per-job state handle."""

    def load(self) -> bool:
        """Load existing state. Returns True if state existed."""
        ...

    def save(self, data: dict) -> None:
        """Persist the full job state dict."""
        ...

    def get(self, key: str, default=None):
        """Read one field from the loaded state."""
        ...

    def snapshot(self) -> dict:
        """Return a copy of the full loaded state dict."""
        ...

    def finish(self) -> None:
        """Delete / mark this job finished."""
        ...


@runtime_checkable
class StorageScanner(Protocol):
    """Backend-level enumeration used by the Poller."""

    def iter_jobs(self) -> Iterator[Storage]:
        """Yield a loaded per-job Storage handle for every persisted job."""
        ...
```

- [ ] **Step 2: Verify it imports**

Run: `python -c "from llm_batch.storage.base import Storage, StorageScanner; print('ok')"`
Expected: prints `ok`

- [ ] **Step 3: Commit**

```bash
git add llm_batch/storage/base.py
git commit -m "feat: add Storage and StorageScanner protocols"
```

---

## Task 3: FileStorage (atomic, deterministic JSON)

**Files:**
- Create: `llm_batch/storage/file.py`
- Test: `tests/test_storage_file.py`

- [ ] **Step 1: Write the failing test**

`tests/test_storage_file.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_storage_file.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm_batch.storage.file'`

- [ ] **Step 3: Write the implementation**

`llm_batch/storage/file.py`:

```python
"""FileStorage — one JSON file per job, atomic and git-friendly.

A `FileStorage(dir)` is a directory-scoped backend. It mints per-job handles
(`for_job`, `for_meta`) and enumerates them (`iter_jobs`). Each handle owns one
JSON file path and implements the per-job `Storage` protocol.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Iterator

META_FILENAME = "pipeline.json"


class _FileJobHandle:
    """Per-job Storage handle bound to a single JSON file."""

    def __init__(self, path: str):
        self._path = path
        self._data: dict = {}

    def load(self) -> bool:
        if not os.path.exists(self._path):
            return False
        with open(self._path, encoding="utf-8") as f:
            self._data = json.load(f)
        return True

    def save(self, data: dict) -> None:
        self._data = data
        directory = os.path.dirname(os.path.abspath(self._path))
        os.makedirs(directory, exist_ok=True)
        # Deterministic, merge-friendly serialization for clean git diffs.
        payload = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        # Atomic write: temp file in the same dir + os.replace.
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
            os.replace(tmp, self._path)
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def snapshot(self) -> dict:
        return dict(self._data)

    def finish(self) -> None:
        if os.path.exists(self._path):
            os.remove(self._path)


class FileStorage:
    """Directory-scoped backend that mints per-job handles and scans them."""

    def __init__(self, directory: str):
        self._dir = directory

    def for_job(self, key: str) -> _FileJobHandle:
        return _FileJobHandle(os.path.join(self._dir, f"{key}.json"))

    def for_meta(self) -> _FileJobHandle:
        return _FileJobHandle(os.path.join(self._dir, META_FILENAME))

    def iter_jobs(self) -> Iterator[_FileJobHandle]:
        """Walk the tree, yielding a loaded handle for every job file.

        Skips pipeline.json meta files and in-flight temp files.
        """
        for root, _dirs, files in os.walk(self._dir):
            for name in sorted(files):
                if name == META_FILENAME or not name.endswith(".json"):
                    continue
                if name.startswith(".tmp-"):
                    continue
                handle = _FileJobHandle(os.path.join(root, name))
                handle.load()
                yield handle
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_storage_file.py -v`
Expected: PASS (9 passed)

- [ ] **Step 5: Re-export from the storage package**

Modify `llm_batch/storage/__init__.py`:

```python
"""Storage backends for llm_batch."""

from llm_batch.storage.file import FileStorage

__all__ = ["FileStorage"]
```

- [ ] **Step 6: Commit**

```bash
git add llm_batch/storage/file.py llm_batch/storage/__init__.py tests/test_storage_file.py
git commit -m "feat: add FileStorage with atomic deterministic writes and recursive scanner"
```

---

## Task 4: Provider layer — dataclasses, `provider_of`, pricing, transient detection

**Files:**
- Create: `llm_batch/providers.py`
- Test: `tests/test_providers.py`

- [ ] **Step 1: Write the failing test**

`tests/test_providers.py`:

```python
import pytest

from llm_batch import providers
from llm_batch.providers import BatchHandle, CompletionResult, PollResult


def test_provider_of_vertex():
    assert providers.provider_of("vertex_ai/gemini-2.5-flash") == "vertex_ai"


def test_provider_of_anthropic():
    assert providers.provider_of("claude-3-5-sonnet-20241022") == "anthropic"
    assert providers.provider_of("anthropic/claude-3-5-sonnet") == "anthropic"


def test_provider_of_openai_bare_and_prefixed():
    assert providers.provider_of("gpt-4o") == "openai"
    assert providers.provider_of("azure/gpt-4o") == "azure"


def test_price_from_usage_sums_prompt_and_completion(litellm_mock):
    litellm_mock.cost_per_token = lambda **k: (0.01, 0.02)
    usd = providers._price_from_usage("gpt-4o", {"prompt_tokens": 100, "completion_tokens": 50})
    assert usd == pytest.approx(0.03)


def test_price_from_usage_swallows_errors(litellm_mock):
    def boom(**k):
        raise RuntimeError("no pricing data")
    litellm_mock.cost_per_token = boom
    assert providers._price_from_usage("weird/model", {"prompt_tokens": 1}) == 0.0


def test_is_transient_by_status_code():
    err = RuntimeError("boom")
    err.status_code = 429
    assert providers._is_transient(err) is True
    err.status_code = 400
    assert providers._is_transient(err) is False


def test_is_transient_by_type_name():
    class APITimeoutError(Exception):
        pass
    assert providers._is_transient(APITimeoutError("slow")) is True

    class BadRequestError(Exception):
        pass
    assert providers._is_transient(BadRequestError("nope")) is False


def test_dataclasses_have_expected_fields():
    h = BatchHandle(provider="openai", model="gpt-4o", batch_job_id="b1")
    assert h.input_file_id is None and h.output_file_id is None
    pr = PollResult(status="pending")
    assert pr.texts is None and pr.cost is None
    cr = CompletionResult(text="hi", usage={}, cost={})
    assert cr.text == "hi"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_providers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm_batch.providers'`

- [ ] **Step 3: Write the implementation (core only — submit/poll/complete/cleanup added in Tasks 5-7)**

`llm_batch/providers.py`:

```python
"""Provider layer — the only module that imports litellm.

Isolates batch submission, polling, interactive completion, pricing, and
provider-side cleanup behind a small, provider-agnostic interface so that
`job.py` never sees litellm or per-provider quirks.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass

import litellm

from llm_batch.errors import TransientProviderError

# Flat per-token batch discount: you pay this fraction of the interactive price.
# All three providers apply ~50% independent of batch size (see spec §10).
BATCH_RATE = 0.5

# litellm error status codes / type-name fragments treated as retryable.
_TRANSIENT_STATUS = {408, 409, 429, 500, 502, 503, 504}
_TRANSIENT_NAME_FRAGMENTS = (
    "timeout", "ratelimit", "serviceunavailable",
    "apiconnection", "internalserver", "overloaded",
)


@dataclass
class BatchHandle:
    """Everything needed to poll and clean up one submitted batch."""

    provider: str
    model: str
    batch_job_id: str
    input_file_id: str | None = None
    output_file_id: str | None = None


@dataclass
class PollResult:
    """Outcome of one poll of a batch job."""

    status: str  # "pending" | "completed" | "failed" | "transient_error"
    texts: list[str] | None = None
    usage: dict | None = None
    cost: dict | None = None
    output_file_id: str | None = None
    error_message: str | None = None


@dataclass
class CompletionResult:
    """Outcome of one interactive completion."""

    text: str
    usage: dict
    cost: dict


def provider_of(model: str) -> str:
    """Map a litellm model string to its custom_llm_provider."""
    if model.startswith("vertex_ai/"):
        return "vertex_ai"
    if model.startswith("claude") or model.startswith("anthropic/"):
        return "anthropic"
    if "/" in model:
        return model.split("/", 1)[0]
    return "openai"


def _price_from_usage(model: str, usage: dict) -> float:
    """Interactive (non-batch) USD for this usage, via litellm.cost_per_token."""
    try:
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )
        return float(prompt_cost + completion_cost)
    except Exception:
        return 0.0


def _cost_dict(model: str, usage: dict, *, batch: bool) -> dict:
    """Build the {charged_usd, interactive_usd, saved_usd} triple."""
    interactive_usd = _price_from_usage(model, usage)
    charged_usd = interactive_usd * BATCH_RATE if batch else interactive_usd
    return {
        "charged_usd": charged_usd,
        "interactive_usd": interactive_usd,
        "saved_usd": interactive_usd - charged_usd,
    }


def _is_transient(exc: Exception) -> bool:
    """Classify a provider exception as retryable (transient) vs terminal."""
    status = getattr(exc, "status_code", None)
    if status in _TRANSIENT_STATUS:
        return True
    name = type(exc).__name__.lower()
    return any(frag in name for frag in _TRANSIENT_NAME_FRAGMENTS)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_providers.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add llm_batch/providers.py tests/test_providers.py
git commit -m "feat: provider layer core — dataclasses, provider_of, pricing, transient detection"
```

---

## Task 5: Provider layer — `submit_batch` and `complete`

**Files:**
- Modify: `llm_batch/providers.py`
- Test: `tests/test_providers.py`

- [ ] **Step 1: Write the failing test (append to `tests/test_providers.py`)**

```python
from tests.conftest import completion_response, make_obj


def test_submit_batch_uploads_jsonl_and_creates_batch(litellm_mock):
    captured = {}

    def create_file(file, purpose, custom_llm_provider):
        captured["file"] = file
        captured["provider"] = custom_llm_provider
        return make_obj(id="files/in-1")

    def create_batch(**kwargs):
        captured["batch_kwargs"] = kwargs
        return make_obj(id="batches/job-1")

    litellm_mock.create_file = create_file
    litellm_mock.create_batch = create_batch

    handle = providers.submit_batch(
        "vertex_ai/gemini-2.5-flash",
        [[{"role": "user", "content": "hi"}], [{"role": "user", "content": "bye"}]],
        response_format={"type": "json_object"},
    )

    assert handle.provider == "vertex_ai"
    assert handle.batch_job_id == "batches/job-1"
    assert handle.input_file_id == "files/in-1"
    assert captured["provider"] == "vertex_ai"
    assert captured["batch_kwargs"]["input_file_id"] == "files/in-1"

    # JSONL body: one line per request, custom_id req-0..req-N, kwargs forwarded into body
    _name, buf, _mime = captured["file"]
    body = buf.getvalue().decode()
    lines = [json.loads(l) for l in body.splitlines() if l.strip()]
    assert [l["custom_id"] for l in lines] == ["req-0", "req-1"]
    assert lines[0]["body"]["messages"] == [{"role": "user", "content": "hi"}]
    assert lines[0]["body"]["response_format"] == {"type": "json_object"}


def test_submit_batch_wraps_transient_errors(litellm_mock):
    def boom(*a, **k):
        e = RuntimeError("429 rate limited")
        e.status_code = 429
        raise e
    litellm_mock.create_file = boom
    with pytest.raises(providers.TransientProviderError):
        providers.submit_batch("gpt-4o", [[{"role": "user", "content": "x"}]])


def test_submit_batch_reraises_terminal_errors(litellm_mock):
    def boom(*a, **k):
        e = ValueError("invalid model")  # not transient
        raise e
    litellm_mock.create_file = boom
    with pytest.raises(ValueError):
        providers.submit_batch("gpt-4o", [[{"role": "user", "content": "x"}]])


def test_complete_returns_text_usage_and_cost(litellm_mock):
    litellm_mock.completion = lambda **k: completion_response("the answer", {
        "prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})
    litellm_mock.cost_per_token = lambda **k: (0.01, 0.02)

    result = providers.complete("gpt-4o", [{"role": "user", "content": "q"}])
    assert result.text == "the answer"
    assert result.usage["total_tokens"] == 150
    # interactive path: charged == interactive, no savings
    assert result.cost["charged_usd"] == pytest.approx(0.03)
    assert result.cost["saved_usd"] == 0.0


def test_complete_forwards_kwargs(litellm_mock):
    seen = {}
    def completion(**k):
        seen.update(k)
        return completion_response("ok")
    litellm_mock.completion = completion
    litellm_mock.cost_per_token = lambda **k: (0.0, 0.0)
    providers.complete("gpt-4o", [{"role": "user", "content": "q"}],
                       response_format={"type": "json_object"}, temperature=0.2)
    assert seen["response_format"] == {"type": "json_object"}
    assert seen["temperature"] == 0.2


def test_complete_wraps_transient_errors(litellm_mock):
    def boom(**k):
        e = RuntimeError("timeout")
        raise e
    # name-based: RuntimeError isn't transient, so simulate a timeout-typed error
    class APITimeoutError(Exception):
        pass
    def boom2(**k):
        raise APITimeoutError("slow")
    litellm_mock.completion = boom2
    with pytest.raises(providers.TransientProviderError):
        providers.complete("gpt-4o", [{"role": "user", "content": "q"}])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_providers.py -k "submit_batch or complete" -v`
Expected: FAIL with `AttributeError: module 'llm_batch.providers' has no attribute 'submit_batch'`

- [ ] **Step 3: Add the implementation to `llm_batch/providers.py`**

Append these functions (and re-export `TransientProviderError` for test convenience):

```python
# Re-exported so tests/callers can reference providers.TransientProviderError.
TransientProviderError = TransientProviderError


def _build_jsonl(model: str, requests: list, **kw) -> str:
    """One JSON line per request; custom_id req-0..req-N; kwargs merged into body."""
    lines = []
    for i, messages in enumerate(requests):
        lines.append(json.dumps({
            "custom_id": f"req-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": model, "messages": messages, **kw},
        }))
    return "\n".join(lines) + "\n"


def submit_batch(model: str, requests: list, **kw) -> BatchHandle:
    """Upload a JSONL of all requests and create a batch job.

    `requests` is a list of message-lists (a single-request job passes a
    one-element list). Extra kwargs (response_format, tools, temperature, ...)
    are forwarded into each request body.
    """
    provider = provider_of(model)
    try:
        jsonl = _build_jsonl(model, requests, **kw)
        file_obj = litellm.create_file(
            file=("batch.jsonl", io.BytesIO(jsonl.encode()), "application/jsonl"),
            purpose="batch",
            custom_llm_provider=provider,
        )
        batch = litellm.create_batch(
            completion_window="24h",
            endpoint="/v1/chat/completions",
            input_file_id=file_obj.id,
            custom_llm_provider=provider,
            model=model,
        )
    except Exception as exc:
        if _is_transient(exc):
            raise TransientProviderError(str(exc)) from exc
        raise
    return BatchHandle(
        provider=provider,
        model=model,
        batch_job_id=batch.id,
        input_file_id=file_obj.id,
    )


def complete(model: str, messages: list, **kw) -> CompletionResult:
    """Interactive path: one immediate completion with usage + cost."""
    try:
        response = litellm.completion(model=model, messages=messages, **kw)
    except Exception as exc:
        if _is_transient(exc):
            raise TransientProviderError(str(exc)) from exc
        raise
    text = response.choices[0].message.content
    usage = _usage_dict(response.usage)
    cost = _cost_dict(model, usage, batch=False)
    return CompletionResult(text=text, usage=usage, cost=cost)


def _usage_dict(usage) -> dict:
    """Normalize a litellm usage object (or dict) into a plain dict."""
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if isinstance(usage, dict):
        src = usage
    else:
        src = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
        }
    return {
        "prompt_tokens": src.get("prompt_tokens", 0),
        "completion_tokens": src.get("completion_tokens", 0),
        "total_tokens": src.get("total_tokens", 0),
    }
```

> Note: the `completion_response` helper in conftest sets `usage` as a dict, so `_usage_dict` handles both the dict and attribute forms.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_providers.py -v`
Expected: PASS (all providers tests pass)

- [ ] **Step 5: Commit**

```bash
git add llm_batch/providers.py tests/test_providers.py
git commit -m "feat: provider layer submit_batch and complete with kwarg passthrough and cost capture"
```

---

## Task 6: Provider layer — `poll_batch` (transient vs terminal)

**Files:**
- Modify: `llm_batch/providers.py`
- Test: `tests/test_providers.py`

- [ ] **Step 1: Write the failing test (append to `tests/test_providers.py`)**

```python
from tests.conftest import batch_output_jsonl


def _handle(model="gpt-4o"):
    return BatchHandle(provider=providers.provider_of(model), model=model,
                       batch_job_id="batches/job-1", input_file_id="files/in-1")


def test_poll_batch_pending(litellm_mock):
    litellm_mock.retrieve_batch = lambda **k: make_obj(status="in_progress")
    result = providers.poll_batch(_handle())
    assert result.status == "pending"


def test_poll_batch_completed_single(litellm_mock):
    litellm_mock.retrieve_batch = lambda **k: make_obj(
        status="completed", output_file_id="files/out-1")
    litellm_mock.file_content = lambda **k: batch_output_jsonl(["extracted text"])
    litellm_mock.cost_per_token = lambda **k: (0.01, 0.02)

    result = providers.poll_batch(_handle())
    assert result.status == "completed"
    assert result.texts == ["extracted text"]
    assert result.output_file_id == "files/out-1"
    # batch discount: charged is half of interactive
    assert result.cost["charged_usd"] == pytest.approx(result.cost["interactive_usd"] * 0.5)
    assert result.cost["saved_usd"] == pytest.approx(result.cost["interactive_usd"] * 0.5)


def test_poll_batch_completed_bundle_ordered_by_custom_id(litellm_mock):
    litellm_mock.retrieve_batch = lambda **k: make_obj(
        status="completed", output_file_id="files/out-1")
    litellm_mock.file_content = lambda **k: batch_output_jsonl(["page one", "page two", "page three"])
    litellm_mock.cost_per_token = lambda **k: (0.0, 0.0)
    result = providers.poll_batch(_handle())
    assert result.texts == ["page one", "page two", "page three"]
    assert result.usage["total_tokens"] == 45  # 3 requests * 15


def test_poll_batch_provider_failed_is_terminal(litellm_mock):
    litellm_mock.retrieve_batch = lambda **k: make_obj(status="failed")
    result = providers.poll_batch(_handle())
    assert result.status == "failed"


def test_poll_batch_expired_is_terminal(litellm_mock):
    litellm_mock.retrieve_batch = lambda **k: make_obj(status="expired")
    assert providers.poll_batch(_handle()).status == "failed"


def test_poll_batch_per_request_error_is_terminal(litellm_mock):
    from tests.conftest import make_obj as mk
    bad_body = '{"custom_id": "req-0", "response": {"status_code": 400, "body": {}}}\n'
    litellm_mock.retrieve_batch = lambda **k: mk(status="completed", output_file_id="files/out-1")
    litellm_mock.file_content = lambda **k: mk(text=bad_body, content=bad_body.encode())
    result = providers.poll_batch(_handle())
    assert result.status == "failed"
    assert "req-0" in result.error_message


def test_poll_batch_transient_on_retrieve(litellm_mock):
    def boom(**k):
        e = RuntimeError("503 unavailable")
        e.status_code = 503
        raise e
    litellm_mock.retrieve_batch = boom
    result = providers.poll_batch(_handle())
    assert result.status == "transient_error"


def test_poll_batch_transient_on_file_content(litellm_mock):
    litellm_mock.retrieve_batch = lambda **k: make_obj(status="completed", output_file_id="files/out-1")
    def boom(**k):
        e = RuntimeError("429")
        e.status_code = 429
        raise e
    litellm_mock.file_content = boom
    result = providers.poll_batch(_handle())
    assert result.status == "transient_error"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_providers.py -k poll_batch -v`
Expected: FAIL with `AttributeError: module 'llm_batch.providers' has no attribute 'poll_batch'`

- [ ] **Step 3: Add the implementation to `llm_batch/providers.py`**

```python
# Provider batch statuses that mean "still working".
_PENDING_STATUSES = {"validating", "in_progress", "finalizing"}
# Provider batch statuses that mean "terminally dead".
_FAILED_STATUSES = {"failed", "expired", "cancelled", "canceling", "cancelling"}


def poll_batch(handle: BatchHandle) -> PollResult:
    """Poll one batch job once and classify the outcome.

    Returns a PollResult with status pending | completed | failed | transient_error.
    Transient errors (429/timeout/5xx) never become `failed` — the Job keeps
    polling (spec §8.1).
    """
    provider = handle.provider
    try:
        batch = litellm.retrieve_batch(
            batch_id=handle.batch_job_id, custom_llm_provider=provider)
    except Exception as exc:
        if _is_transient(exc):
            return PollResult(status="transient_error", error_message=str(exc))
        raise

    status = getattr(batch, "status", None)
    if status in _PENDING_STATUSES:
        return PollResult(status="pending")
    if status in _FAILED_STATUSES:
        return PollResult(
            status="failed",
            error_message=f"batch {handle.batch_job_id} reported status={status}",
        )
    if status != "completed":
        return PollResult(
            status="failed",
            error_message=f"batch {handle.batch_job_id} unknown status={status}",
        )

    output_file_id = getattr(batch, "output_file_id", None)
    try:
        content = litellm.file_content(
            file_id=output_file_id, custom_llm_provider=provider)
    except Exception as exc:
        if _is_transient(exc):
            return PollResult(status="transient_error", error_message=str(exc))
        raise

    raw = content.text if hasattr(content, "text") else content.content.decode()
    results_by_id: dict[str, str] = {}
    total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for line in raw.splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        custom_id = obj.get("custom_id")
        if obj.get("error"):
            return PollResult(
                status="failed", output_file_id=output_file_id,
                error_message=f"{custom_id}: {obj['error']}")
        response = obj.get("response", {})
        status_code = response.get("status_code", 200)
        if status_code != 200:
            return PollResult(
                status="failed", output_file_id=output_file_id,
                error_message=f"{custom_id}: request failed with status {status_code}")
        body = response["body"]
        results_by_id[custom_id] = body["choices"][0]["message"]["content"]
        usage = _usage_dict(body.get("usage"))
        total["prompt_tokens"] += usage["prompt_tokens"]
        total["completion_tokens"] += usage["completion_tokens"]
        total["total_tokens"] += usage["total_tokens"]

    texts = [results_by_id[f"req-{i}"] for i in range(len(results_by_id))]
    cost = _cost_dict(handle.model, total, batch=True)
    return PollResult(
        status="completed", texts=texts, usage=total, cost=cost,
        output_file_id=output_file_id)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_providers.py -k poll_batch -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add llm_batch/providers.py tests/test_providers.py
git commit -m "feat: provider layer poll_batch with transient/terminal classification and cost capture"
```

---

## Task 7: Provider layer — `cleanup` and `cleanup_orphans`

**Files:**
- Modify: `llm_batch/providers.py`
- Test: `tests/test_providers.py`

- [ ] **Step 1: Write the failing test (append to `tests/test_providers.py`)**

```python
def test_cleanup_openai_deletes_input_and_output_files(litellm_mock):
    deleted = []
    litellm_mock.file_delete = lambda file_id, custom_llm_provider: deleted.append(file_id)
    handle = BatchHandle(provider="openai", model="gpt-4o", batch_job_id="b1",
                         input_file_id="files/in-1", output_file_id="files/out-1")
    providers.cleanup(handle)
    assert set(deleted) == {"files/in-1", "files/out-1"}


def test_cleanup_openai_is_best_effort(litellm_mock):
    def boom(**k):
        raise RuntimeError("already deleted")
    litellm_mock.file_delete = boom
    handle = BatchHandle(provider="openai", model="gpt-4o", batch_job_id="b1",
                         input_file_id="files/in-1")
    providers.cleanup(handle)  # must not raise


def test_cleanup_anthropic_is_noop(litellm_mock):
    def boom(**k):
        raise AssertionError("should not delete for anthropic")
    litellm_mock.file_delete = boom
    handle = BatchHandle(provider="anthropic", model="claude-3-5-sonnet", batch_job_id="b1")
    providers.cleanup(handle)  # no-op, no call


def test_cleanup_vertex_deletes_gcs_objects(monkeypatch):
    deleted = []

    class FakeBlob:
        def __init__(self, uri):
            self.uri = uri
        def delete(self):
            deleted.append(self.uri)

    def fake_delete_gcs(uri):
        FakeBlob(uri).delete()

    monkeypatch.setattr(providers, "_delete_gcs_object", fake_delete_gcs)
    handle = BatchHandle(provider="vertex_ai", model="vertex_ai/gemini-2.5-flash",
                         batch_job_id="b1",
                         input_file_id="gs://bucket/in.jsonl",
                         output_file_id="gs://bucket/out/")
    providers.cleanup(handle)
    assert deleted == ["gs://bucket/in.jsonl", "gs://bucket/out/"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_providers.py -k cleanup -v`
Expected: FAIL with `AttributeError: module 'llm_batch.providers' has no attribute 'cleanup'`

- [ ] **Step 3: Add the implementation to `llm_batch/providers.py`**

```python
def cleanup(handle: BatchHandle) -> None:
    """Delete provider-side batch artifacts. Best-effort; never raises.

    Per spec §7: OpenAI deletes file objects (quota); Vertex deletes GCS objects
    (billing); Gemini-API and Anthropic auto-expire so this is a no-op.
    """
    provider = handle.provider
    if provider == "openai":
        for file_id in (handle.input_file_id, handle.output_file_id):
            if not file_id:
                continue
            try:
                litellm.file_delete(file_id=file_id, custom_llm_provider="openai")
            except Exception:
                pass  # best-effort: result is already persisted before we get here
    elif provider == "vertex_ai":
        for uri in (handle.input_file_id, handle.output_file_id):
            if not uri:
                continue
            try:
                _delete_gcs_object(uri)
            except Exception:
                pass
    # anthropic / gemini: nothing to clean up (free, auto-expiring)


def _delete_gcs_object(uri: str) -> None:
    """Delete a single gs://bucket/path object (or all objects under a prefix)."""
    if not uri.startswith("gs://"):
        return
    from google.cloud import storage  # lazy: only Vertex callers need this

    without_scheme = uri[len("gs://"):]
    bucket_name, _, path = without_scheme.partition("/")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    if uri.endswith("/"):
        for blob in client.list_blobs(bucket_name, prefix=path):
            blob.delete()
    else:
        bucket.blob(path).delete()


def cleanup_orphans(provider: str, live_ids: set, older_than_days: int) -> list[str]:
    """Best-effort sweep of provider-side files with no live local state.

    Implemented for OpenAI (file objects) via the openai SDK; a no-op for other
    providers (Vertex/Anthropic/Gemini), where listing is provider-specific or
    unnecessary. Returns the list of deleted ids. Never raises.
    """
    if provider != "openai":
        return []
    import time

    try:
        from openai import OpenAI

        client = OpenAI()
        cutoff = time.time() - older_than_days * 86400
        deleted = []
        for file_obj in client.files.list(purpose="batch"):
            if file_obj.id in live_ids:
                continue
            created = getattr(file_obj, "created_at", None)
            if created is not None and created > cutoff:
                continue
            try:
                client.files.delete(file_obj.id)
                deleted.append(file_obj.id)
            except Exception:
                pass
        return deleted
    except Exception:
        return []
```

> The `time` import inside `cleanup_orphans` keeps `time.time()` out of the unit-tested core (the poller test in Task 13 monkeypatches `providers.cleanup_orphans` rather than exercising the openai SDK).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_providers.py -k cleanup -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Run the full providers suite**

Run: `pytest tests/test_providers.py -v`
Expected: PASS (all providers tests)

- [ ] **Step 6: Commit**

```bash
git add llm_batch/providers.py tests/test_providers.py
git commit -m "feat: provider-side cleanup (OpenAI/Vertex) and best-effort orphan sweep"
```

---

## Task 8: Job state machine core

Implements `execute()` for all four states, single + bundle shapes, interactive and batch submit, polling with transient/terminal handling, idempotency, the `result` property, and persist-before-cleanup ordering. The deadline fallback is added in Task 9; here `_should_use_interactive()` only honors an explicit `mode="interactive"`.

**Files:**
- Create: `llm_batch/job.py`
- Test: `tests/test_job.py`

- [ ] **Step 1: Write the failing test**

`tests/test_job.py`:

```python
import pytest

from llm_batch import job as job_module
from llm_batch.job import Job
from llm_batch.errors import BatchJobFailed
from llm_batch.providers import BatchHandle, PollResult, CompletionResult
from llm_batch.storage.file import FileStorage


@pytest.fixture
def fake_providers(monkeypatch):
    """Stub the provider layer for Job tests; configurable per test."""
    calls = {"submit": 0, "poll": 0, "complete": 0, "cleanup": []}

    def submit_batch(model, requests, **kw):
        calls["submit"] += 1
        return BatchHandle(provider="openai", model=model,
                           batch_job_id="batches/job-1", input_file_id="files/in-1")

    def complete(model, messages, **kw):
        calls["complete"] += 1
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


def _handle(tmp_path, key="ocr"):
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_job.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm_batch.job'`

- [ ] **Step 3: Write the implementation**

`llm_batch/job.py`:

```python
"""Job — the unit of resumable work and its execute() state machine."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from llm_batch import providers
from llm_batch.errors import BatchJobFailed
from llm_batch.providers import BatchHandle


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now().isoformat()


def _parse_iso(value: str | None) -> datetime | None:
    if value is None:
        return None
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


_ZERO_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
_ZERO_COST = {"charged_usd": 0.0, "interactive_usd": 0.0, "saved_usd": 0.0}

# Fields reconstructed from storage when resuming a job.
_PERSISTED_FIELDS = (
    "key", "state", "mode", "model", "n_requests", "bundle", "batch_job_id",
    "input_file_id", "output_file_id", "submitted_at", "completed_at", "deadline",
    "actual_mode", "response", "usage", "cost", "error_message", "poll_attempts",
    "rerun_cmd", "cleanup_provider_files",
)


def _accumulate(target: dict, src: dict) -> None:
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        target[k] = target.get(k, 0) + (src or {}).get(k, 0)


def _accumulate_cost(target: dict, src: dict) -> None:
    for k in ("charged_usd", "interactive_usd", "saved_usd"):
        target[k] = target.get(k, 0.0) + (src or {}).get(k, 0.0)


class Job:
    """One or many requests, a model, a mode, and a storage handle."""

    NOT_SENT = "not_sent"
    SENT = "sent"
    COMPLETED = "completed"
    FAILED = "failed"

    def __init__(self, storage, key=None, *, messages=None, requests=None,
                 model=None, mode="batch", deadline=None,
                 max_batch_turnaround_hours=24, rerun_cmd=None,
                 cleanup_provider_files=True, **litellm_kwargs):
        if messages is not None and requests is not None:
            raise ValueError("pass either messages= or requests=, not both")

        self._storage = storage
        self._key = key
        self._is_bundle = requests is not None
        if requests is not None:
            self._payload = list(requests)
        elif messages is not None:
            self._payload = [messages]
        else:
            self._payload = None  # poll-only reconstruction

        self._model = model
        self._mode = mode
        self._deadline = deadline
        self._max_turnaround = max_batch_turnaround_hours
        self._rerun_cmd = rerun_cmd
        self._cleanup = cleanup_provider_files
        self._kw = litellm_kwargs
        self._state: dict = {}

        if self._storage.load():
            self._state = {f: self._storage.get(f) for f in _PERSISTED_FIELDS}
            # Persisted values win for resumption.
            self._key = self._state.get("key", self._key)
            self._model = self._state.get("model") or self._model
            self._mode = self._state.get("mode") or self._mode
            self._deadline = self._state.get("deadline", self._deadline)
            if self._state.get("bundle") is not None:
                self._is_bundle = bool(self._state.get("bundle"))
            if self._state.get("rerun_cmd"):
                self._rerun_cmd = self._state.get("rerun_cmd")
            if self._state.get("cleanup_provider_files") is not None:
                self._cleanup = self._state.get("cleanup_provider_files")

    # --- public API -----------------------------------------------------

    @property
    def state(self) -> str:
        return self._state.get("state", self.NOT_SENT)

    @property
    def result(self):
        if self.state != self.COMPLETED:
            return None
        return self._state.get("response")

    def snapshot(self) -> dict:
        return dict(self._state)

    def execute(self):
        state = self.state
        if state == self.COMPLETED:
            return self._state.get("response")
        if state == self.FAILED:
            raise BatchJobFailed(
                self._state.get("error_message") or "job previously failed",
                self._state.get("batch_job_id"))
        if state == self.SENT:
            return self._poll()
        return self._submit()  # NOT_SENT

    # --- internals ------------------------------------------------------

    def _should_use_interactive(self) -> bool:
        # Deadline-aware fallback is added in Task 9; for now only an explicit
        # interactive mode forces the immediate path.
        return self._mode == "interactive"

    def _base_state(self) -> dict:
        return {
            "key": self._key,
            "model": self._model,
            "mode": self._mode,
            "n_requests": len(self._payload) if self._payload else 0,
            "bundle": self._is_bundle,
            "deadline": self._deadline,
            "rerun_cmd": self._rerun_cmd,
            "cleanup_provider_files": self._cleanup,
            "submitted_at": _now_iso(),
            "completed_at": None,
            "batch_job_id": None,
            "input_file_id": None,
            "output_file_id": None,
            "response": None,
            "usage": dict(_ZERO_USAGE),
            "cost": dict(_ZERO_COST),
            "error_message": None,
            "poll_attempts": 0,
        }

    def _save(self) -> None:
        self._storage.save(self._state)

    def _submit(self):
        if self._payload is None:
            raise RuntimeError(f"job {self._key!r} has no request payload to submit")

        if self._should_use_interactive():
            return self._submit_interactive()
        return self._submit_batch()

    def _submit_interactive(self):
        texts = []
        usage = dict(_ZERO_USAGE)
        cost = dict(_ZERO_COST)
        try:
            for messages in self._payload:
                cr = providers.complete(self._model, messages, **self._kw)
                texts.append(cr.text)
                _accumulate(usage, cr.usage)
                _accumulate_cost(cost, cr.cost)
        except providers.TransientProviderError:
            return None  # leave state NOT_SENT; retried next run

        response = texts if self._is_bundle else texts[0]
        state = self._base_state()
        state.update(state=self.COMPLETED, actual_mode="interactive",
                     completed_at=_now_iso(), response=response,
                     usage=usage, cost=cost)
        self._state = state
        self._save()
        return response

    def _submit_batch(self):
        try:
            handle = providers.submit_batch(self._model, self._payload, **self._kw)
        except providers.TransientProviderError:
            return None  # leave state NOT_SENT; retried next run

        state = self._base_state()
        state.update(state=self.SENT, actual_mode="batch",
                     batch_job_id=handle.batch_job_id,
                     input_file_id=handle.input_file_id)
        self._state = state
        self._save()
        return None

    def _batch_handle(self) -> BatchHandle:
        return BatchHandle(
            provider=providers.provider_of(self._model),
            model=self._model,
            batch_job_id=self._state.get("batch_job_id"),
            input_file_id=self._state.get("input_file_id"),
            output_file_id=self._state.get("output_file_id"),
        )

    def _poll(self):
        handle = self._batch_handle()
        result = providers.poll_batch(handle)

        if result.status in ("pending", "transient_error"):
            self._state["poll_attempts"] = self._state.get("poll_attempts", 0) + 1
            if result.status == "transient_error":
                self._state["error_message"] = result.error_message
            self._save()
            return None

        if result.status == "failed":
            self._mark_failed(result.error_message, handle, result.output_file_id)
            raise BatchJobFailed(result.error_message or "batch failed",
                                 handle.batch_job_id)

        # completed
        response = result.texts if self._is_bundle else result.texts[0]
        self._state.update(
            state=self.COMPLETED, response=response, completed_at=_now_iso(),
            usage=result.usage, cost=result.cost,
            output_file_id=result.output_file_id, error_message=None)
        self._save()  # persist BEFORE provider-side cleanup (spec §7.1)
        if self._cleanup:
            handle.output_file_id = result.output_file_id
            try:
                providers.cleanup(handle)
            except Exception:
                pass  # best-effort; result already durably saved
        return response

    def _mark_failed(self, message, handle=None, output_file_id=None):
        self._state["state"] = self.FAILED
        self._state["error_message"] = message
        self._state["completed_at"] = _now_iso()
        if output_file_id:
            self._state["output_file_id"] = output_file_id
        self._save()
        if handle is not None and self._cleanup:
            if output_file_id:
                handle.output_file_id = output_file_id
            try:
                providers.cleanup(handle)
            except Exception:
                pass
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_job.py -v`
Expected: PASS (11 passed)

- [ ] **Step 5: Commit**

```bash
git add llm_batch/job.py tests/test_job.py
git commit -m "feat: Job execute() state machine — submit/poll/idempotency, single+bundle, persist-before-cleanup"
```

---

## Task 9: Deadline-aware fallback

Enhance `_should_use_interactive()` so that, in batch mode, a job submitted with too little time before the deadline falls back to an immediate interactive call (spec §4). `actual_mode` records what actually happened.

**Files:**
- Modify: `llm_batch/job.py` (`_should_use_interactive`)
- Test: `tests/test_job_fallback.py`

- [ ] **Step 1: Write the failing test**

`tests/test_job_fallback.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_job_fallback.py -v`
Expected: FAIL — `test_tight_deadline_falls_back_to_interactive` fails (job still submits as batch because `_should_use_interactive` ignores the deadline)

- [ ] **Step 3: Update `_should_use_interactive` in `llm_batch/job.py`**

Replace the method body:

```python
    def _should_use_interactive(self) -> bool:
        if self._mode == "interactive":
            return True
        if self._deadline is None:
            return False
        deadline_dt = _parse_iso(self._deadline)
        if deadline_dt is None:
            return False
        hours_remaining = (deadline_dt - _now()).total_seconds() / 3600.0
        # Too little time for another worst-case batch round → go interactive now.
        return hours_remaining < self._max_turnaround
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_job_fallback.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add llm_batch/job.py tests/test_job_fallback.py
git commit -m "feat: deadline-aware batch→interactive fallback (conservative strategy)"
```

---

## Task 10: `set_completed()` and `retry()`

`set_completed(text)` marks a job COMPLETED with no LLM call (e.g. when `fitz` extracts PDF text directly). `retry()` resets a FAILED job to NOT_SENT for an explicit re-submit.

**Files:**
- Modify: `llm_batch/job.py`
- Test: `tests/test_job_preset_retry.py`

- [ ] **Step 1: Write the failing test**

`tests/test_job_preset_retry.py`:

```python
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
    assert fake_providers["submit"] == 1


def test_retry_noop_if_not_failed(tmp_path, fake_providers):
    job = Job(FileStorage(str(tmp_path)).for_job("x"), "x",
              messages=[{"role": "user", "content": "hi"}], model="gpt-4o", mode="batch")
    job.execute()                                   # SENT
    job.retry()                                     # ignored (not FAILED)
    assert job.state == Job.SENT
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_job_preset_retry.py -v`
Expected: FAIL with `AttributeError: 'Job' object has no attribute 'set_completed'`

- [ ] **Step 3: Add the methods to `llm_batch/job.py` (after `execute`)**

```python
    def set_completed(self, text) -> None:
        """Mark this job COMPLETED with a preset result and no LLM call.

        Used when the caller already has the answer (e.g. fitz extracted the
        PDF text directly). `text` is stored verbatim as the response.
        """
        if self.state == self.COMPLETED:
            return
        state = self._base_state()
        state.update(state=self.COMPLETED, actual_mode="preset",
                     completed_at=_now_iso(), response=text)
        self._state = state
        self._save()

    def retry(self) -> None:
        """Reset a FAILED job to NOT_SENT so the next execute() resubmits."""
        if self.state != self.FAILED:
            return
        self._state["state"] = self.NOT_SENT
        self._state["error_message"] = None
        self._save()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_job_preset_retry.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add llm_batch/job.py tests/test_job_preset_retry.py
git commit -m "feat: Job.set_completed() preset and Job.retry() reset"
```

---

## Task 11: Pipeline

A thin wrapper over a shared state dir + config. Records `pipeline_started_at` once in `pipeline.json`, converts `time_budget_hours` sugar into an absolute `deadline`, mints namespaced jobs that inherit config, aggregates cost, and cleans up local state on `finish()`.

**Files:**
- Create: `llm_batch/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

`tests/test_pipeline.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm_batch.pipeline'`

- [ ] **Step 3: Write the implementation**

`llm_batch/pipeline.py`:

```python
"""Pipeline — shared state dir + config across a multi-step job sequence."""

from __future__ import annotations

from datetime import timedelta

from llm_batch.job import Job, _now, _now_iso
from llm_batch.storage.file import FileStorage


class Pipeline:
    """Convenience wrapper that mints namespaced Jobs over one state dir.

    A bare Job works standalone; Pipeline just removes boilerplate and records
    the deadline anchor (`pipeline_started_at`) once in pipeline.json.
    """

    def __init__(self, state_dir, *, mode="batch", deadline=None,
                 time_budget_hours=None, max_batch_turnaround_hours=24,
                 rerun_cmd=None, storage=None, cleanup_provider_files=True):
        self._state_dir = state_dir
        self._mode = mode
        self._max_turnaround = max_batch_turnaround_hours
        self._rerun_cmd = rerun_cmd
        self._cleanup = cleanup_provider_files
        self._backend = storage if storage is not None else FileStorage(state_dir)
        self._meta = self._load_or_init_meta(deadline, time_budget_hours)
        self._deadline = self._meta.get("deadline")

    def _load_or_init_meta(self, deadline, time_budget_hours) -> dict:
        handle = self._backend.for_meta()
        if handle.load():
            return handle.snapshot()
        started = _now()
        resolved_deadline = deadline
        if resolved_deadline is None and time_budget_hours is not None:
            resolved_deadline = (started + timedelta(hours=time_budget_hours)).isoformat()
        meta = {
            "pipeline_started_at": started.isoformat(),
            "deadline": resolved_deadline,
            "mode": self._mode,
            "rerun_cmd": self._rerun_cmd,
        }
        handle.save(meta)
        return meta

    def meta(self) -> dict:
        return dict(self._meta)

    def save_meta(self, extra: dict) -> None:
        """Merge extra fields into pipeline.json (e.g. pdf_url, domain deadline)."""
        self._meta.update(extra)
        self._backend.for_meta().save(self._meta)

    def job(self, key, *, messages=None, requests=None, model=None, **litellm_kwargs) -> Job:
        return Job(
            self._backend.for_job(key), key,
            messages=messages, requests=requests, model=model,
            mode=self._mode, deadline=self._deadline,
            max_batch_turnaround_hours=self._max_turnaround,
            rerun_cmd=self._rerun_cmd, cleanup_provider_files=self._cleanup,
            **litellm_kwargs)

    def cost_summary(self) -> dict:
        total = {"charged_usd": 0.0, "interactive_usd": 0.0, "saved_usd": 0.0}
        for handle in self._backend.iter_jobs():
            cost = handle.get("cost") or {}
            for k in total:
                total[k] += cost.get(k, 0.0)
        return total

    def finish(self) -> None:
        """Delete this pipeline's local state files."""
        for handle in list(self._backend.iter_jobs()):
            handle.finish()
        self._backend.for_meta().finish()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pipeline.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add llm_batch/pipeline.py tests/test_pipeline.py
git commit -m "feat: Pipeline with meta anchor, time-budget sugar, cost aggregation, finish"
```

---

## Task 12: Poller

Scans every `SENT` job in a state tree, polls each once (reusing `Job.execute()`), advances each completed *pipeline* exactly once via its `rerun_cmd`, fails jobs stuck past `max_batch_age`, and offers an opt-in provider orphan sweep. Adds a public `Job.mark_failed()` used for the stale path.

**Files:**
- Modify: `llm_batch/job.py` (add `mark_failed`)
- Create: `llm_batch/poller.py`
- Test: `tests/test_poller.py`

- [ ] **Step 1: Write the failing test**

`tests/test_poller.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_poller.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm_batch.poller'`

- [ ] **Step 3: Add `mark_failed` to `llm_batch/job.py` (after `retry`)**

```python
    def mark_failed(self, message: str) -> None:
        """Force this job to FAILED and attempt provider cleanup (poller use)."""
        self._mark_failed(message, self._batch_handle(),
                          self._state.get("output_file_id"))
```

- [ ] **Step 4: Write `llm_batch/poller.py`**

```python
"""Poller — central advancement of SENT jobs across a state tree."""

from __future__ import annotations

import shlex
import subprocess
from datetime import timedelta

from llm_batch import providers
from llm_batch.errors import BatchJobFailed
from llm_batch.job import Job, _now, _parse_iso


class Poller:
    """Scans SENT jobs, advances completed pipelines, fails stale jobs.

    `backend` is a directory/namespace-level storage object exposing
    `iter_jobs()` (and `for_job`). `run_cmd` is injectable for testing; it
    defaults to running the command list with subprocess.
    """

    def __init__(self, backend, *, max_batch_age_days=3, run_cmd=None):
        self._backend = backend
        self._max_age = timedelta(days=max_batch_age_days)
        self._run = run_cmd if run_cmd is not None else _default_run

    def poll_all(self) -> dict:
        rerun_cmds: list[str] = []
        seen_cmds: set[str] = set()

        for handle in self._backend.iter_jobs():
            if handle.get("state") != Job.SENT:
                continue

            if self._is_stale(handle):
                job = Job(handle)
                job.mark_failed(f"stale: no completion within {self._max_age.days}d")
                continue

            job = Job(handle)
            try:
                job.execute()  # polls once
            except BatchJobFailed:
                continue       # terminal failure → no rerun

            if job.state == Job.COMPLETED:
                cmd = handle.get("rerun_cmd")
                if cmd and cmd not in seen_cmds:
                    seen_cmds.add(cmd)
                    rerun_cmds.append(cmd)

        for cmd in rerun_cmds:
            self._run(shlex.split(cmd))

        return {"advanced": rerun_cmds}

    def _is_stale(self, handle) -> bool:
        submitted = handle.get("submitted_at")
        if not submitted:
            return False
        submitted_dt = _parse_iso(submitted)
        return (_now() - submitted_dt) > self._max_age

    def sweep_orphans(self, older_than_days: int = 3) -> dict:
        """Best-effort deletion of provider-side files with no live local state."""
        live_ids: set[str] = set()
        provider_set: set[str] = set()
        for handle in self._backend.iter_jobs():
            for field in ("batch_job_id", "input_file_id", "output_file_id"):
                value = handle.get(field)
                if value:
                    live_ids.add(value)
            model = handle.get("model")
            if model:
                provider_set.add(providers.provider_of(model))

        provider_set.add("openai")  # the one provider with a concrete sweep
        deleted: dict[str, list] = {}
        for provider in provider_set:
            deleted[provider] = providers.cleanup_orphans(
                provider, live_ids, older_than_days)
        return deleted


def _default_run(argv: list[str]) -> None:
    subprocess.run(argv, check=False)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_poller.py -v`
Expected: PASS (6 passed)

- [ ] **Step 6: Commit**

```bash
git add llm_batch/job.py llm_batch/poller.py tests/test_poller.py
git commit -m "feat: Poller — advance completed pipelines via rerun_cmd, stale timeout, orphan sweep"
```

---

## Task 13: CLI (`llm-batch poll`)

**Files:**
- Create: `llm_batch/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

`tests/test_cli.py`:

```python
import pytest

from llm_batch import cli as cli_module


def test_poll_command_invokes_poller(monkeypatch, tmp_path):
    calls = {"poll_all": 0, "sweep": None, "max_age": None, "state_dir": None}

    class FakePoller:
        def __init__(self, backend, *, max_batch_age_days=3, run_cmd=None):
            calls["max_age"] = max_batch_age_days
            calls["state_dir"] = backend._dir       # FileStorage stores dir in _dir
        def poll_all(self):
            calls["poll_all"] += 1
            return {"advanced": []}
        def sweep_orphans(self, older_than_days):
            calls["sweep"] = older_than_days

    monkeypatch.setattr(cli_module, "Poller", FakePoller)
    rc = cli_module.main(["poll", str(tmp_path)])
    assert rc == 0
    assert calls["poll_all"] == 1
    assert calls["sweep"] is None              # not requested
    assert calls["max_age"] == 3
    assert calls["state_dir"] == str(tmp_path)


def test_poll_command_with_sweep_orphans(monkeypatch, tmp_path):
    swept = {}

    class FakePoller:
        def __init__(self, backend, *, max_batch_age_days=3, run_cmd=None):
            pass
        def poll_all(self):
            return {"advanced": []}
        def sweep_orphans(self, older_than_days):
            swept["days"] = older_than_days

    monkeypatch.setattr(cli_module, "Poller", FakePoller)
    rc = cli_module.main(["poll", str(tmp_path), "--sweep-orphans", "5"])
    assert rc == 0
    assert swept["days"] == 5


def test_poll_command_custom_max_age(monkeypatch, tmp_path):
    seen = {}

    class FakePoller:
        def __init__(self, backend, *, max_batch_age_days=3, run_cmd=None):
            seen["max_age"] = max_batch_age_days
        def poll_all(self):
            return {"advanced": []}
        def sweep_orphans(self, older_than_days):
            pass

    monkeypatch.setattr(cli_module, "Poller", FakePoller)
    cli_module.main(["poll", str(tmp_path), "--max-batch-age-days", "7"])
    assert seen["max_age"] == 7


def test_no_command_errors(monkeypatch):
    with pytest.raises(SystemExit):
        cli_module.main([])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm_batch.cli'`

- [ ] **Step 3: Write the implementation**

`llm_batch/cli.py`:

```python
"""`llm-batch` command-line entry point."""

from __future__ import annotations

import argparse

from llm_batch.poller import Poller
from llm_batch.storage.file import FileStorage


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="llm-batch")
    sub = parser.add_subparsers(dest="command", required=True)

    poll = sub.add_parser("poll", help="poll SENT batch jobs and advance pipelines")
    poll.add_argument("state_dir", help="root directory of pipeline state")
    poll.add_argument("--sweep-orphans", type=int, default=None, metavar="DAYS",
                      help="also delete provider-side orphans older than DAYS")
    poll.add_argument("--max-batch-age-days", type=int, default=3,
                      help="mark jobs stuck in SENT past this age as FAILED")

    args = parser.parse_args(argv)

    if args.command == "poll":
        backend = FileStorage(args.state_dir)
        poller = Poller(backend, max_batch_age_days=args.max_batch_age_days)
        poller.poll_all()
        if args.sweep_orphans is not None:
            poller.sweep_orphans(older_than_days=args.sweep_orphans)
        return 0

    return 1
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cli.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add llm_batch/cli.py tests/test_cli.py
git commit -m "feat: llm-batch poll CLI with --sweep-orphans and --max-batch-age-days"
```

---

## Task 14: PostgresStorage

A JSONB-backed backend mirroring `FileStorage`'s interface: `for_job`, `for_meta`, `iter_jobs`, and per-job `load/save/get/snapshot/finish`. Namespaced so one table holds many pipelines. Driven in tests by a fake connection (no real DB).

**Files:**
- Create: `llm_batch/storage/postgres.py`
- Modify: `llm_batch/storage/__init__.py`
- Test: `tests/test_storage_postgres.py`

- [ ] **Step 1: Write the failing test**

`tests/test_storage_postgres.py`:

```python
import json

import pytest

from llm_batch.storage.base import Storage, StorageScanner
from llm_batch.storage.postgres import PostgresStorage


class FakeCursor:
    """Minimal cursor over an in-memory dict table keyed by (namespace, key)."""

    def __init__(self, store):
        self._store = store
        self._result = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=()):
        sql_norm = " ".join(sql.split()).lower()
        if sql_norm.startswith("create table"):
            self._result = None
        elif sql_norm.startswith("insert"):
            namespace, key, data_json, state = params
            self._store[(namespace, key)] = (data_json, state)
            self._result = None
        elif "where namespace = %s and key = %s" in sql_norm and sql_norm.startswith("select"):
            namespace, key = params
            row = self._store.get((namespace, key))
            self._result = [(row[0],)] if row else []
        elif "key <> %s" in sql_norm and sql_norm.startswith("select"):
            namespace, exclude = params
            self._result = [(d,) for (ns, k), (d, s) in self._store.items()
                            if ns == namespace and k != exclude]
        elif sql_norm.startswith("delete"):
            namespace, key = params
            self._store.pop((namespace, key), None)
            self._result = None

    def fetchone(self):
        return self._result[0] if self._result else None

    def fetchall(self):
        return list(self._result or [])


class FakeConn:
    def __init__(self):
        self.store = {}
        self.committed = 0

    def cursor(self):
        return FakeCursor(self.store)

    def commit(self):
        self.committed += 1


@pytest.fixture
def conn():
    return FakeConn()


def test_for_job_round_trip(conn):
    backend = PostgresStorage(conn, namespace="doc1")
    handle = backend.for_job("ocr")
    assert handle.load() is False
    handle.save({"key": "ocr", "state": "sent", "model": "gpt-4o"})

    fresh = PostgresStorage(conn, namespace="doc1").for_job("ocr")
    assert fresh.load() is True
    assert fresh.get("state") == "sent"
    assert fresh.snapshot()["model"] == "gpt-4o"


def test_get_default(conn):
    handle = PostgresStorage(conn, namespace="d").for_job("missing")
    handle.load()
    assert handle.get("state", "not_sent") == "not_sent"


def test_finish_deletes_row(conn):
    handle = PostgresStorage(conn, namespace="d").for_job("j")
    handle.save({"key": "j", "state": "completed"})
    handle.finish()
    assert PostgresStorage(conn, namespace="d").for_job("j").load() is False


def test_iter_jobs_scoped_to_namespace(conn):
    PostgresStorage(conn, namespace="doc1").for_job("a").save({"key": "a", "state": "sent"})
    PostgresStorage(conn, namespace="doc1").for_job("b").save({"key": "b", "state": "completed"})
    PostgresStorage(conn, namespace="doc2").for_job("a").save({"key": "a", "state": "sent"})

    handles = list(PostgresStorage(conn, namespace="doc1").iter_jobs())
    assert sorted(h.get("state") for h in handles) == ["completed", "sent"]
    assert isinstance(handles[0], Storage)


def test_meta_uses_reserved_key(conn):
    backend = PostgresStorage(conn, namespace="doc1")
    backend.for_meta().save({"pipeline_started_at": "t"})
    # meta is NOT returned by iter_jobs
    assert list(backend.iter_jobs()) == []
    assert backend.for_meta().load() is True


def test_is_scanner(conn):
    assert isinstance(PostgresStorage(conn, namespace="d"), StorageScanner)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_storage_postgres.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm_batch.storage.postgres'`

- [ ] **Step 3: Write the implementation**

`llm_batch/storage/postgres.py`:

```python
"""PostgresStorage — JSONB-backed per-job state, namespaced for many pipelines.

Required schema (created on demand by `ensure_schema`):

    CREATE TABLE IF NOT EXISTS llm_batch_jobs (
        namespace  TEXT NOT NULL,
        key        TEXT NOT NULL,
        data       JSONB NOT NULL,
        state      TEXT,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (namespace, key)
    );
"""

from __future__ import annotations

import json
from typing import Iterator

TABLE = "llm_batch_jobs"
# Reserved key for the pipeline meta row, kept out of iter_jobs().
_META_KEY = "__pipeline_meta__"


class _PgJobHandle:
    def __init__(self, conn, namespace: str, key: str):
        self._conn = conn
        self._namespace = namespace
        self._key = key
        self._data: dict = {}

    def load(self) -> bool:
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT data FROM {TABLE} WHERE namespace = %s AND key = %s",
                (self._namespace, self._key))
            row = cur.fetchone()
        if not row:
            return False
        raw = row[0]
        self._data = raw if isinstance(raw, dict) else json.loads(raw)
        return True

    def save(self, data: dict) -> None:
        self._data = data
        payload = json.dumps(data, sort_keys=True)
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {TABLE} (namespace, key, data, state)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (namespace, key) DO UPDATE SET
                    data = EXCLUDED.data,
                    state = EXCLUDED.state,
                    updated_at = NOW()
                """,
                (self._namespace, self._key, payload, data.get("state")))
        self._conn.commit()

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def snapshot(self) -> dict:
        return dict(self._data)

    def finish(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {TABLE} WHERE namespace = %s AND key = %s",
                (self._namespace, self._key))
        self._conn.commit()


class PostgresStorage:
    """Namespace-scoped backend that mints per-job handles over one table."""

    def __init__(self, conn, namespace: str):
        self._conn = conn
        self._namespace = namespace

    def ensure_schema(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {TABLE} (
                    namespace  TEXT NOT NULL,
                    key        TEXT NOT NULL,
                    data       JSONB NOT NULL,
                    state      TEXT,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (namespace, key)
                )
                """)
        self._conn.commit()

    def for_job(self, key: str) -> _PgJobHandle:
        return _PgJobHandle(self._conn, self._namespace, key)

    def for_meta(self) -> _PgJobHandle:
        return _PgJobHandle(self._conn, self._namespace, _META_KEY)

    def iter_jobs(self) -> Iterator[_PgJobHandle]:
        # Exclude the reserved meta row at the SQL level so iter_jobs only ever
        # yields real jobs.
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT data FROM {TABLE} WHERE namespace = %s AND key <> %s",
                (self._namespace, _META_KEY))
            rows = cur.fetchall()
        for (raw,) in rows:
            data = raw if isinstance(raw, dict) else json.loads(raw)
            handle = _PgJobHandle(self._conn, self._namespace, data.get("key"))
            handle._data = data
            yield handle
```

> The exclusion is done with `key <> %s` against the reserved `_META_KEY`. The test's `FakeCursor` has a matching branch (`"key <> %s" in sql_norm`) that filters the meta row out, so `test_meta_uses_reserved_key` sees an empty `iter_jobs()`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_storage_postgres.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Re-export from the storage package**

Modify `llm_batch/storage/__init__.py`:

```python
"""Storage backends for llm_batch."""

from llm_batch.storage.file import FileStorage
from llm_batch.storage.postgres import PostgresStorage

__all__ = ["FileStorage", "PostgresStorage"]
```

- [ ] **Step 6: Commit**

```bash
git add llm_batch/storage/postgres.py llm_batch/storage/__init__.py tests/test_storage_postgres.py
git commit -m "feat: PostgresStorage JSONB backend, namespaced, scanner-compatible"
```

---

## Task 15: Public exports, end-to-end test, and README

Wire up `llm_batch/__init__.py`, add a full-stack integration test that drives the real `providers` layer with only litellm mocked, and write the README.

**Files:**
- Modify: `llm_batch/__init__.py`
- Create: `tests/test_end_to_end.py`
- Create: `README.md`

- [ ] **Step 1: Write the failing test**

`tests/test_end_to_end.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_end_to_end.py -v`
Expected: FAIL with `ImportError: cannot import name 'Pipeline' from 'llm_batch'`

- [ ] **Step 3: Write the public exports**

`llm_batch/__init__.py`:

```python
"""llm_batch — transparent, resumable interactive/batch LLM processing.

Public API:
    Job, Pipeline, Poller            — core
    FileStorage, PostgresStorage     — storage backends
    BatchJobFailed, TransientProviderError  — errors
"""

from llm_batch.errors import BatchJobFailed, TransientProviderError
from llm_batch.job import Job
from llm_batch.pipeline import Pipeline
from llm_batch.poller import Poller
from llm_batch.storage.file import FileStorage
from llm_batch.storage.postgres import PostgresStorage

__all__ = [
    "Job",
    "Pipeline",
    "Poller",
    "FileStorage",
    "PostgresStorage",
    "BatchJobFailed",
    "TransientProviderError",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_end_to_end.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Run the full suite**

Run: `pytest -v`
Expected: PASS (all tests across all modules)

- [ ] **Step 6: Write `README.md`**

````markdown
# llm-batch

Transparent, resumable **interactive ↔ batch** LLM processing on top of
[litellm](https://github.com/BerriAI/litellm). Flip one parameter (`mode`) and
the same call site runs immediately or as a ~50%-cheaper batch job — with
persistent, cron-resumable state and a deadline-aware fallback.

## Why

litellm exposes batch and interactive primitives but does not make them
interchangeable behind one call site, nor does it persist state for resumable,
cron-driven pipelines. `llm-batch` does both.

## Install

```bash
pip install llm-batch              # core (litellm)
pip install "llm-batch[postgres]"  # + PostgresStorage backend
```

## Quick start

```python
from llm_batch import Pipeline

pipeline = Pipeline(
    state_dir="./state/doc_abc/",
    mode="batch",                       # or "interactive"
    deadline="2026-06-11T10:00:00Z",    # batch falls back to interactive near the deadline
    rerun_cmd="python continue.py doc_abc",
)

# Step 1 — a bundle of requests advancing as one resumable unit.
ocr = pipeline.job("ocr", requests=[msgs_p1, msgs_p2], model="vertex_ai/gemini-2.5-flash")
pages = ocr.execute()                   # list[str] | None  (None while the batch runs)
if pages is None:
    raise SystemExit(1)                 # re-run later; poller advances it

# Step 2 — a single request.
analysis = pipeline.job("analysis", messages=build_msgs(pages), model="vertex_ai/gemini-2.5-flash")
result = analysis.execute()             # str | None
if result is None:
    raise SystemExit(1)

print(pipeline.cost_summary())          # {'charged_usd': ..., 'saved_usd': ...}
pipeline.finish()                       # delete local state files
```

`execute()` is idempotent: re-running submits what is unsent, polls what is
sent, and returns the stored result once complete. Exit codes for cron:
`0` done, `1` pending, `2` failed.

## Modes and the deadline fallback

- `mode="interactive"` — always call the model now (full price, instant).
- `mode="batch"` — submit as batch **unless** `deadline − now < max_batch_turnaround_hours`
  (default 24h), in which case it transparently runs interactively so the hard
  deadline is still met. Pass `time_budget_hours=` as sugar for a relative deadline.

## Cron / polling

Two patterns, one core:

1. **Self-resuming script** — your pipeline script is the cron line; each tick
   polls and advances.
2. **Central poller** — for fleets:

   ```bash
   llm-batch poll ./state                      # poll all SENT jobs, advance via rerun_cmd
   llm-batch poll ./state --sweep-orphans 3    # also delete provider orphans >3 days old
   ```

The library is **filesystem-only and never touches git** — you own
`git pull` / `commit` / `push` once per run. State files use deterministic JSON
(sorted keys, trailing newline) for clean diffs.

## Storage backends

- `FileStorage(dir)` — one atomic JSON file per job (default).
- `PostgresStorage(conn, namespace)` — JSONB rows for callers already on a DB.

## Provider-side cleanup

After a result is durably saved, provider-side batch artifacts are deleted by
default (`cleanup_provider_files=True`): OpenAI file objects (quota) and Vertex
GCS objects (billing). Gemini-API and Anthropic auto-expire (no-op).

See `examples/government_contracts/` for a full GitHub Actions deployment.
````

- [ ] **Step 7: Commit**

```bash
git add llm_batch/__init__.py tests/test_end_to_end.py README.md
git commit -m "feat: public exports, end-to-end integration test, and README"
```

---

## Self-Review (completed during planning)

**1. Spec coverage** — every spec section maps to a task:

| Spec § | Requirement | Task |
|---|---|---|
| §3.1–3.2 | `Job` + `execute()` state machine | 8 |
| §3.3 | single (`messages=`) vs bundle (`requests=`) shapes | 8 |
| §3.4 | `Pipeline`, `pipeline.json`, namespaced `job()`, `finish()` | 11 |
| §4 | two modes + conservative deadline fallback | 9 |
| §5.1 | `Storage` / `StorageScanner` protocols | 2 |
| §5.2 | `FileStorage` atomic + deterministic JSON | 3 |
| §5.3 | `PostgresStorage` | 14 |
| §5.4 | full per-job state schema (+ `bundle` flag) | 8 |
| §6.2 | `Poller` + `rerun_cmd` de-dup dispatch | 12 |
| §7.1 | completion cleanup, persist-before-delete | 8 |
| §7.2 | stale-job timeout → FAILED + cleanup | 12 |
| §7.3 | `sweep_orphans` | 12 |
| §8.1 | transient vs terminal classification | 6, 8 |
| §8.2 | sticky `FAILED`, `retry()`, exit codes | 8, 10 |
| §9 | provider layer (submit/poll/complete/cleanup/provider_of) | 4–7 |
| §9 | usage/cost capture, structured-output kwarg passthrough | 5, 6 |
| §11 | CLI | 13 |
| §12 | module structure | all |
| §13 | TDD with litellm mocked | all (conftest in 0) |
| §14 | `set_completed()` | 10 |

**2. Placeholder scan** — no "TBD"/"implement later"/"add error handling"; every code step contains complete code. Confirmed.

**3. Type consistency** — `BatchHandle`/`PollResult`/`CompletionResult` field names are used identically in `providers.py` (Tasks 4–7) and `job.py` (Task 8). `Storage` handle methods (`load/save/get/snapshot/finish`) match across `FileStorage` (3), `PostgresStorage` (14), and their consumers (`Job`, `Pipeline`, `Poller`). `provider_of` / `cleanup` / `cleanup_orphans` signatures match between `providers.py` and `poller.py`. Job state constants (`NOT_SENT/SENT/COMPLETED/FAILED`) are referenced consistently.

**Out of scope (correctly absent):** no `litellm.cancel_batch`, no speculative cancellation — deferred to Phase 2 per `docs/todo/2026-06-08-canceling-batch-jobs.md`.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-08-llm-interactive-batch-lib-plan.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
