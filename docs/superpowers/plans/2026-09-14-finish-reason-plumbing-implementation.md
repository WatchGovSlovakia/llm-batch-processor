# finish_reason plumbing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface the provider's `finish_reason` through `providers.PollResult`/
`CompletionResult` and `Job`, and add regression coverage proving the library's existing
generic `**litellm_kwargs` passthrough (`max_tokens`, `thinking`, etc.) already reaches the
provider request body end-to-end.

**Architecture:** `finish_reason` follows the exact same shape as the existing `.usage`/
`.cost` fields: a value on the provider-layer result dataclasses (`PollResult`,
`CompletionResult`), persisted into `Job._state` on every completion path, exposed via a
`Job.finish_reason` property. No new kwargs, no new Job states, no behavior change — this is
additive visibility into a signal the provider already returns.

**Tech Stack:** Python, pytest, dataclasses, litellm (via the `providers` module boundary).

**Spec:** `/llm-batch-processor/docs/superpowers/specs/2026-09-14-finish-reason-plumbing-design.md`

**Consumed by:** pdfmux's own implementation plan (`pdfmux/docs/superpowers/plans/2026-09-14-llm-batch-output-cap-implementation.md`) depends on `Job.finish_reason` existing — complete this plan first.

## Global Constraints

- `PollResult.finish_reason` is a scalar: populated for a single-request batch job (the only
  usage shape any known caller has today); `None` for any multi-request/bundle response
  (spec §2).
- This library must stay provider-agnostic — no consumer-specific values or behavior belong
  here (spec Summary).
- All new/changed tests follow RED→GREEN TDD (spec §3).

---

### Task 1: `finish_reason` on `PollResult` and `CompletionResult`, plus conftest helper support

**Files:**
- Modify: `/llm-batch-processor/llm_batch/providers.py:51-70` (the `PollResult`/`CompletionResult` dataclasses)
- Modify: `/llm-batch-processor/tests/conftest.py:35-67` (`completion_response`, `batch_output_jsonl` helpers)
- Test: `/llm-batch-processor/tests/test_providers.py`

**Interfaces:**
- Produces: `PollResult(..., finish_reason: str | None = None)`, `CompletionResult(..., finish_reason: str | None = None)` — both keyword-only in practice (every existing call site already uses kwargs).
- Produces: `conftest.completion_response(text, usage=None, finish_reason=None)` — sets `choice.finish_reason`.
- Produces: `conftest.batch_output_jsonl(texts, usage=None, finish_reasons=None)` — `finish_reasons` is an optional list aligned to `texts`; when omitted, each line's body has no `finish_reason` key (so `.get()` in production code returns `None`, exercising the "absent" path).

- [ ] **Step 1: Write the failing tests**

Append to `/llm-batch-processor/tests/test_providers.py`:

```python
def test_dataclasses_finish_reason_defaults_to_none():
    pr = PollResult(status="pending")
    assert pr.finish_reason is None
    cr = CompletionResult(text="hi", usage={}, cost={})
    assert cr.finish_reason is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /llm-batch-processor && python -m pytest tests/test_providers.py::test_dataclasses_finish_reason_defaults_to_none -v`
Expected: FAIL with `AttributeError: 'PollResult' object has no attribute 'finish_reason'`.

- [ ] **Step 3: Add the fields**

In `/llm-batch-processor/llm_batch/providers.py`, edit the two dataclasses:

```python
@dataclass
class PollResult:
    """Outcome of one poll of a batch job."""

    status: str  # "pending" | "completed" | "failed" | "transient_error" | "canceled"
    texts: list[str] | None = None
    usage: dict | None = None
    cost: dict | None = None
    output_file_id: str | None = None
    error_message: str | None = None
    # Scalar: populated for a single-request batch job; None for a bundle
    # response until/unless a future consumer needs per-item values.
    finish_reason: str | None = None


@dataclass
class CompletionResult:
    """Outcome of one interactive completion."""

    text: str
    usage: dict
    cost: dict
    finish_reason: str | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /llm-batch-processor && python -m pytest tests/test_providers.py::test_dataclasses_finish_reason_defaults_to_none -v`
Expected: PASS

- [ ] **Step 5: Extend the conftest helpers (no test needed yet — exercised by Task 2)**

In `/llm-batch-processor/tests/conftest.py`, replace `completion_response` and `batch_output_jsonl`:

```python
def batch_output_jsonl(texts, usage=None, finish_reasons=None):
    """Build the raw JSONL body that litellm.file_content returns for a completed batch.

    `texts` is a list aligned to custom_id req-0..req-N. `finish_reasons`, if
    given, is a list aligned to `texts`; omitted entries mean no
    "finish_reason" key in that line's body (production code must treat a
    missing key the same as an explicit None).
    """
    usage = usage or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    lines = []
    for i, text in enumerate(texts):
        choice: dict = {"message": {"content": text}}
        if finish_reasons is not None and finish_reasons[i] is not None:
            choice["finish_reason"] = finish_reasons[i]
        lines.append(json.dumps({
            "custom_id": f"req-{i}",
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [choice],
                    "usage": usage,
                },
            },
        }))
    body = "\n".join(lines) + "\n"
    return make_obj(text=body, content=body.encode())


def completion_response(text, usage=None, finish_reason=None):
    """Build a stand-in for a litellm.completion(...) response."""
    usage = usage or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    message = make_obj(content=text)
    choice = make_obj(message=message, finish_reason=finish_reason)
    return make_obj(choices=[choice], usage=usage)
```

- [ ] **Step 6: Run the full providers + job test suites to confirm no regression**

Run: `cd /llm-batch-processor && python -m pytest tests/test_providers.py tests/test_job.py -v`
Expected: PASS (existing tests don't pass `finish_reason`/`finish_reasons`, so they exercise the default-`None` path unchanged)

- [ ] **Step 7: Commit**

```bash
cd /llm-batch-processor
git add llm_batch/providers.py tests/conftest.py tests/test_providers.py
git commit -m "feat: add finish_reason field to PollResult/CompletionResult"
```

---

### Task 2: Populate `finish_reason` in `providers.complete()` and `providers.poll_batch()`

**Files:**
- Modify: `/llm-batch-processor/llm_batch/providers.py:196-207` (`complete()`), `:239-324` (`poll_batch()`)
- Test: `/llm-batch-processor/tests/test_providers.py`

**Interfaces:**
- Consumes: `PollResult`/`CompletionResult.finish_reason` (Task 1), `conftest.completion_response(..., finish_reason=...)`, `conftest.batch_output_jsonl(..., finish_reasons=...)`.
- Produces: `complete()` and `poll_batch()` now set `finish_reason` on their returned dataclass instead of leaving the default.

- [ ] **Step 1: Write the failing tests**

Append to `/llm-batch-processor/tests/test_providers.py`:

```python
def test_complete_returns_finish_reason(litellm_mock):
    litellm_mock.completion = lambda **k: completion_response(
        "the answer", finish_reason="stop")
    litellm_mock.cost_per_token = lambda **k: (0.0, 0.0)
    result = providers.complete("gpt-4o", [{"role": "user", "content": "q"}])
    assert result.finish_reason == "stop"


def test_complete_finish_reason_none_when_absent(litellm_mock):
    litellm_mock.completion = lambda **k: completion_response("the answer")
    litellm_mock.cost_per_token = lambda **k: (0.0, 0.0)
    result = providers.complete("gpt-4o", [{"role": "user", "content": "q"}])
    assert result.finish_reason is None


def test_poll_batch_completed_single_request_returns_finish_reason(litellm_mock):
    from tests.conftest import batch_output_jsonl

    litellm_mock.retrieve_batch = lambda **k: make_obj(
        status="completed", output_file_id="files/out-1")
    litellm_mock.file_content = lambda **k: batch_output_jsonl(
        ["the text"], finish_reasons=["length"])
    handle = BatchHandle(provider="openai", model="gpt-4o",
                          batch_job_id="b1", n_requests=1)
    result = providers.poll_batch(handle)
    assert result.status == "completed"
    assert result.finish_reason == "length"


def test_poll_batch_completed_bundle_returns_none_finish_reason(litellm_mock):
    from tests.conftest import batch_output_jsonl

    litellm_mock.retrieve_batch = lambda **k: make_obj(
        status="completed", output_file_id="files/out-1")
    litellm_mock.file_content = lambda **k: batch_output_jsonl(
        ["p1", "p2"], finish_reasons=["stop", "length"])
    handle = BatchHandle(provider="openai", model="gpt-4o",
                          batch_job_id="b1", n_requests=2)
    result = providers.poll_batch(handle)
    assert result.finish_reason is None  # bundle: scalar not populated


def test_poll_batch_completed_missing_finish_reason_key_is_none(litellm_mock):
    from tests.conftest import batch_output_jsonl

    litellm_mock.retrieve_batch = lambda **k: make_obj(
        status="completed", output_file_id="files/out-1")
    litellm_mock.file_content = lambda **k: batch_output_jsonl(["the text"])
    handle = BatchHandle(provider="openai", model="gpt-4o",
                          batch_job_id="b1", n_requests=1)
    result = providers.poll_batch(handle)
    assert result.finish_reason is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /llm-batch-processor && python -m pytest tests/test_providers.py -k finish_reason -v`
Expected: FAIL — `complete()`/`poll_batch()` don't set `finish_reason` yet, so it stays `None` even for `test_complete_returns_finish_reason` and `test_poll_batch_completed_single_request_returns_finish_reason`.

- [ ] **Step 3: Implement in `complete()`**

In `/llm-batch-processor/llm_batch/providers.py`, edit `complete()`:

```python
def complete(model: str, messages: list, **kw) -> CompletionResult:
    """Interactive path: one immediate completion with usage + cost."""
    try:
        response = litellm.completion(model=model, messages=messages, **kw)
    except Exception as exc:
        if _is_transient(exc):
            raise TransientProviderError(str(exc)) from exc
        raise
    choice = response.choices[0]
    text = choice.message.content
    usage = _usage_dict(response.usage)
    cost = _cost_dict(model, usage, batch=False)
    finish_reason = getattr(choice, "finish_reason", None)
    return CompletionResult(text=text, usage=usage, cost=cost, finish_reason=finish_reason)
```

- [ ] **Step 4: Implement in `poll_batch()`**

In `/llm-batch-processor/llm_batch/providers.py`, edit the request-line loop and the final `PollResult` construction inside `poll_batch()`:

```python
    raw = content.text if hasattr(content, "text") else content.content.decode()
    results_by_id: dict[str, str] = {}
    finish_reasons_by_id: dict[str, str | None] = {}
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
        finish_reasons_by_id[custom_id] = body["choices"][0].get("finish_reason")
        usage = _usage_dict(body.get("usage"))
        total["prompt_tokens"] += usage["prompt_tokens"]
        total["completion_tokens"] += usage["completion_tokens"]
        total["total_tokens"] += usage["total_tokens"]

    # Verify completeness: a "completed" batch must yield every submitted
    # request (spec §5.4 — n_requests lets us confirm we got every result).
    expected = handle.n_requests if handle.n_requests is not None else len(results_by_id)
    texts = []
    for i in range(expected):
        custom_id = f"req-{i}"
        if custom_id not in results_by_id:
            return PollResult(
                status="failed", output_file_id=output_file_id,
                error_message=(f"missing result for {custom_id} "
                               f"({len(results_by_id)}/{expected} returned)"))
        texts.append(results_by_id[custom_id])
    cost = _cost_dict(handle.model, total, batch=True)
    # Scalar finish_reason only makes sense for a single-request batch
    # (the only usage shape any known caller has today); a multi-request/
    # bundle response leaves it None (spec §2 — PollResult.finish_reason).
    finish_reason = finish_reasons_by_id.get("req-0") if len(texts) == 1 else None
    return PollResult(
        status="completed", texts=texts, usage=total, cost=cost,
        output_file_id=output_file_id, finish_reason=finish_reason)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /llm-batch-processor && python -m pytest tests/test_providers.py -v`
Expected: PASS (all, including the pre-existing ones)

- [ ] **Step 6: Commit**

```bash
cd /llm-batch-processor
git add llm_batch/providers.py tests/test_providers.py
git commit -m "feat: populate finish_reason from complete() and poll_batch()"
```

---

### Task 3: `Job.finish_reason` — persistence, property, and every completion path

**Files:**
- Modify: `/llm-batch-processor/llm_batch/job.py:57-63` (`_PERSISTED_FIELDS`), `:151-157` (properties, add `finish_reason`), `:262-286` (`_base_state`), `:310-330` (`_submit_interactive`), `:385-398` (`_finish_completed`), `:473-499` (`_reissue_interactive`)
- Test: `/llm-batch-processor/tests/test_job.py`

**Interfaces:**
- Consumes: `PollResult.finish_reason`, `CompletionResult.finish_reason` (Tasks 1-2).
- Produces: `Job.finish_reason -> str | None` property, mirroring `.usage`/`.cost` exactly (reads `self._state.get('finish_reason')`, `None` when absent or job not completed).

- [ ] **Step 1: Write the failing tests**

Append to `/llm-batch-processor/tests/test_job.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /llm-batch-processor && python -m pytest tests/test_job.py -k finish_reason -v`
Expected: FAIL — `Job` has no `.finish_reason` attribute yet (`AttributeError`).

- [ ] **Step 3: Add the property, persisted field, and base-state default**

In `/llm-batch-processor/llm_batch/job.py`:

Edit `_PERSISTED_FIELDS` (around line 57):

```python
_PERSISTED_FIELDS = (
    "key", "state", "mode", "model", "n_requests", "bundle", "batch_job_id",
    "input_file_id", "output_file_id", "submitted_at", "completed_at", "deadline",
    "actual_mode", "response", "usage", "cost", "finish_reason", "error_message",
    "poll_attempts", "rerun_cmd", "cleanup_provider_files",
    "fallback_strategy", "payload", "litellm_kwargs", "cancel_requested_at",
)
```

Add the property right after `cost` (around line 157):

```python
    @property
    def finish_reason(self) -> str | None:
        return self._state.get('finish_reason')
```

Add the default to `_base_state()` (around line 262-286), alongside `"cost": dict(_ZERO_COST),`:

```python
            "cost": dict(_ZERO_COST),
            "finish_reason": None,
```

- [ ] **Step 4: Set `finish_reason` in `_finish_completed()`**

Edit `_finish_completed` (around line 385):

```python
    def _finish_completed(self, handle, result):
        response = result.texts if self._is_bundle else result.texts[0]
        self._state.update(
            state=self.COMPLETED, response=response, completed_at=_now_iso(),
            usage=result.usage, cost=result.cost, finish_reason=result.finish_reason,
            output_file_id=result.output_file_id, error_message=None)
        self._save()  # persist BEFORE provider-side cleanup (spec §7.1)
        if self._cleanup:
            handle.output_file_id = result.output_file_id
            try:
                providers.cleanup(handle)
            except Exception:
                pass  # best-effort; result already durably saved
        return response
```

- [ ] **Step 5: Set `finish_reason` in `_submit_interactive()`**

Edit `_submit_interactive` (around line 310):

```python
    def _submit_interactive(self):
        texts = []
        usage = dict(_ZERO_USAGE)
        cost = dict(_ZERO_COST)
        last_finish_reason = None
        try:
            for messages in self._payload:
                cr = providers.complete(self._model, messages, **self._kw)
                texts.append(cr.text)
                _accumulate(usage, cr.usage)
                _accumulate_cost(cost, cr.cost)
                last_finish_reason = cr.finish_reason
        except providers.TransientProviderError:
            return None  # leave state NOT_SENT; retried next run

        response = texts if self._is_bundle else texts[0]
        state = self._base_state()
        state.update(state=self.COMPLETED, actual_mode="interactive",
                     completed_at=_now_iso(), response=response,
                     usage=usage, cost=cost,
                     finish_reason=None if self._is_bundle else last_finish_reason)
        self._state = state
        self._save()
        return response
```

- [ ] **Step 6: Set `finish_reason` in `_reissue_interactive()`**

Edit `_reissue_interactive` (around line 473) the same way:

```python
    def _reissue_interactive(self, handle):
        """Re-run the persisted request interactively and mark COMPLETED."""
        texts = []
        usage = dict(_ZERO_USAGE)
        cost = dict(_ZERO_COST)
        last_finish_reason = None
        try:
            for messages in self._payload:
                cr = providers.complete(self._model, messages, **self._kw)
                texts.append(cr.text)
                _accumulate(usage, cr.usage)
                _accumulate_cost(cost, cr.cost)
                last_finish_reason = cr.finish_reason
        except providers.TransientProviderError:
            self._save()  # stay CANCEL_REQUESTED; retry interactive next poll
            return None

        response = texts if self._is_bundle else texts[0]
        self._state.update(
            state=self.COMPLETED, actual_mode="interactive",
            completed_at=_now_iso(), response=response, usage=usage, cost=cost,
            finish_reason=None if self._is_bundle else last_finish_reason,
            error_message=None)
        self._save()
        if self._cleanup:
            try:
                providers.cleanup(handle)  # remove the (cancelled) batch artifacts
            except Exception:
                pass
        return response
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `cd /llm-batch-processor && python -m pytest tests/test_job.py -v`
Expected: PASS (all, including pre-existing ones — `_base_state()`'s new `finish_reason: None` key doesn't affect any existing assertion since none of them enumerate the full state dict)

- [ ] **Step 8: Commit**

```bash
cd /llm-batch-processor
git add llm_batch/job.py tests/test_job.py
git commit -m "feat: thread finish_reason through Job's completion paths"
```

---

### Task 4: Regression test — `max_tokens`/`thinking` kwargs already reach the request body

**Files:**
- Modify: `/llm-batch-processor/tests/test_job.py:10-33` (`fake_providers` fixture)
- Test: `/llm-batch-processor/tests/test_job.py`

**Interfaces:**
- Consumes: `Job.__init__(..., **litellm_kwargs)` (existing, unchanged), `Pipeline.job(..., **litellm_kwargs)` (existing, unchanged).
- Produces: nothing new in production code — this task only adds coverage proving the existing generic passthrough carries `max_tokens`/`thinking` (the exact kwargs pdfmux's `BatchLLMExtractor` passes) all the way into the submitted/completed request body.

- [ ] **Step 1: Extend `fake_providers` to capture kwargs**

In `/llm-batch-processor/tests/test_job.py`, edit the `fake_providers` fixture:

```python
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
```

(Only the two inner functions and the initial `calls` dict changed — `submit`/`complete`/`cleanup` counters and behavior are untouched, so every existing test using this fixture keeps passing.)

- [ ] **Step 2: Write the tests**

Append to `/llm-batch-processor/tests/test_job.py`:

```python
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
```

- [ ] **Step 3: Run tests**

Run: `cd /llm-batch-processor && python -m pytest tests/test_job.py -k "max_tokens_and_thinking" -v`
Expected: PASS immediately — this confirms the existing `**litellm_kwargs` passthrough (Pipeline.job → Job.__init__ → self._kw → providers.submit_batch/complete) already does what the spec asks. No production code change in this step.

- [ ] **Step 4: Run the full llm-batch-processor test suite**

Run: `cd /llm-batch-processor && python -m pytest -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
cd /llm-batch-processor
git add tests/test_job.py
git commit -m "test: cover max_tokens/thinking kwarg passthrough end-to-end"
```

---

## Final verification

- [ ] **Run the full suite**

```bash
cd /llm-batch-processor && python -m pytest -v
```

Expected: PASS everywhere.

- [ ] **Cross-check against the spec's §3 test list**

Re-read `/llm-batch-processor/docs/superpowers/specs/2026-09-14-finish-reason-plumbing-design.md` §3 line by line and confirm each bullet has a corresponding test:
- `PollResult`/`CompletionResult` default `finish_reason` to `None` → Task 1, Step 1
- `complete()` returns/omits `finish_reason` → Task 2, Step 1
- `poll_batch()` single-request → `finish_reason` set; bundle → `None`; missing key → `None` → Task 2, Step 1
- `Job.finish_reason` None before completion → Task 3, Step 1
- `Job.finish_reason` returns stored value after completed batch poll / interactive completion → Task 3, Step 1
- `Job.finish_reason` None after a completed bundle job → Task 3, Step 1
- `Job.finish_reason` None when state has no key → Task 3, Step 1
- `max_tokens`/`thinking` kwarg reaches the request body (`Job` and `Pipeline.job`) → Task 4

- [ ] **Notify the pdfmux side this plan is complete**

pdfmux's plan (`pdfmux/docs/superpowers/plans/2026-09-14-llm-batch-output-cap-implementation.md`) depends on `Job.finish_reason` — its Task 2 (`BatchLLMExtractor` caps output + flags truncated pages) cannot proceed until this plan's Task 3 is committed.
