import json

import pytest

from llm_batch import providers
from llm_batch.providers import BatchHandle, CompletionResult, PollResult
from tests.conftest import completion_response, make_obj


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
    class APITimeoutError(Exception):
        pass
    def boom2(**k):
        raise APITimeoutError("slow")
    litellm_mock.completion = boom2
    with pytest.raises(providers.TransientProviderError):
        providers.complete("gpt-4o", [{"role": "user", "content": "q"}])


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


def test_poll_batch_missing_bundle_result_is_terminal(litellm_mock):
    # provider reports completed but returns fewer results than submitted
    handle = BatchHandle(provider="openai", model="gpt-4o", batch_job_id="b1",
                         input_file_id="f-in", n_requests=3)
    litellm_mock.retrieve_batch = lambda **k: make_obj(status="completed", output_file_id="files/out-1")
    litellm_mock.file_content = lambda **k: batch_output_jsonl(["only one of three"])
    litellm_mock.cost_per_token = lambda **k: (0.0, 0.0)
    result = providers.poll_batch(handle)
    assert result.status == "failed"
    assert "missing" in result.error_message.lower()


def test_poll_batch_complete_bundle_with_expected_count(litellm_mock):
    handle = BatchHandle(provider="openai", model="gpt-4o", batch_job_id="b1",
                         input_file_id="f-in", n_requests=2)
    litellm_mock.retrieve_batch = lambda **k: make_obj(status="completed", output_file_id="files/out-1")
    litellm_mock.file_content = lambda **k: batch_output_jsonl(["a", "b"])
    litellm_mock.cost_per_token = lambda **k: (0.0, 0.0)
    result = providers.poll_batch(handle)
    assert result.status == "completed"
    assert result.texts == ["a", "b"]


def test_provider_supports_cancel():
    from llm_batch import providers
    assert providers.provider_supports_cancel("openai") is True
    assert providers.provider_supports_cancel("azure") is True
    assert providers.provider_supports_cancel("vertex_ai") is True
    assert providers.provider_supports_cancel("anthropic") is False
    assert providers.provider_supports_cancel("gemini") is False


def test_poll_batch_cancelled_is_distinct_status(litellm_mock):
    from llm_batch import providers
    from llm_batch.providers import BatchHandle
    litellm_mock.retrieve_batch = lambda **k: make_obj(status="cancelled")
    handle = BatchHandle(provider="openai", model="gpt-4o", batch_job_id="b1")
    result = providers.poll_batch(handle)
    assert result.status == "canceled"


def test_poll_batch_cancelling_is_pending(litellm_mock):
    from llm_batch import providers
    from llm_batch.providers import BatchHandle
    litellm_mock.retrieve_batch = lambda **k: make_obj(status="cancelling")
    handle = BatchHandle(provider="openai", model="gpt-4o", batch_job_id="b1")
    result = providers.poll_batch(handle)
    assert result.status == "pending"


def test_cancel_delegates_to_litellm_with_provider(litellm_mock):
    from llm_batch import providers
    from llm_batch.providers import BatchHandle
    seen = {}

    def cancel_batch(*, batch_id, custom_llm_provider, **k):
        seen["batch_id"] = batch_id
        seen["provider"] = custom_llm_provider
        return make_obj(id=batch_id, status="cancelling")

    litellm_mock.cancel_batch = cancel_batch
    handle = BatchHandle(provider="vertex_ai", model="vertex_ai/gemini-1.5-pro", batch_job_id="b9")
    assert providers.cancel(handle) is None
    assert seen == {"batch_id": "b9", "provider": "vertex_ai"}


def test_cancel_wraps_transient_errors(litellm_mock):
    from llm_batch import providers

    class Boom(Exception):
        status_code = 503

    def cancel_batch(**k):
        raise Boom("service unavailable")

    litellm_mock.cancel_batch = cancel_batch
    handle = providers.BatchHandle(provider="openai", model="gpt-4o", batch_job_id="b1")
    with pytest.raises(providers.TransientProviderError):
        providers.cancel(handle)


def test_dataclasses_finish_reason_defaults_to_none():
    pr = PollResult(status="pending")
    assert pr.finish_reason is None
    cr = CompletionResult(text="hi", usage={}, cost={})
    assert cr.finish_reason is None


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
    litellm_mock.retrieve_batch = lambda **k: make_obj(
        status="completed", output_file_id="files/out-1")
    litellm_mock.file_content = lambda **k: batch_output_jsonl(
        ["p1", "p2"], finish_reasons=["stop", "length"])
    handle = BatchHandle(provider="openai", model="gpt-4o",
                          batch_job_id="b1", n_requests=2)
    result = providers.poll_batch(handle)
    assert result.finish_reason is None  # bundle: scalar not populated


def test_poll_batch_completed_missing_finish_reason_key_is_none(litellm_mock):
    litellm_mock.retrieve_batch = lambda **k: make_obj(
        status="completed", output_file_id="files/out-1")
    litellm_mock.file_content = lambda **k: batch_output_jsonl(["the text"])
    handle = BatchHandle(provider="openai", model="gpt-4o",
                          batch_job_id="b1", n_requests=1)
    result = providers.poll_batch(handle)
    assert result.finish_reason is None


# --- per-request items, cached tokens, batch cache pricing --------------------------

def test_cached_tokens_from_object_and_dict():
    assert providers._cached_tokens(None) == 0
    assert providers._cached_tokens({"prompt_tokens": 5}) == 0
    assert providers._cached_tokens({"prompt_tokens_details": {"cached_tokens": 7}}) == 7
    obj = make_obj(prompt_tokens=5, prompt_tokens_details=make_obj(cached_tokens=3))
    assert providers._cached_tokens(obj) == 3
    assert providers._cached_tokens(make_obj(prompt_tokens_details=None)) == 0


def test_complete_reports_cached_tokens(litellm_mock):
    usage = make_obj(prompt_tokens=100, completion_tokens=10, total_tokens=110,
                     prompt_tokens_details=make_obj(cached_tokens=60))
    litellm_mock.completion = lambda **k: completion_response("hi", usage=usage, finish_reason="stop")
    cr = providers.complete("gpt-4o", [{"role": "user", "content": "x"}])
    assert cr.cached_tokens == 60
    assert cr.usage == {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}
    assert providers.completion_item(cr) == {
        "text": "hi", "finish_reason": "stop",
        "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110, "cached_tokens": 60},
        "cost": cr.cost}


def test_poll_batch_bundle_has_per_request_items(litellm_mock):
    usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
             "prompt_tokens_details": {"cached_tokens": 4}}
    litellm_mock.retrieve_batch = lambda **k: make_obj(status="completed", output_file_id="files/out")
    litellm_mock.file_content = lambda **k: batch_output_jsonl(
        ["a", "b"], usage=usage, finish_reasons=["stop", "length"])
    res = providers.poll_batch(BatchHandle(provider="openai", model="gpt-4o",
                                           batch_job_id="b1", n_requests=2))
    assert res.status == "completed" and res.finish_reason is None
    assert [i["text"] for i in res.items] == ["a", "b"]
    assert [i["finish_reason"] for i in res.items] == ["stop", "length"]
    assert res.items[1]["usage"] == {"prompt_tokens": 10, "completion_tokens": 5,
                                     "total_tokens": 15, "cached_tokens": 4}
    assert res.usage == {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}


def _fake_prices(**k):
    """prompt: uncached 1.0/token, cached 0.1/token; completion 2.0/token."""
    cached = k.get("cache_read_input_tokens", 0)
    return ((k["prompt_tokens"] - cached) * 1.0 + cached * 0.1, k["completion_tokens"] * 2.0)


@pytest.mark.parametrize("policy,charged", [
    ("batch_only", 60.0),        # 0.5 * (100 + 20)
    ("stack", 33.0),             # 0.5 * (40 + 6 + 20)
    ("cache_precedence", 36.0),  # 6 + 0.5 * (40 + 20)
])
def test_batch_cost_policy_for_cached_tokens(litellm_mock, monkeypatch, policy, charged):
    litellm_mock.cost_per_token = _fake_prices
    monkeypatch.setattr(providers, "BATCH_CACHE_POLICY", policy)
    usage = {"prompt_tokens": 100, "completion_tokens": 10, "cached_tokens": 60}
    cost = providers._cost_dict("m", usage, batch=True)
    assert cost["interactive_usd"] == pytest.approx(66.0)
    assert cost["charged_usd"] == pytest.approx(charged)
    assert cost["saved_usd"] == pytest.approx(66.0 - charged)


def test_batch_cost_without_cached_tokens_is_unchanged(litellm_mock, monkeypatch):
    litellm_mock.cost_per_token = _fake_prices
    for policy in ("batch_only", "stack", "cache_precedence"):
        monkeypatch.setattr(providers, "BATCH_CACHE_POLICY", policy)
        cost = providers._cost_dict("m", {"prompt_tokens": 100, "completion_tokens": 10}, batch=True)
        assert cost["charged_usd"] == pytest.approx(60.0)


# --- provider-reported cost -------------------------------------------------

def _usage_with_cost(cost):
    return {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105, "cost": cost}


def test_complete_prefers_provider_reported_cost(litellm_mock):
    litellm_mock.completion = lambda **k: completion_response("x", _usage_with_cost(1.165e-05))
    litellm_mock.cost_per_token = lambda **k: (0.0, 0.0)  # unknown to litellm
    cost = providers.complete("openrouter/qwen/qwen3.8-flash", [{"role": "user", "content": "q"}]).cost
    assert cost["charged_usd"] == pytest.approx(1.165e-05)
    assert cost["saved_usd"] == 0.0
    assert cost["cost_source"] == "provider"


def test_complete_provider_cost_beats_litellm_price(litellm_mock):
    litellm_mock.completion = lambda **k: completion_response("x", _usage_with_cost(0.5))
    litellm_mock.cost_per_token = lambda **k: (0.01, 0.02)
    cost = providers.complete("openrouter/m", [{"role": "user", "content": "q"}]).cost
    assert cost["charged_usd"] == pytest.approx(0.5)
    assert cost["cost_source"] == "provider"


def test_complete_without_provider_cost_uses_litellm(litellm_mock):
    litellm_mock.completion = lambda **k: completion_response("x", {
        "prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})
    litellm_mock.cost_per_token = lambda **k: (0.01, 0.02)
    cost = providers.complete("gpt-4o", [{"role": "user", "content": "q"}]).cost
    assert cost["charged_usd"] == pytest.approx(0.03)
    assert cost["cost_source"] == "litellm"


def test_zero_price_with_tokens_is_unknown_and_warns(litellm_mock, caplog):
    litellm_mock.completion = lambda **k: completion_response("x", {
        "prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105})
    litellm_mock.cost_per_token = lambda **k: (0.0, 0.0)
    with caplog.at_level("WARNING"):
        cost = providers.complete("openrouter/unknown/model", [{"role": "user", "content": "q"}]).cost
    assert cost["cost_source"] == "unknown"
    assert cost["charged_usd"] == 0.0
    assert "unknown/model" in caplog.text


def test_zero_tokens_zero_price_is_not_unknown(litellm_mock):
    cost = providers._cost_dict("m", {"prompt_tokens": 0, "completion_tokens": 0}, batch=False)
    assert cost["cost_source"] == "litellm"


def test_batch_cost_dict_has_cost_source(litellm_mock):
    litellm_mock.cost_per_token = lambda **k: (0.01, 0.02)
    cost = providers._cost_dict("m", {"prompt_tokens": 10, "completion_tokens": 1}, batch=True)
    assert cost["cost_source"] == "litellm"
