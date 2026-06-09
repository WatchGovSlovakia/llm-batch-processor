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
