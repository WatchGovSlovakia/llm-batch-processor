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
