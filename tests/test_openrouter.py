"""OpenRouter Batch API provider: inline submit, inline results, delete, orphan sweep.

No network: `openrouter_mock` replaces providers._openrouter_http, and the HTTP
helper itself is tested against an httpx.MockTransport.
"""

import json

import httpx
import pytest

from llm_batch import providers
from llm_batch.job import Job
from llm_batch.providers import BatchHandle
from llm_batch.storage.file import FileStorage

MODEL = "openrouter/anthropic/claude-haiku-5.5"


class FakeOpenRouter:
    """Records (method, path, json, params) and answers from `routes`.

    A route value is a dict (returned as the JSON body), an Exception (raised),
    or a callable(json, params) returning either.
    """

    def __init__(self):
        self.calls = []
        self.routes = {}

    def __call__(self, method, path, *, json=None, params=None):
        self.calls.append((method, path, json, params))
        try:
            answer = self.routes[(method, path)]
        except KeyError:
            raise AssertionError(f"unexpected OpenRouter call {method} {path}")
        if callable(answer) and not isinstance(answer, Exception):
            answer = answer(json, params)
        if isinstance(answer, Exception):
            raise answer
        return answer


@pytest.fixture
def openrouter_mock(monkeypatch, litellm_mock):
    fake = FakeOpenRouter()
    monkeypatch.setattr(providers, "_openrouter_http", fake)
    return fake


def _err(status, message="boom"):
    return providers.OpenRouterAPIError(status, message)


def _result(custom_id, text, *, finish_reason="stop", usage=None):
    usage = usage or {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12, "cost": 1e-06}
    return {"id": f"batch_req_{custom_id}", "custom_id": custom_id, "error": None,
            "response": {"status_code": 200, "request_id": "r",
                         "body": {"object": "chat.completion",
                                  "choices": [{"index": 0, "finish_reason": finish_reason,
                                               "message": {"role": "assistant", "content": text}}],
                                  "usage": usage}}}


def _batch(status, results=None, error=None, usage=None):
    return {"id": "batch-1", "object": "batch", "status": status, "results": results,
            "error": error, "usage": usage, "provider": "Anthropic",
            "request_counts": {"total": len(results or []), "completed": 0, "failed": 0}}


def _handle(n=1, model=MODEL):
    return BatchHandle(provider="openrouter", model=model, batch_job_id="batch-1", n_requests=n)


# --- provider identity ----------------------------------------------------------

def test_provider_of_openrouter():
    assert providers.provider_of(MODEL) == "openrouter"


def test_openrouter_has_no_cancel():
    assert providers.provider_supports_cancel("openrouter") is False


def test_cancel_openrouter_is_not_supported():
    with pytest.raises(NotImplementedError):
        providers.cancel(_handle())


# --- submit ---------------------------------------------------------------------

def test_submit_posts_inline_requests_without_file_upload(openrouter_mock):
    openrouter_mock.routes[("POST", "/batches")] = {"id": "batch-1", "status": "validating"}

    handle = providers.submit_batch(
        MODEL, [[{"role": "user", "content": "hi"}], [{"role": "user", "content": "bye"}]],
        max_tokens=8, response_format={"type": "json_object"})

    assert handle == BatchHandle(provider="openrouter", model=MODEL, batch_job_id="batch-1",
                                 input_file_id=None, n_requests=2)
    [(method, path, body, _params)] = openrouter_mock.calls
    assert (method, path) == ("POST", "/batches")
    assert body["endpoint"] == "/v1/chat/completions"
    assert body["model"] == "anthropic/claude-haiku-5.5"  # litellm routing prefix stripped
    assert body["completion_window"] == "24h"
    # OpenRouter streams the body: `requests` must come after the other fields.
    assert list(body)[-1] == "requests"
    assert [r["custom_id"] for r in body["requests"]] == ["req-0", "req-1"]
    assert body["requests"][1]["body"] == {"messages": [{"role": "user", "content": "bye"}],
                                           "max_tokens": 8,
                                           "response_format": {"type": "json_object"}}


@pytest.mark.parametrize("status", [429, 500, 502, 503])
def test_submit_transient_status_is_retryable(openrouter_mock, status):
    openrouter_mock.routes[("POST", "/batches")] = _err(status)
    with pytest.raises(providers.TransientProviderError):
        providers.submit_batch(MODEL, [[{"role": "user", "content": "x"}]])


@pytest.mark.parametrize("status", [400, 401, 402, 404, 413, 422])
def test_submit_run_level_status_raises(openrouter_mock, status):
    openrouter_mock.routes[("POST", "/batches")] = _err(status, "Insufficient credits")
    with pytest.raises(providers.OpenRouterAPIError) as info:
        providers.submit_batch(MODEL, [[{"role": "user", "content": "x"}]])
    assert info.value.status_code == status
    assert not providers._is_transient(info.value)


def test_submit_job_stays_not_sent_on_payment_required(tmp_path, openrouter_mock):
    openrouter_mock.routes[("POST", "/batches")] = _err(402, "Insufficient credits")
    job = Job(FileStorage(str(tmp_path)).for_job("k"), "k",
              messages=[{"role": "user", "content": "x"}], model=MODEL)
    with pytest.raises(providers.OpenRouterAPIError):
        job.execute()
    assert job.state == Job.NOT_SENT


# --- poll -----------------------------------------------------------------------

@pytest.mark.parametrize("status", ["validating", "in_progress", "finalizing", "cancelling"])
def test_poll_running_is_pending(openrouter_mock, status):
    openrouter_mock.routes[("GET", "/batches/batch-1")] = _batch(status)
    assert providers.poll_batch(_handle()).status == "pending"


def test_poll_completed_without_results_yet_is_pending(openrouter_mock):
    # The quickstart treats `completed` with `results: null` as not ready yet.
    openrouter_mock.routes[("GET", "/batches/batch-1")] = _batch("completed", results=None)
    assert providers.poll_batch(_handle()).status == "pending"


def test_poll_failed_reports_batch_error(openrouter_mock):
    openrouter_mock.routes[("GET", "/batches/batch-1")] = _batch(
        "failed", error={"message": "Batch request 'req-0' contains unsupported content."})
    result = providers.poll_batch(_handle())
    assert result.status == "failed"
    assert "unsupported content" in result.error_message


def test_poll_expired_is_failed_and_cancelled_is_canceled(openrouter_mock):
    openrouter_mock.routes[("GET", "/batches/batch-1")] = _batch("expired")
    assert providers.poll_batch(_handle()).status == "failed"
    openrouter_mock.routes[("GET", "/batches/batch-1")] = _batch("cancelled")
    assert providers.poll_batch(_handle()).status == "canceled"


def test_poll_completed_parses_inline_results_in_request_order(openrouter_mock):
    openrouter_mock.routes[("GET", "/batches/batch-1")] = _batch("completed", results=[
        _result("req-1", "two", finish_reason="length",
                usage={"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24,
                       "prompt_tokens_details": {"cached_tokens": 5}, "cost": 3e-06}),
        _result("req-0", "one"),
    ])
    result = providers.poll_batch(_handle(n=2))
    assert result.status == "completed"
    assert result.texts == ["one", "two"]
    assert result.output_file_id is None
    assert result.finish_reason is None  # bundle
    assert [i["finish_reason"] for i in result.items] == ["stop", "length"]
    assert result.items[1]["usage"] == {"prompt_tokens": 20, "completion_tokens": 4,
                                        "total_tokens": 24, "cached_tokens": 5}
    assert result.usage == {"prompt_tokens": 30, "completion_tokens": 6, "total_tokens": 36}
    # Billed cost comes from OpenRouter, not the litellm price table.
    assert result.items[1]["cost"]["charged_usd"] == pytest.approx(3e-06)
    assert result.items[1]["cost"]["cost_source"] == "provider"
    assert result.cost["charged_usd"] == pytest.approx(4e-06)


def test_poll_single_request_sets_scalar_finish_reason(openrouter_mock):
    openrouter_mock.routes[("GET", "/batches/batch-1")] = _batch(
        "completed", results=[_result("req-0", "ok", finish_reason="stop")])
    result = providers.poll_batch(_handle())
    assert result.texts == ["ok"] and result.finish_reason == "stop"


def test_poll_without_per_result_cost_falls_back_to_batch_total(openrouter_mock):
    no_cost = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
    openrouter_mock.routes[("GET", "/batches/batch-1")] = _batch(
        "completed",
        results=[_result("req-0", "a", usage=no_cost),
                 _result("req-1", "b", usage={**no_cost, "prompt_tokens": 30, "total_tokens": 32})],
        usage={"prompt_tokens": 40, "completion_tokens": 4, "total_tokens": 44, "cost": 8e-06})
    result = providers.poll_batch(_handle(n=2))
    assert result.cost["charged_usd"] == pytest.approx(8e-06)
    # Split by each request's share of the batch's total tokens (12 vs 32 of 44).
    assert result.items[0]["cost"]["charged_usd"] == pytest.approx(8e-06 * 12 / 44)
    assert {i["cost"]["cost_source"] for i in result.items} == {"provider"}


def test_poll_per_request_error_fails_the_job(openrouter_mock):
    bad = {"id": "x", "custom_id": "req-1", "response": None,
           "error": {"type": "invalid_request_error", "message": "max_tokens too large",
                     "param": "max_tokens"}}
    openrouter_mock.routes[("GET", "/batches/batch-1")] = _batch(
        "completed", results=[_result("req-0", "a"), bad])
    result = providers.poll_batch(_handle(n=2))
    assert result.status == "failed"
    assert "req-1" in result.error_message and "max_tokens too large" in result.error_message


def test_poll_missing_result_fails_the_job(openrouter_mock):
    openrouter_mock.routes[("GET", "/batches/batch-1")] = _batch(
        "completed", results=[_result("req-0", "a")])
    result = providers.poll_batch(_handle(n=2))
    assert result.status == "failed"
    assert "missing" in result.error_message


@pytest.mark.parametrize("status", [429, 500, 502])
def test_poll_transient_http_error(openrouter_mock, status):
    openrouter_mock.routes[("GET", "/batches/batch-1")] = _err(status)
    assert providers.poll_batch(_handle()).status == "transient_error"


def test_poll_expired_results_are_failed(openrouter_mock):
    openrouter_mock.routes[("GET", "/batches/batch-1")] = _err(410, "Batch results have expired.")
    result = providers.poll_batch(_handle())
    assert result.status == "failed"
    assert "expired" in result.error_message


def test_poll_not_found_is_retried(openrouter_mock):
    # A GET right after the create can answer 404 before the batch is visible.
    openrouter_mock.routes[("GET", "/batches/batch-1")] = _err(404, "Batch job batch-1 not found.")
    assert providers.poll_batch(_handle()).status == "transient_error"


@pytest.mark.parametrize("status", [401, 402])
def test_poll_run_level_error_raises(openrouter_mock, status):
    # 402 on GET: the finalized cost exceeds the balance; top up and poll again.
    openrouter_mock.routes[("GET", "/batches/batch-1")] = _err(status)
    with pytest.raises(providers.OpenRouterAPIError):
        providers.poll_batch(_handle())


# --- cleanup ----------------------------------------------------------------------

def test_cleanup_deletes_the_batch(openrouter_mock):
    openrouter_mock.routes[("DELETE", "/batches/batch-1")] = {"id": "batch-1"}
    providers.cleanup(_handle())
    assert [c[:2] for c in openrouter_mock.calls] == [("DELETE", "/batches/batch-1")]


def test_cleanup_is_best_effort(openrouter_mock):
    openrouter_mock.routes[("DELETE", "/batches/batch-1")] = _err(409, "still running")
    providers.cleanup(_handle())  # must not raise


def test_cleanup_orphans_deletes_old_terminal_batches_not_live(openrouter_mock, monkeypatch):
    monkeypatch.setattr(providers.time, "time", lambda: 1_000_000.0)
    pages = {
        None: {"data": [{"id": "b-live"}, {"id": "b-old"}], "has_more": True, "last_id": "b-old"},
        "b-old": {"data": [{"id": "b-older"}], "has_more": False, "last_id": "b-older"},
    }
    openrouter_mock.routes[("GET", "/batches")] = lambda _j, params: pages[dict(params).get("after")]
    openrouter_mock.routes[("DELETE", "/batches/b-old")] = {}
    openrouter_mock.routes[("DELETE", "/batches/b-older")] = _err(500)  # skipped, not raised

    deleted = providers.cleanup_orphans("openrouter", {"b-live"}, older_than_days=3)

    assert deleted == ["b-old"]
    _m, _p, _j, params = openrouter_mock.calls[0]
    assert ("created_before", 1_000_000 - 3 * 86400) in params
    assert {v for k, v in params if k == "status"} == {"completed", "failed", "expired", "cancelled"}


def test_cleanup_orphans_never_raises(openrouter_mock):
    openrouter_mock.routes[("GET", "/batches")] = _err(401)
    assert providers.cleanup_orphans("openrouter", set(), 3) == []


# --- cost and errors ---------------------------------------------------------------

def test_batch_rate_is_per_provider(monkeypatch):
    assert providers.batch_rate("openrouter") == 0.5
    monkeypatch.setitem(providers.BATCH_RATES, "openrouter", 0.25)
    assert providers.batch_rate("openrouter") == 0.25
    assert providers.batch_rate("some-new-provider") == providers.BATCH_RATE


def test_batch_cost_dict_prefers_provider_cost(litellm_mock):
    litellm_mock.cost_per_token = lambda **k: (0.01, 0.02)
    cost = providers._cost_dict(MODEL, {"prompt_tokens": 10, "completion_tokens": 1},
                                batch=True, provider_usd=2e-06)
    assert cost == {"charged_usd": 2e-06, "interactive_usd": pytest.approx(4e-06),
                    "saved_usd": pytest.approx(2e-06), "cost_source": "provider"}


def test_transient_provider_error_is_transient():
    assert providers._is_transient(providers.TransientProviderError("net")) is True


# --- the HTTP helper itself (httpx.MockTransport, no network) --------------------

def test_http_helper_sends_auth_and_parses_json(monkeypatch):
    seen = {}

    def handler(request):
        seen["auth"] = request.headers["authorization"]
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(202, json={"id": "batch-1"})

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("OPENROUTER_API_BASE", raising=False)
    monkeypatch.setattr(providers, "_OPENROUTER_TRANSPORT", httpx.MockTransport(handler))
    body = providers._openrouter_http("POST", "/batches", json={"a": 1})
    assert body == {"id": "batch-1"}
    assert seen == {"auth": "Bearer test-key", "url": "https://openrouter.ai/api/v1/batches",
                    "body": {"a": 1}}


def test_http_helper_wraps_error_status(monkeypatch):
    transport = httpx.MockTransport(lambda r: httpx.Response(
        402, json={"error": {"code": 402, "message": "Insufficient credits"}}))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(providers, "_OPENROUTER_TRANSPORT", transport)
    with pytest.raises(providers.OpenRouterAPIError) as info:
        providers._openrouter_http("GET", "/batches/x")
    assert info.value.status_code == 402
    assert "Insufficient credits" in str(info.value)


def test_http_helper_network_error_is_transient(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("dns failure", request=request)

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(providers, "_OPENROUTER_TRANSPORT", httpx.MockTransport(handler))
    with pytest.raises(providers.TransientProviderError):
        providers._openrouter_http("GET", "/batches/x")


def test_http_helper_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(providers.OpenRouterAPIError) as info:
        providers._openrouter_http("GET", "/batches/x")
    assert info.value.status_code == 401
