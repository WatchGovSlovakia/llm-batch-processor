"""Opt-in live test of the OpenRouter batch provider (real API, real money: ~$0.000002).

Skipped unless LLM_BATCH_LIVE=1 and OPENROUTER_API_KEY are set. The model can be
overridden with LLM_BATCH_LIVE_OPENROUTER_MODEL (it must have a batch variant).
"""

import os
import time

import pytest

from llm_batch import providers

pytestmark = pytest.mark.skipif(
    os.environ.get("LLM_BATCH_LIVE") != "1" or not os.environ.get("OPENROUTER_API_KEY"),
    reason="live provider test: set LLM_BATCH_LIVE=1 and OPENROUTER_API_KEY")

MODEL = os.environ.get("LLM_BATCH_LIVE_OPENROUTER_MODEL", "openrouter/mistralai/ministral-8b-2512")
TIMEOUT_S = float(os.environ.get("LLM_BATCH_LIVE_TIMEOUT_S", "1800"))


def test_openrouter_batch_round_trip():
    handle = providers.submit_batch(
        MODEL, [[{"role": "user", "content": "Reply OK"}], [{"role": "user", "content": "Reply YES"}]],
        max_tokens=8)
    try:
        deadline = time.time() + TIMEOUT_S
        result = providers.poll_batch(handle)
        while result.status in ("pending", "transient_error") and time.time() < deadline:
            time.sleep(30)
            result = providers.poll_batch(handle)
        assert result.status == "completed", result.error_message
        assert len(result.texts) == 2 and all(result.texts)
        assert result.usage["total_tokens"] > 0
        assert {i["cost"]["cost_source"] for i in result.items} == {"provider"}
        assert 0 < result.cost["charged_usd"] < 0.001
    finally:
        providers.cleanup(handle)
    # A deleted batch answers 404, which poll retries (a fresh batch can 404 too).
    assert providers.poll_batch(handle).status == "transient_error"
