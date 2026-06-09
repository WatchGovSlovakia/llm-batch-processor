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
