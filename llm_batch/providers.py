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
