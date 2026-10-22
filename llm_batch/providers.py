"""Provider layer — the only module that imports litellm.

Isolates batch submission, polling, interactive completion, pricing, and
provider-side cleanup behind a small, provider-agnostic interface so that
`job.py` never sees litellm or per-provider quirks.
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass

import litellm

from llm_batch.errors import TransientProviderError

logger = logging.getLogger(__name__)

# Flat per-token batch discount: you pay this fraction of the interactive price.
# All three providers apply ~50% independent of batch size (see spec §10).
BATCH_RATE = 0.5

# Providers whose batch jobs litellm can cancel (litellm>=1.83.0 adds Vertex).
# Anthropic and gemini are absent: they fall back to the conservative strategy.
_CANCELLABLE = {"openai", "azure", "vertex_ai"}


def provider_supports_cancel(provider: str) -> bool:
    """True iff litellm.cancel_batch can cancel a batch for this provider."""
    return provider in _CANCELLABLE

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
    n_requests: int | None = None  # expected result count, to verify completeness


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
    # Per-request {text, finish_reason, usage, cost}, in request order.
    items: list[dict] | None = None


@dataclass
class CompletionResult:
    """Outcome of one interactive completion."""

    text: str
    usage: dict
    cost: dict
    finish_reason: str | None = None
    cached_tokens: int = 0


def provider_of(model: str) -> str:
    """Map a litellm model string to its custom_llm_provider."""
    if model.startswith("vertex_ai/"):
        return "vertex_ai"
    if model.startswith("claude") or model.startswith("anthropic/"):
        return "anthropic"
    if "/" in model:
        return model.split("/", 1)[0]
    return "openai"


# How batch pricing treats provider-cached input tokens. Set from measurements
# (pdf2md spike, docs/findings/2026-10-16-gemini-cost-spike.md, R2):
#   "batch_only"       batch discount on every token, no cache discount
#   "stack"            cache discount and batch discount multiply
#   "cache_precedence" cached tokens at the cache rate, the rest at the batch rate
BATCH_CACHE_POLICY = "batch_only"


def _price(model: str, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0) -> float:
    """Interactive (non-batch) USD via litellm.cost_per_token; cached tokens at the cache-read rate."""
    try:
        kw = {"model": model, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
        if cached_tokens:
            kw["cache_read_input_tokens"] = cached_tokens
        prompt_cost, completion_cost = litellm.cost_per_token(**kw)
        return float(prompt_cost + completion_cost)
    except Exception:
        return 0.0


def _price_from_usage(model: str, usage: dict) -> float:
    """Interactive (non-batch) USD for this usage, via litellm.cost_per_token."""
    return _price(model, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0),
                  usage.get("cached_tokens", 0))


def _provider_cost(raw_usage) -> float | None:
    """USD the provider itself billed (e.g. OpenRouter's ``usage.cost``), or None."""
    if raw_usage is None:
        return None
    value = (raw_usage.get("cost") if isinstance(raw_usage, dict)
             else getattr(raw_usage, "cost", None))
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _cost_dict(model: str, usage: dict, *, batch: bool, provider_usd: float | None = None) -> dict:
    """Build {charged_usd, interactive_usd, saved_usd, cost_source}.

    cost_source: "provider" (billed cost reported by the provider), "litellm"
    (litellm price table) or "unknown" (tokens were used but no price is known;
    the USD values are then 0.0 and must not be read as free).
    """
    if provider_usd is not None and not batch:
        return {"charged_usd": provider_usd, "interactive_usd": provider_usd,
                "saved_usd": 0.0, "cost_source": "provider"}
    prompt = usage.get("prompt_tokens", 0)
    completion = usage.get("completion_tokens", 0)
    cached = usage.get("cached_tokens", 0)
    interactive_usd = _price(model, prompt, completion, cached)
    if not batch:
        charged_usd = interactive_usd
    elif not cached or BATCH_CACHE_POLICY == "stack":
        charged_usd = interactive_usd * BATCH_RATE
    elif BATCH_CACHE_POLICY == "batch_only":
        charged_usd = _price(model, prompt, completion) * BATCH_RATE
    elif BATCH_CACHE_POLICY == "cache_precedence":
        uncached_usd = _price(model, prompt - cached, completion)
        charged_usd = (interactive_usd - uncached_usd) + uncached_usd * BATCH_RATE
    else:
        raise ValueError(f"unknown BATCH_CACHE_POLICY {BATCH_CACHE_POLICY!r}")
    source = "litellm"
    if interactive_usd <= 0 and (prompt or completion):
        source = "unknown"
        logger.warning("no price known for model %r (%d prompt + %d completion tokens); "
                       "cost reported as 0.0 with cost_source='unknown'", model, prompt, completion)
    return {
        "charged_usd": charged_usd,
        "interactive_usd": interactive_usd,
        "saved_usd": interactive_usd - charged_usd,
        "cost_source": source,
    }


def _cached_tokens(usage) -> int:
    """Provider-reported cached prompt tokens (OpenAI prompt_tokens_details.cached_tokens)."""
    if usage is None:
        return 0
    details = (usage.get("prompt_tokens_details") if isinstance(usage, dict)
               else getattr(usage, "prompt_tokens_details", None))
    if details is None:
        return 0
    value = details.get("cached_tokens") if isinstance(details, dict) else getattr(details, "cached_tokens", None)
    return int(value or 0)


def _item(model: str, text: str, finish_reason, raw_usage, *, batch: bool) -> dict:
    usage = {**_usage_dict(raw_usage), "cached_tokens": _cached_tokens(raw_usage)}
    return {"text": text, "finish_reason": finish_reason, "usage": usage,
            "cost": _cost_dict(model, usage, batch=batch)}


def completion_item(cr: CompletionResult) -> dict:
    """Per-request item for an interactive completion (same shape as PollResult.items)."""
    return {"text": cr.text, "finish_reason": cr.finish_reason,
            "usage": {**cr.usage, "cached_tokens": cr.cached_tokens}, "cost": dict(cr.cost)}


def _is_transient(exc: Exception) -> bool:
    """Classify a provider exception as retryable (transient) vs terminal."""
    status = getattr(exc, "status_code", None)
    if status in _TRANSIENT_STATUS:
        return True
    name = type(exc).__name__.lower()
    return any(frag in name for frag in _TRANSIENT_NAME_FRAGMENTS)


# Re-exported so tests/callers can reference providers.TransientProviderError.
TransientProviderError = TransientProviderError


def _bare_model(model: str) -> str:
    """Strip the litellm '<provider>/' routing prefix, e.g. 'vertex_ai/gemini-2.5-flash'
    -> 'gemini-2.5-flash'. Providers' own batch APIs expect their native model name, not
    litellm's routing string (Vertex derives the model resource path from this field)."""
    prefix = f"{provider_of(model)}/"
    return model[len(prefix):] if model.startswith(prefix) else model


def _build_jsonl(model: str, requests: list, **kw) -> str:
    """One JSON line per request; custom_id req-0..req-N; kwargs merged into body."""
    bare_model = _bare_model(model)
    lines = []
    for i, messages in enumerate(requests):
        lines.append(json.dumps({
            "custom_id": f"req-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": bare_model, "messages": messages, **kw},
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
        n_requests=len(requests),
    )


def cancel(handle: BatchHandle) -> None:
    """Initiate cancellation of a batch job and return immediately.

    Delegates to litellm for all providers (OpenAI/Azure/Vertex). Does NOT wait
    for the cancel to take effect — the outcome is observed by a later poll_batch
    (the job transitions through CANCEL_REQUESTED). Transient errors are wrapped so
    the caller can leave the job in place and retry on the next scheduled poll.
    """
    try:
        litellm.cancel_batch(
            batch_id=handle.batch_job_id,
            custom_llm_provider=handle.provider,
        )
    except Exception as exc:
        if _is_transient(exc):
            raise TransientProviderError(str(exc)) from exc
        raise


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
    cached = _cached_tokens(response.usage)
    cost = _cost_dict(model, {**usage, "cached_tokens": cached}, batch=False,
                      provider_usd=_provider_cost(response.usage))
    finish_reason = getattr(choice, "finish_reason", None)
    return CompletionResult(text=text, usage=usage, cost=cost, finish_reason=finish_reason,
                            cached_tokens=cached)


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


# Provider batch statuses that mean "still working" — including an in-progress
# cancellation, which is non-terminal (we keep polling until it settles).
_PENDING_STATUSES = {"validating", "in_progress", "finalizing", "canceling", "cancelling"}
# Provider batch statuses that mean "the job was cancelled" (terminal). Reported
# distinctly from failure so the speculative state machine can re-issue interactively.
_CANCELED_STATUSES = {"cancelled", "canceled"}
# Provider batch statuses that mean "terminally dead".
_FAILED_STATUSES = {"failed", "expired"}


def poll_batch(handle: BatchHandle) -> PollResult:
    """Poll one batch job once and classify the outcome.

    Returns a PollResult with status pending | completed | failed | transient_error | canceled.
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
    if status in _CANCELED_STATUSES:
        return PollResult(
            status="canceled",
            error_message=f"batch {handle.batch_job_id} reported status={status}",
        )
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
    finish_reasons_by_id: dict[str, str | None] = {}
    items_by_id: dict[str, dict] = {}
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
        items_by_id[custom_id] = _item(handle.model, results_by_id[custom_id],
                                       finish_reasons_by_id[custom_id], body.get("usage"), batch=True)
        usage = _usage_dict(body.get("usage"))
        total["prompt_tokens"] += usage["prompt_tokens"]
        total["completion_tokens"] += usage["completion_tokens"]
        total["total_tokens"] += usage["total_tokens"]

    # Verify completeness: a "completed" batch must yield every submitted
    # request (spec §5.4 — n_requests lets us confirm we got every result).
    expected = handle.n_requests if handle.n_requests is not None else len(results_by_id)
    texts = []
    items = []
    for i in range(expected):
        custom_id = f"req-{i}"
        if custom_id not in results_by_id:
            return PollResult(
                status="failed", output_file_id=output_file_id,
                error_message=(f"missing result for {custom_id} "
                               f"({len(results_by_id)}/{expected} returned)"))
        texts.append(results_by_id[custom_id])
        items.append(items_by_id[custom_id])
    cost = {k: sum(item["cost"][k] for item in items) for k in ("charged_usd", "interactive_usd", "saved_usd")}
    # Scalar finish_reason only makes sense for a single-request batch
    # (the only usage shape any known caller has today); a multi-request/
    # bundle response leaves it None (spec §2 — PollResult.finish_reason).
    finish_reason = finish_reasons_by_id.get("req-0") if len(texts) == 1 else None
    return PollResult(
        status="completed", texts=texts, usage=total, cost=cost,
        output_file_id=output_file_id, finish_reason=finish_reason, items=items)


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
