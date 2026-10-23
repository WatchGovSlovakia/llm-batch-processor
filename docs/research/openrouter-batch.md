# OpenRouter Batch API Research

Researched 2026-10-24 for `docs/todo/2026-10-22-openrouter-batch-api.md`. Sources are the
OpenRouter docs (fetched that day as Markdown via `https://openrouter.ai/docs/llms.txt`), the
public model catalogue (`GET https://openrouter.ai/api/v1/models`), the installed
`litellm==1.106.2`, and a small live probe (section 9). "Verified" means a live API call
confirmed it. "Docs" means only the docs say so.

## Executive Summary

OpenRouter's Batch API (`/api/v1/batches`) takes requests **inline in the create call** (no
JSONL upload), returns **results inline in `GET /batches/{id}`**, and bills the batch variant
of a model, which is **typically 50% of the sync price** (66 of 75 `:batch` variants are
exactly 0.5x). There is **no cancel endpoint**. `DELETE` works on terminal batches only.
Per-result `usage` carries no `cost`. The billed cost is only in the batch-level
`usage.cost`. litellm has no OpenRouter batch support, so the provider talks HTTP directly.
**Base64/data-URI images are rejected** (verified). Only public `http(s)` image URLs work, which
matters for pdf2md, since it sends page images inline.

---

## Answers to the open questions

### 1. Cancel: no cancel call (docs + verified)

- Batch guide, Limitations table: "Cancel a batch | Wait for the batch to finish, then delete it."
  ([batch-api](https://openrouter.ai/docs/guides/features/batch-api.md))
- Quickstart: "You can only delete a finished batch. A batch that is still running returns
  `409`. There is no cancel endpoint yet."
  ([batch-quickstart#delete-the-batch](https://openrouter.ai/docs/batch-quickstart.md))
- The Python SDK page lists `list`, `create_batches`, `delete` and `get_batches`, with no cancel.
  ([python sdk batch](https://openrouter.ai/docs/client-sdks/python/sdks/batch/README.md))
- `cancelled` exists as a status and webhook event: "The provider cancelled the batch." It is
  something the upstream provider does, not the caller.
- **Verified:** `POST /batches/{id}/cancel` returned `404 Not Found`. `DELETE` on a running batch
  returned `409 {"message": "Only completed, failed, expired, or cancelled batches can be
  deleted. Wait for the batch to finish and try again."}`

**Consequence:** `openrouter` is not in `_CANCELLABLE`. `speculative` degrades to
`conservative`, and `Job.abandon` only marks the job failed. The batch still runs and is billed.

### 2. Per-result format (docs + verified)

`GET /batches/{id}` on a terminal batch:

```json
{"id": "batch-1792798821-…", "object": "batch", "status": "completed", "provider": "Mistral",
 "request_counts": {"total": 2, "completed": 2, "failed": 0},
 "usage": {"prompt_tokens": 10, "completion_tokens": 16, "total_tokens": 26,
           "cost": 1.95e-06, "is_byok": false},
 "results": [
  {"id": "batch-…-0", "custom_id": "req-0", "error": null,
   "response": {"status_code": 200, "request_id": null,
     "body": {"object": "chat.completion", "model": "mistralai/ministral-8b-2512",
       "choices": [{"index": 0, "message": {"role": "assistant", "content": "Got it! 😊 **OK"},
                    "finish_reason": "length", "native_finish_reason": "length"}],
       "usage": {"prompt_tokens": 5, "completion_tokens": 8, "total_tokens": 13,
                 "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, …}}}}}],
 "error": null}
```

(verified, trimmed; the raw response is in the probe output)

- Each result has `custom_id` and either `response` (with `status_code` and an OpenAI-shaped
  `body`) or `error` (`{type, message, param, error_type}`). "Results are not in input order,
  so match them on `custom_id`." (quickstart)
- **`usage.cost` is not per result.** The OpenAPI schema for the chat-completions result `usage`
  lists `completion_tokens`, `prompt_tokens`, `prompt_tokens_details`, `server_tool_use`,
  `total_tokens`, and no `cost`. The live probe confirms this. Only the **batch-level**
  `usage.cost` holds the billed amount. Docs: "For a completed batch, `usage.cost` reports the
  amount OpenRouter charged." (batch-api, Pricing). The probe's `1.95e-06` for 26 tokens equals
  26 x $0.075/M, which is exactly the `:batch` catalogue price.
- "A completed batch can contain failed requests. Each result has either a `response` or an
  `error`." (batch-api)
- A batch-level failure is in `error.message`. Verified:
  `"Batch request 'req-0' contains unsupported 'image_url' content. Only public http(s) image
  URLs are supported in batch; base64/data-URI images are rejected."`
- `completed` can briefly come with `results: null`. The quickstart's own poll loop waits for
  `batch.status !== 'completed' || batch.results !== null`.
- List items now carry `usage` too (verified), but `results` is always `null` in a list.

**Consequence:** the provider uses provider-reported cost (`cost_source: "provider"`). The
batch total is exact. The per-request split is proportional to `total_tokens` and is an
approximation. If a future API adds per-result `usage.cost`, that value is used instead.

### 3. Pricing / discount (docs + catalogue)

- "Price | Typically 50% of the sync per-token price. Each model page shows its batch price."
  "Some components are not discounted: web search calls bill at standard rates, and prompt
  caching rates vary by model. The price on each model page is the source of truth."
  (batch-api)
- Catalogue, 2026-10-24: 75 `…:batch` model ids. Comparing `pricing.prompt/completion`
  with the base id gives exactly **0.5x for 66 of them**. The outliers are 0.455x
  (`google/gemini-3.8-flash`), 0.72x (`qwen/qwen3.7-plus`), 0.8x, 0.2x and 0.33x, plus three
  variants priced *above* sync. `anthropic/claude-haiku-5.5:batch` is 0.5x on prompt,
  completion and cache read.
- Estimate check: "Before OpenRouter accepts a batch, it estimates the cost. If the estimate is
  more than your balance, the submit returns `402`." `GET` can also return `402`: "The
  finalized batch cost exceeds the available balance; metadata is returned with
  `results: null`. Top up credits and poll again to receive the persisted results." (OpenAPI
  `get /batches/{id}`)

**Consequence:** `BATCH_RATES["openrouter"] = 0.5`. Since the charged cost is
provider-reported, the rate only derives the `interactive_usd` reference
(`charged / rate`). For the outlier models that reference is off.

### 4. Limits (docs; numbers deliberately unpublished)

- "Up to 200 MB per batch, with a cap on the number of requests." "Batch limits generally
  follow the batch limits of the upstream providers. OpenRouter adjusts them over time, and
  they can differ between accounts, so this page does not list fixed numbers. When a submit
  hits a limit, `error.message` states the current value." (batch-api, Limits)
- Over the limit: requests per batch or body > 200 MB returns `413`. Batches in progress,
  requests in progress, or requests submitted per minute return `429`.
- `completion_window`: "`24h`, which is the default and the only accepted value."
  (quickstart). Real turnaround depends on the provider: "Most providers allow 24 hours. Google
  AI Studio allows 48 hours, and Google Vertex allows 96 hours." Vertex "can queue a batch for
  up to 72 hours". **This exceeds `Poller(max_batch_age_days=3)`.**
- Retention: "Deleted 30 days after you submit, or when you delete the batch." After that,
  `GET` returns `410 Gone`.
- Images: "Base64 or data-URI images | Host the image and send a public `https` URL."
  **Verified rejected** (the batch moves to `failed` within ~2 s, at no cost). Inline PDFs only
  for "Accounts enabled by support … on Anthropic and DeepInfra, up to 20 MiB each". Image-URL
  support differs by upstream: Anthropic, Mistral, OpenAI, Vertex and AI Studio yes;
  Fireworks, Parasail and Morph no.
- Other unsupported features: image/audio output, `:online`/web plugin, ZDR, `stream: true`,
  provider `order`/`sort` (only `provider.only` and `allow_fallbacks`).
- The exact numeric limits (requests per batch, in-progress caps) are **unverified**: the docs
  don't publish them and the probe didn't hit them.

### 5. Model naming: use the plain slug (docs + verified)

- Quickstart: "A model from the batch models list. Use its normal slug, for example
  `anthropic/claude-haiku-5.5`. A model with no batch variant returns `400`."
- **Verified:** submitting `mistralai/ministral-8b-2512:batch` was also accepted. The batch
  object echoed `"model": "mistralai/ministral-8b-2512"`, while the per-result body reported
  `"mistralai/ministral-8b-2512:batch"`. Cost and behaviour were identical.

**Decision:** callers pass the normal `openrouter/<vendor>/<model>` string, and the provider
strips the `openrouter/` prefix and sends the slug unchanged. That keeps one model string
valid for both batch and the interactive deadline fallback (`litellm.completion`). A `:batch`
suffix is passed through unchanged, and OpenRouter accepts it in batch mode.

### 6. Transport: direct HTTP (verified in the litellm source)

- `litellm/batches/main.py` (`create_batch`, `retrieve_batch`, `cancel_batch`) dispatches on
  `OPENAI_COMPATIBLE_BATCH_AND_FILES_PROVIDERS = {openai, hosted_vllm, litellm_proxy}`, plus
  `azure`, `vertex_ai`, `anthropic`, `bedrock`, `xai` and `mistral` branches. Anything else
  raises `BadRequestError("LiteLLM doesn't support custom_llm_provider=… for 'create_batch'")`.
  `litellm/llms/openrouter/` has chat, responses, embedding and image modules, and no batches.
- Faking it as `openai` with an OpenRouter `api_base` cannot work either: litellm's
  `create_batch` needs an `input_file_id` (the OpenAI file flow), while OpenRouter wants inline
  `requests` with `endpoint`, `model` and `requests` last in the body.
- The `openrouter` Python SDK is not installed in the project venv. `httpx` is already
  installed as a litellm dependency, so a ~20-line HTTP helper adds no new dependency.

**Decision:** `providers._openrouter_http` (httpx, `Authorization: Bearer $OPENROUTER_API_KEY`,
base URL `OPENROUTER_API_BASE` or `https://openrouter.ai/api/v1`).

---

## 7. Error classification

| Call | Status | Meaning (docs) | llm-batch handling |
|---|---|---|---|
| POST /batches | 400 / 404 / 413 / 422 | bad input, no batch variant, no provider in `only`, too big, `:online` | raise; job stays NOT_SENT |
| POST /batches | 401 / 402 | bad key / estimate exceeds balance | raise (run-level); job stays NOT_SENT |
| POST /batches | 429 / 5xx | rate or in-progress limits; service down | `TransientProviderError` |
| GET /batches/{id} | 404 | not found. **Verified: also returned right after create** | `transient_error` (retry). A truly deleted batch ends as stale in the Poller |
| GET /batches/{id} | 410 | results past 30-day retention | `failed` |
| GET /batches/{id} | 401 / 402 | bad key / finalized cost exceeds balance | raise (run-level); job stays SENT; top up and re-run |
| GET /batches/{id} | 429 / 5xx | | `transient_error` |
| DELETE | 409 | still running, or BYOK key disabled | swallowed (best-effort cleanup) |
| network | | | `TransientProviderError` |

## 8. Status mapping

| OpenRouter `status` | `PollResult.status` |
|---|---|
| `validating`, `in_progress`, `finalizing`, `cancelling` | `pending` |
| `completed` with `results: null` | `pending` |
| `completed` | `completed`, or `failed` if any result has an `error` or a `req-i` is missing |
| `failed`, `expired` | `failed` (with `error.message`) |
| `cancelled` | `canceled` |

## 9. Live probe (2026-10-24)

Scripts and raw responses (no secrets) are in `./tmp/` of the working clone, which is not
committed. Model: `mistralai/ministral-8b-2512` ($0.15/M sync, $0.075/M batch, image input),
with `max_tokens: 8`.

| Batch | Content | Result |
|---|---|---|
| A | plain slug, 2 text requests | 202 `validating`, then `completed` in **66 s**; batch `usage.cost` 1.95e-06; no per-result cost |
| B | `:batch` suffix, 1 request | accepted, `completed` in 64 s, cost 9.75e-07 |
| C | 1 request with a base64 1x1 PNG | 202, then `failed` in 2 s: "base64/data-URI images are rejected"; cost 0 |

Also verified: `DELETE` running → 409; `POST …/cancel` → 404; list with repeated `status=`
filters works, newest first; bad key → 401 `User not found.`; unknown id → 404; `DELETE` on
terminal → 200 `{"deletion": {"openrouter": "deleted", "upstream": {"provider": "Mistral",
"status": "deleted"}}}`, then `GET` → 404.

The opt-in live test `tests/test_openrouter_live.py` ran three times. The first two exposed the
404-right-after-create race, and their batches were deleted afterwards. The third passed in
94 s.

**Total live spend: 8.8e-06 USD** (A 1.95e-06 + B 9.75e-07 + C 0 + three live-test batches at
1.95e-06 each), as reported by OpenRouter's `usage.cost`. The account's `/key` usage delta
(+0.0026) can't isolate it, because other traffic used the same key at the same time (most
likely the running pdf2md conversion).

## 10. Not verified

- Numeric limits (requests per batch, in-progress caps, per-minute submits).
- Whether per-result `usage.cost` appears for other upstreams (only Mistral was probed).
- Behaviour of `expired` and provider `cancelled` batches, and whether partial results come
  with them.
- Image-URL requests (none sent; docs only).
- Prompt-cache pricing inside batches ("prompt caching rates vary by model").
