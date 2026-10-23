# TODO: OpenRouter Batch API provider

**Status (2026-10-24): partly done.** Points 1-6 and 8 are implemented on
`feat/openrouter-batch` (direct HTTP via httpx, no cancel, provider-reported cost,
per-provider `BATCH_RATES`). The open questions are answered in
[`docs/research/openrouter-batch.md`](../research/openrouter-batch.md); the README has an
"OpenRouter batch" section. What remains:

- **Point 7 (pdf2md), not done here.** Add `openrouter` to pdf2md's
  `BATCH_CAPABLE_PROVIDERS` (`src/pdf2md/config.py`), update the tests that expect
  openrouter+batch to be rejected, and bump the llm-batch pin.
- **Images.** OpenRouter batch rejects base64/data-URI images, and pdf2md sends page
  images as `data:image/png;base64,...` (`src/pdf2md/llm.py`). Its LLM pages need
  public https image URLs (hosted page images) before openrouter batch can work for them.
- **Unverified limits:** max requests per batch, max body size, and the actual turnaround
  for non-Vertex providers. Vertex-routed batches may take up to 96 h, longer than the
  default `max_batch_age_days=3` orphan sweep; see the research note.

OpenRouter now has a delayed (async) batch API. Supporting it would give llm-batch a
batch path for any model OpenRouter serves this way. Example: `anthropic/claude-haiku-5.5`,
listed as `anthropic/claude-haiku-5.5:batch`.

Today `openrouter/...` models are interactive-only: pdf2md requires `--mode interactive`
for them.

Sources (fetched 2026-10-22):
- SDK docs: https://openrouter.ai/docs/client-sdks/python/sdks/batch/README
- Model page: https://openrouter.ai/anthropic/claude-haiku-5.5:batch

## What the docs say (verify before building on it)

- **SDK methods** (`open_router.batch`):
  - `create_batches(endpoint, model, requests, completion_window=…, provider=…, callback_url=…)`
    returns HTTP 202 with status `validating`.
  - `get_batches(id)`: a batch in a terminal status carries `results`.
  - `list(limit, after, status, created_after, created_before)`.
  - `delete(id)` works only on terminal batches; it returns 409 while a batch is running.
- **Requests are sent inline.** `requests: [{custom_id, body}]`, with no JSONL file
  upload. This differs from the OpenAI/Vertex flow in `providers.submit_batch`
  (`create_file` + `create_batch`).
- **One endpoint per batch:** `/v1/chat/completions`, `/v1/responses`, `/v1/messages`
  or `/v1/embeddings`.
- **Statuses:** `validating` → … → terminal `completed | failed | expired | cancelled`.
  A batch-level error is in `error.message`.
- **Webhooks** (optional): `callback_url` receives `batch.completed|failed|cancelled|expired`.
  llm-batch polls, so these aren't needed.
- **`provider`:** accepts `{only, allow_fallbacks}`. The docs example pins `google-vertex`.

## Open questions (answer before designing)

1. **Cancel.** The SDK page lists no cancel method, even though a `batch.cancelled`
   event exists.
   - If there is no cancel call, add `openrouter` to the conservative path (not
     `_CANCELLABLE`) so `Job.abandon` / `Pipeline.abandon_all` don't try one.
2. **Per-result format in `results`:** `custom_id`, response body, per-request error,
   `usage`, and whether OpenRouter's `usage.cost` is included.
   - If the real billed cost comes back, prefer it over litellm's price table in
     `_cost_dict`.
3. **Pricing.** Is there a batch discount, and how big? The Haiku 5.5 page shows no
   batch price. If there is no discount, the batch path's value is rate limits and
   throughput, not cost.
   - Set `BATCH_RATE` per provider, not as one global 0.5.
4. **Limits:** max requests per batch, max payload (images!), retention of `results`,
   and the allowed `completion_window` values.
5. **Model naming.** Is the `:batch` suffix needed in `model`, or is it only a catalog
   page?
6. **Transport.** Do it through litellm (does `litellm.create_batch` support
   `custom_llm_provider="openrouter"`?), or call the OpenRouter SDK/HTTP API
   directly in a new provider branch?

## Sketch

1. In `providers.submit_batch`, add a branch for `openrouter` that builds inline
   `requests` (reusing `_build_jsonl`'s per-request body logic) and calls
   `create_batches`.
   - `BatchHandle.input_file_id` = None, since there's no file.
2. In `providers.poll_batch`, map OpenRouter statuses onto `PollResult`. Parse the
   inline `results` into the same per-request items (`_item`): text, finish_reason,
   usage and cost.
3. Map `cancel` if the API supports it; otherwise treat the provider as non-cancellable.
4. `cleanup` should call `delete(id)` after the results are consumed, which is allowed
   on terminal batches. `cleanup_orphans`: `list(status=…, created_before=…)`.
5. In `_is_transient`, treat 429/500/502 as transient and 402 (payment required) as
   run-level/auth-like.
6. Tests: a fake OpenRouter client (no network), plus one opt-in live test with the
   cheapest batch-capable model.
7. Then in pdf2md, allow `--mode batch` for `openrouter/...` models once the provider
   exists. Today the config validation rejects that combination.

8. Cost: like the interactive path, the OpenRouter batch provider should use the provider-reported billed cost (`usage.cost`, `cost_source: "provider"`) rather than the litellm price table, which returns 0.0 for models it does not know.
