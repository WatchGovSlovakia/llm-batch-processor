# Anthropic Message Batches API Research

## Executive Summary

Anthropic's **Message Batches API** provides asynchronous, cost-effective batch processing of Claude API requests with a **flat 50% discount** on both input and output tokens regardless of batch size. Batches typically complete within 1 hour (SLA: up to 24 hours), with results accessible via streaming `.jsonl` files. Key limitations: 100,000 requests per batch (or 256 MB), with per-request result types (succeeded/errored/canceled/expired) tracked independently. litellm provides native support via `anthropic_batches` custom provider.

---

## 1. Pricing & Discount Model

### Discount Structure

**All usage in batch mode is charged at 50% of standard API prices.**

From [Pricing - Claude API Docs](https://platform.claude.com/docs/en/about-claude/pricing):

| Model | Standard Input | Batch Input | Standard Output | Batch Output |
|-------|---|---|---|---|
| Claude Opus 4.8 | $5 / MTok | $2.50 / MTok | $25 / MTok | $12.50 / MTok |
| Claude Sonnet 4.6 | $3 / MTok | $1.50 / MTok | $15 / MTok | $7.50 / MTok |
| Claude Haiku 4.5 | $1 / MTok | $0.50 / MTok | $5 / MTok | $2.50 / MTok |

### Key Detail: Discount is Flat Per-Token

The discount is **NOT variable based on batch size**. The 50% discount is applied uniformly to every token regardless of whether the batch contains 1 request or 100,000 requests. This means:
- Batching 1 request = 50% discount on that request
- Batching 100,000 requests = 50% discount on all 100,000 requests

Batch and prompt caching discounts can be combined for additional savings (multiplicative stacking).

**Source:** [Pricing - Claude API Docs](https://platform.claude.com/docs/en/about-claude/pricing#batch-processing)

---

## 2. Processing Timeline & SLA

### Typical Completion

- **Most batches complete within 1 hour**
- **Maximum SLA: 24 hours** (results expire after 24 hours if not retrieved)

### Result Availability

- Results are accessible via a `.jsonl` streaming endpoint once processing ends
- Results are not guaranteed to be in the same order as submitted requests; use `custom_id` for correlation
- Results can be archived (batches marked `archived_at`), after which results become unavailable

**Source:** [Batch processing - Claude Docs](https://platform.claude.com/docs/en/docs/build-with-claude/batch-processing)

---

## 3. API Mechanics

### Core Workflow

1. **Create Batch** (`POST /v1/messages/batches`)
   - Submit list of requests with unique `custom_id` and `params`
   - Returns batch object with `id` and `processing_status: "in_progress"`

2. **Poll Status** (`GET /v1/messages/batches/{message_batch_id}`)
   - Check `processing_status` (values: `in_progress`, `canceling`, `ended`)
   - Track per-request counts: `processing`, `succeeded`, `errored`, `canceled`, `expired`

3. **Retrieve Results** (`GET /v1/messages/batches/{message_batch_id}/results`)
   - Stream `.jsonl` file with one result per line
   - Each result contains `custom_id` and `result` object

4. **List Batches** (`GET /v1/messages/batches`)
   - Optional pagination with `after_id`, `before_id`, `limit`

### Request Format (Create Batch)

```json
{
  "requests": [
    {
      "custom_id": "request-1",
      "params": {
        "model": "claude-opus-4-8",
        "max_tokens": 1024,
        "messages": [
          {"role": "user", "content": "Hello, Claude"}
        ]
      }
    },
    {
      "custom_id": "request-2",
      "params": {
        "model": "claude-opus-4-8",
        "max_tokens": 1024,
        "messages": [
          {"role": "user", "content": "What is 2+2?"}
        ]
      }
    }
  ]
}
```

### Response Format (Batch Status)

```json
{
  "id": "msgbatch_013Zva2CMHLNnXjNJJKqJ2EF",
  "type": "message_batch",
  "processing_status": "in_progress",
  "request_counts": {
    "processing": 100,
    "succeeded": 50,
    "errored": 30,
    "canceled": 10,
    "expired": 10
  },
  "created_at": "2024-08-20T18:37:24.100435Z",
  "ended_at": "2024-08-20T18:37:24.100435Z",
  "expires_at": "2024-08-20T18:37:24.100435Z",
  "results_url": "https://api.anthropic.com/v1/messages/batches/msgbatch_013Zva2CMHLNnXjNJJKqJ2EF/results"
}
```

### Result Format (.jsonl)

Each line is a JSON object (no nesting):

```json
{"custom_id": "request-1", "result": {"type": "succeeded", "message": {...}}}
{"custom_id": "request-2", "result": {"type": "errored", "error": {...}}}
{"custom_id": "request-3", "result": {"type": "expired"}}
```

### Processing Status Values

**Batch-level statuses** (`processing_status`):
- `in_progress` — Batch is being processed
- `canceling` — Cancellation initiated but not yet complete
- `ended` — All requests have been processed or timed out

**Per-request result types**:
- `succeeded` — Request completed successfully, `message` field contains response
- `errored` — Request failed, `error` field contains error object
- `canceled` — Request was canceled before processing
- `expired` — Request timed out (no result within timeout window)

**Source:** [List Message Batches API Reference](https://platform.claude.com/docs/en/api/listing-message-batches), [Retrieve Message Batch results API Reference](https://platform.claude.com/docs/en/api/retrieving-message-batch-results)

---

## 4. Limits & Constraints

### Batch Size Limits

- **Max requests per batch:** 100,000 requests
- **Max batch size:** 256 MB (request + response combined)
- **Whichever limit is hit first** terminates the batch

### Rate Limits

The batch API has dedicated rate limiting separate from the standard Messages API:

- **enqueued_batch_requests:** Maximum number of batch requests in the processing queue at once (typically 500,000 across the organization)
- **requests_per_minute (RPM):** Applies to batch creation endpoints

Query the **Rate Limits API** for exact values:

```bash
curl "https://api.anthropic.com/v1/organizations/rate_limits?group_type=batch" \
  -H "anthropic-version: 2023-06-01" \
  -H "x-api-key: $ANTHROPIC_ADMIN_KEY"
```

Response example:
```json
{
  "data": [
    {
      "type": "rate_limit",
      "group_type": "batch",
      "models": null,
      "limits": [{"type": "enqueued_batch_requests", "value": 500000}]
    }
  ]
}
```

**Source:** [Rate Limits API - Claude API Docs](https://platform.claude.com/docs/en/manage-claude/rate-limits-api)

### Data Retention

- **Result availability:** Results accessible for 24 hours after batch creation (via `expires_at` field)
- **Archived batches:** Once archived, results become inaccessible
- **Zero Data Retention (ZDR):** **Batch API is NOT eligible for ZDR**; data retained per standard policy
- **Files uploaded to container (code execution):** Expire at time shown in `container.expires_at`

**Source:** [Batch processing - Claude Docs](https://platform.claude.com/docs/en/docs/build-with-claude/batch-processing)

---

## 5. Supported Features

### Full Feature Support

Almost any request possible with the Messages API can be submitted in a batch, including:

- **Tool use:** Client-side tools, server-side tools (web search, web fetch, code execution, MCP connectors, etc.)
- **Prompt caching:** Batch requests can leverage cached content
- **Vision:** Image inputs via `image_media_type` and `image_data`
- **Extended thinking:** Longer reasoning (where available)
- **All Claude models:** The batch API works with all active Claude model versions

### Unsupported Features

- **Fast mode:** Not available with Batch API
- **Zero Data Retention (ZDR):** Batch API uses standard retention

**Source:** [Batch processing - Claude Docs](https://platform.claude.com/docs/en/docs/build-with-claude/batch-processing)

---

## 6. litellm Integration & Support

### Current Status

litellm provides **native support for Anthropic batch operations** via the `anthropic_batches` custom provider.

**Supported endpoints:**
- `create_batch()` — Create a batch with requests
- `retrieve_batch()` — Get batch status and metadata
- `cancel_batch()` — Cancel an in-progress batch
- `list_batch()` — List batches (pagination support)
- `batch_content()` or `file_content()` — Retrieve batch results

### How litellm Handles Requests

litellm supports **inline request submission** (no file upload required):

```python
response = litellm.create_batch(
    model="anthropic_batches/claude-opus-4-8",
    requests=[
        {
            "custom_id": "msg-1",
            "params": {
                "model": "claude-opus-4-8",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hi"}]
            }
        }
    ]
)
```

Unlike OpenAI (which requires JSONL file upload), Anthropic batches can submit requests directly in the payload.

### Authentication

```python
import litellm
litellm.api_key = os.environ["ANTHROPIC_API_KEY"]
```

### Polling & Result Retrieval

```python
# Poll status
batch = litellm.retrieve_batch(batch_id=batch_id, custom_llm_provider="anthropic_batches")

# Retrieve results (streaming .jsonl)
results = litellm.batch_content(batch_id=batch_id, custom_llm_provider="anthropic_batches")
```

### Cost Tracking

For batch pass-through cost tracking to work correctly in litellm, define the Anthropic model in `proxy_config.yaml`:

```yaml
model_list:
  - model_name: "claude-opus-4-8-batch"
    litellm_params:
      model: "anthropic_batches/claude-opus-4-8"
      custom_llm_provider: "anthropic_batches"
```

This ensures the polling mechanism correctly identifies the provider and calculates costs.

**Source:** [Anthropic Batches support (GitHub Discussion #17280)](https://github.com/BerriAI/litellm/discussions/17280), [/batches - liteLLM Docs](https://docs.litellm.ai/docs/batches), [Anthropic Passthrough - liteLLM Docs](https://docs.litellm.ai/docs/pass_through/anthropic_completion)

---

## 7. Authentication

### API Key

- **Environment variable:** `ANTHROPIC_API_KEY`
- **Key format:** `sk-ant-...` (standard API key)
- **Not required:** Admin API keys (used for Rate Limits API only)

### Required Headers

```bash
curl https://api.anthropic.com/v1/messages/batches \
  -H "anthropic-version: 2023-06-01" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "content-type: application/json"
```

**Source:** [API Overview - Claude API Docs](https://platform.claude.com/docs/en/api/overview)

---

## 8. Example: End-to-End Workflow

### Create a batch

```bash
curl -X POST https://api.anthropic.com/v1/messages/batches \
  -H "anthropic-version: 2023-06-01" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "content-type: application/json" \
  -d '{
    "requests": [
      {
        "custom_id": "test-1",
        "params": {
          "model": "claude-opus-4-8",
          "max_tokens": 100,
          "messages": [{"role": "user", "content": "Say hello"}]
        }
      }
    ]
  }'
```

Response:
```json
{
  "id": "msgbatch_1234...",
  "type": "message_batch",
  "processing_status": "in_progress",
  ...
}
```

### Poll for completion

```bash
curl https://api.anthropic.com/v1/messages/batches/msgbatch_1234 \
  -H "anthropic-version: 2023-06-01" \
  -H "x-api-key: $ANTHROPIC_API_KEY"
```

When `processing_status` is `ended`, retrieve results.

### Retrieve results

```bash
curl https://api.anthropic.com/v1/messages/batches/msgbatch_1234/results \
  -H "anthropic-version: 2023-06-01" \
  -H "x-api-key: $ANTHROPIC_API_KEY"
```

Returns streaming `.jsonl` with results:
```
{"custom_id": "test-1", "result": {"type": "succeeded", "message": {...}}}
```

---

## 9. Key Considerations for Library Implementation

1. **Polling Strategy:** Implement exponential backoff for status checks; most batches complete within 1 hour but can take up to 24 hours
2. **Result Streaming:** The results endpoint returns a streaming `.jsonl` file; parse line-by-line to avoid loading entire batch in memory
3. **Error Handling:** Per-request errors are in the result set, not batch-level errors; process each result's error field independently
4. **Result Correlation:** Always use `custom_id` to match results to requests since order is not guaranteed
5. **Rate Limiting:** Monitor `enqueued_batch_requests` limit via Rate Limits API before submitting large numbers of batches
6. **Cost Optimization:** Batch 50%+ discount applies uniformly; combine with prompt caching for additional savings
7. **Retention Window:** Plan to retrieve results within 24 hours; implement archival strategy for long-term storage

---

---

## Batch Data Storage, Retention & Cleanup

### 1. Storage Model: Inline Request Submission (Confirmed)

Anthropic's Batch API **uses inline request submission** with no separate file upload required. Unlike OpenAI's Batches API (which requires uploading JSONL files to a Files API), Anthropic accepts requests directly in the batch creation payload:

```json
{
  "requests": [
    {
      "custom_id": "request-1",
      "params": {
        "model": "claude-opus-4-8",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hello"}]
      }
    }
  ]
}
```

**There is no uploaded "input file" object to manage.** Requests are submitted directly in the `requests` array of the create-batch payload. The Batch API stores the full request and response data server-side, but there are no separate file objects created as part of batch submission.

**Source:** [Batch processing - Claude Docs](https://platform.claude.com/docs/en/build-with-claude/batch-processing#how-to-use-the-message-batches-api)

---

### 2. Storage Cost: Free (No Separate Charge)

Anthropic **does not charge separately for storing batch request/result data.** All costs are token-based:

- **Token pricing:** 50% discount on all input/output tokens (flat rate, independent of batch size)
- **Storage charge:** None documented; storage is included in standard operations
- **Data retention:** No additional fee for the 29-day retention window

**Source:** [Batch processing - Claude Docs](https://platform.claude.com/docs/en/build-with-claude/batch-processing#pricing)

---

### 3. Retention & Auto-deletion: 29 Days, Then Auto-Expires

**Retention period:**
- Batch results are **available for 29 days after batch creation** (not after processing completion)
- Results can still be viewed after 29 days, but **results will no longer be available for download**
- Processing must complete within 24 hours; batches expire if not complete within 24 hours

**Auto-deletion:**
- Results **automatically expire and become inaccessible after 29 days**
- No manual action required; data is automatically purged
- Once expired, batch results cannot be retrieved, though the batch metadata (creation date, status) remains visible in the list endpoint

**Source:** [Batch processing - Claude Docs](https://platform.claude.com/docs/en/build-with-claude/batch-processing#batch-limitations) ("Batch results are available for 29 days after creation. After that, you may still view the Batch, but its results will no longer be available for download.")

---

### 4. Explicit Cleanup: Delete Endpoint Available (Optional)

Anthropic provides a **DELETE endpoint for explicit batch deletion:**

```bash
curl https://api.anthropic.com/v1/messages/batches/{message_batch_id} \
  -X DELETE \
  -H 'anthropic-version: 2023-06-01' \
  -H "X-Api-Key: $ANTHROPIC_API_KEY"
```

**Deletion rules:**
- Batches can only be deleted after processing is complete (`processing_status == "ended"`)
- To delete an in-progress batch, **cancel it first** using the cancel endpoint, then delete
- Response: `{ "id": "...", "type": "message_batch_deleted" }`

**Is cleanup necessary?** **No.** Deletion is optional. Since results auto-expire and are automatically purged after 29 days with no lingering costs, explicit deletion is not required for cleanup. However, you *may* delete batches earlier if desired for explicit data removal (e.g., compliance, privacy).

**Source:** [Delete a Message Batch - Claude API Reference](https://platform.claude.com/docs/en/api/deleting-message-batches)

---

### 5. Zero Data Retention (ZDR) Eligibility: NOT Eligible

The Batch API is **explicitly NOT eligible for Zero Data Retention (ZDR).**

- **ZDR status:** No
- **Why:** Batch API is a fundamentally stateful feature requiring server-side storage of both inputs and outputs during and after processing
- **Retention policy:** Uses **standard 29-day retention** (not ZDR)
- **HIPAA eligibility:** Also **NO**

From official docs: _"This feature is **not** eligible for [Zero Data Retention (ZDR)](/docs/en/build-with-claude/api-and-data-retention). Data is retained according to the feature's standard retention policy."_

**Source:** [Batch processing - Claude Docs](https://platform.claude.com/docs/en/build-with-claude/batch-processing) (top-level note) and [API and data retention - Claude Docs](https://platform.claude.com/docs/en/manage-claude/api-and-data-retention#feature-eligibility)

---

### Summary & Recommendation

**Storage model:** Inline request submission (no file upload object)  
**Storage cost:** Free (no charge; token pricing only)  
**Retention:** 29 days, auto-expires and auto-purged  
**Auto-deletion:** Yes, automatic after 29 days  
**Explicit cleanup endpoint:** Yes (DELETE), optional  
**ZDR eligible:** No  

**Recommendation for library authors:**  
**Explicit cleanup is NOT necessary.** Anthropic Batch API does not require manual deletion logic. Results automatically expire and are purged after 29 days with zero cost overhead. If your library needs to clean up batch data:
- For cost control: not needed (storage is free)
- For data privacy/compliance: optional; use DELETE endpoint if you must remove data before auto-expiry
- Simplest approach: retrieve results within 29 days, then let auto-expiry handle cleanup with no further action required

**Sources:**
- [Batch processing - Claude Docs](https://platform.claude.com/docs/en/build-with-claude/batch-processing)
- [API and data retention - Claude Docs](https://platform.claude.com/docs/en/manage-claude/api-and-data-retention)
- [Delete a Message Batch - Claude API Reference](https://platform.claude.com/docs/en/api/deleting-message-batches)

---

## Batch Cancellation: Cost, Latency, Tooling

### 1. Can You Cancel an In-Progress Batch?

**Yes.** Cancellation is fully supported via:

- **Endpoint:** `POST /v1/messages/batches/{message_batch_id}/cancel` ([Cancel a Message Batch - Claude API Reference](https://platform.claude.com/docs/en/api/canceling-message-batches))
- **Native SDKs:** Anthropic Python SDK: `client.messages.batches.cancel(message_batch_id)` or `client.beta.messages.batches.cancel(message_batch_id)`
- **Status transitions:** 
  - Immediately after cancel call: `processing_status` becomes `"canceling"`
  - When cancellation finalizes: `processing_status` becomes `"ended"`
  - Per-request result types remain: `"succeeded"`, `"errored"`, `"canceled"`, `"expired"` (tracked in `request_counts`)
  - A `cancel_initiated_at` field (RFC 3339 datetime) is populated on the batch object

**Key behavior:** The batch enters a `canceling` state, but "cancellation may not take immediate effect." The system may complete any **non-interruptible, in-progress requests** before finalizing cancellation. This means a partial result set is typical — requests already sent to the model may finish before cancellation is finalized. ([Cancel a Message Batch - Claude API Reference](https://platform.claude.com/docs/en/api/canceling-message-batches))

---

### 2. Cancellation Latency

**No quantified SLA is published.** Anthropic's documentation states:

- Cancellation initiates immediately (status changes to `"canceling"`)
- **The system may complete any in-progress, non-interruptible requests before finalizing cancellation**
- No explicit timeout or latency guarantee is documented

**Best practice:** Poll the batch status (`GET /v1/messages/batches/{message_batch_id}`) until `processing_status == "ended"`. Most cancellations likely complete within seconds to minutes, but Anthropic does not publish a numeric latency target. The longest latency would be bound by the longest in-flight request's remaining execution time.

**Source:** [Cancel a Message Batch - Claude API Reference](https://platform.claude.com/docs/en/api/canceling-message-batches)

---

### 3. Billing on Cancel: Token Charges & Costs

**Billing is based on result type:**

| Result Type | Billed? | Notes |
|---|---|---|
| `succeeded` | ✅ Yes | Full token cost (input + output tokens) charged |
| `errored` | ❌ No | "You will not be billed for these requests" |
| `canceled` | ❌ No | "User canceled the batch before this request could be sent to the model. You will not be billed for these requests." |
| `expired` | ❌ No | "Batch reached its 24 hour expiration before this request could be sent to the model. You will not be billed for these requests." |

**Cancellation fee:** None. No separate charge for initiating cancellation.

**Key implication for "submit then cancel" pattern:** You pay **only for requests that successfully reached the model and produced output** (i.e., `result.type == "succeeded"`). Requests canceled before processing begins incur **zero charge**. However, requests already in progress when cancellation is initiated may complete and be charged — cancellation cannot interrupt them mid-execution.

**Source:** [Batch processing - Claude Docs](https://platform.claude.com/docs/en/docs/build-with-claude/batch-processing) — request result types and billing table

---

### 4. Abandoned/Expired Batch Cost

If you **let a batch expire** (do not retrieve results within 24 hours), the same billing rule applies:

- **Expired requests are NOT billed.** Per-request result type `"expired"` means "Batch reached its 24 hour expiration before this request could be sent to the model. You will not be billed for these requests."
- You are charged only for requests that reached `"succeeded"` before the 24-hour deadline.
- **No expiration fee:** There is no penalty or extra charge for allowing results to expire.
- **Result retention:** Results remain accessible for **29 days** after batch creation (not from processing completion), but can only be retrieved if accessed before the batch `expires_at` timestamp (24 hours).

**Source:** [Batch processing - Claude Docs](https://platform.claude.com/docs/en/docs/build-with-claude/batch-processing)

---

### 5. Storage Cost During the Wait

**Zero storage charge.** Anthropic's Batch API uses **inline request submission** (no uploaded JSONL file):

- Requests are submitted directly in the `requests` array of the create-batch payload
- **No separate file storage object** is created (unlike OpenAI)
- **No storage cost** is documented — all costs are token-based (50% discount applies uniformly)
- Results are stored server-side for 24 hours (processing window) and remain accessible for 29 days (retention window), but no additional charge is incurred for this storage

**Source:** [Batch processing - Claude Docs](https://platform.claude.com/docs/en/docs/build-with-claude/batch-processing#how-to-use-the-message-batches-api) and [Batch Data Storage, Retention & Cleanup](https://platform.claude.com/docs/en/docs/build-with-claude/batch-processing) (section 8.2 in this document)

---

### 6. litellm Support for Anthropic Batch Cancellation

**⚠️ Current status (as of June 2026): litellm does NOT support `cancel_batch` for Anthropic.**

**litellm's `cancel_batch` function signature:**

```python
def cancel_batch(
    batch_id: str,
    model: Optional[str] = None,
    custom_llm_provider: Union[Literal["openai", "azure", "vertex_ai"], str] = "openai",
    metadata: Optional[Dict[str, str]] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, str]] = None,
    **kwargs,
) -> Union[LiteLLMBatch, Coroutine[Any, Any, LiteLLMBatch]]
```

**Supported providers:** `"openai"`, `"azure"`, `"vertex_ai"` only. **Anthropic is NOT in the literal types.**

**Workaround:** Use the **Anthropic native Python SDK directly** for cancellation:
```python
import anthropic

client = anthropic.Anthropic(api_key="YOUR_KEY")
client.messages.batches.cancel(message_batch_id="msgbatch_...")
```

**Why the gap?** litellm has `create_batch()` and `retrieve_batch()` support for Anthropic (via `custom_llm_provider="anthropic_batches"`), but the `cancel_batch()` endpoint was not yet extended to Anthropic. This is a **known limitation** ([Anthropic Batches support + inline requests feature · BerriAI/litellm · Discussion #17280](https://github.com/BerriAI/litellm/discussions/17280)).

**Source:** [litellm batches.py on GitHub](https://github.com/BerriAI/litellm/blob/main/litellm/batches/main.py) and [/batches - liteLLM Docs](https://docs.litellm.ai/docs/batches)

---

## Summary: Is "Submit Then Cancel If Deadline Nears" Essentially Free?

**Answer: Mostly yes, with caveats.**

✅ **Cost-free cancellation:** Initiating cancellation itself has no fee, and requests canceled before processing begins incur zero token charges.

⚠️ **Partial billing risk:** Requests already executing when cancellation is initiated will complete and be charged at the 50% batch discount rate. Cancellation **cannot interrupt in-flight requests** — this is the main cost exposure.

⏱️ **Latency:** Cancellation status changes immediately to `"canceling"`, but finalization (when `processing_status == "ended"`) depends on how many requests are mid-execution. Latency is typically fast (seconds to minutes) but **not quantified by Anthropic**.

❌ **litellm gap:** litellm's `cancel_batch()` does not support Anthropic. Use the **native Anthropic SDK** (`client.messages.batches.cancel(...)`) instead.

**Recommendation for library authors:** The pattern is viable for a "deadline-based fallback" (cancel if a deadline approaches and re-run interactively). The cost overhead is bounded to in-flight requests only, which should be a small fraction of total batch size. Deploy with monitoring on `cancel_initiated_at` vs. `succeeded` request counts to quantify actual cancellation overhead in your workload.

---

## References

- [Batch processing - Claude Docs](https://platform.claude.com/docs/en/build-with-claude/batch-processing)
- [Create a Message Batch API Reference](https://platform.claude.com/docs/en/api/creating-message-batches)
- [Retrieve Message Batch results API Reference](https://platform.claude.com/docs/en/api/retrieving-message-batch-results)
- [List Message Batches API Reference](https://platform.claude.com/docs/en/api/listing-message-batches)
- [Delete a Message Batch - Claude API Reference](https://platform.claude.com/docs/en/api/deleting-message-batches)
- [Cancel a Message Batch - Claude API Reference](https://platform.claude.com/docs/en/api/canceling-message-batches)
- [Pricing - Claude API Docs](https://platform.claude.com/docs/en/about-claude/pricing)
- [Rate Limits API - Claude API Docs](https://platform.claude.com/docs/en/manage-claude/rate-limits-api)
- [API and data retention - Claude Docs](https://platform.claude.com/docs/en/manage-claude/api-and-data-retention)
- [litellm Batches Documentation](https://docs.litellm.ai/docs/batches)
- [Anthropic SDK Python - Batches API](https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/resources/beta/messages/batches.py)
- [Anthropic Batches support + inline requests feature · BerriAI/litellm · Discussion #17280](https://github.com/BerriAI/litellm/discussions/17280)
