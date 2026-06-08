# OpenAI Batch API Research

## Executive Summary

OpenAI's Batch API is a dedicated asynchronous processing service offering a **flat 50% cost reduction** across all tokens (input and output), regardless of batch size—meaning the discount is per-token and not conditional on batch volume. Batches process within a 24-hour completion window (SLA) with typical completions much faster. The API requires JSONL-formatted input with custom IDs for result mapping, supports up to 50,000 requests per batch at 200 MB max file size, and has a separate rate limit pool from standard APIs. LiteLLM provides clean SDK wrappers (`acreate_batch`, `aretrieve_batch`, `afile_content`) with the custom_llm_provider value `"openai"`.

---

## 1. Pricing & Discount Model

### Cost Reduction: 50% Flat Discount

The Batch API offers a **50% discount on both input and output tokens** compared to standard synchronous API pricing. This discount is **flat and model-agnostic**—it applies uniformly regardless of batch size.

**Key Finding (Answers Core Question):**
- The discount is **per-token, not batch-size-dependent**. Adding more requests to a single batch does not change the per-token rate; all tokens in all requests get the 50% reduction.
- This means there's no minimum batch size to achieve the discount, and putting 100 requests vs. 1,000 requests in one batch applies the same proportional savings.

### Pricing Examples (Source: [Pricing | OpenAI API](https://developers.openai.com/api/docs/pricing))

**Text Models:**
- **gpt-5.5**: Input $5.00 → $2.50 per 1M tokens; Output $15.00 → $7.50 per 1M tokens (50% reduction)
- **gpt-5.4**: Input $2.50 → $1.25 per 1M tokens; Output $10.00 → $5.00 per 1M tokens
- **gpt-5.4-mini**: Input $0.75 → $0.375 per 1M tokens; Output $3.00 → $1.50 per 1M tokens

**Video Generation (Sora):**
- **sora-2**: $0.10 → $0.05 per second
- **sora-2-pro 1080p**: $0.70 → $0.35 per second

**Embeddings & Other Models:**
The same 50% reduction applies across all supported endpoints (embeddings, moderations, image generation, etc.).

### Billing Structure

- Billing is per-token, calculated after batch completion.
- You are charged for completed requests only. If a batch expires (fails to complete within 24h), you are charged only for work that completed before expiration.
- No additional per-batch or per-request fees.

---

## 2. Turnaround & SLA

### Completion Window: 24 Hours

- **SLA Guarantee**: Each batch completes within **24 hours** or the remaining work is cancelled.
- **Typical Turnaround**: Often faster than 24h; actual completion depends on system load.
- **Status at Expiration**: If not completed within 24h, the batch status becomes `"expired"` and any completed work is returned (and charged).

### Batch Status Lifecycle

Batches transition through these states during processing:

1. **validating** – Input file and requests are being validated.
2. **in_progress** – Batch is actively processing requests.
3. **finalizing** – Requests are complete; results are being assembled.
4. **completed** – All requests processed; results available for download.
5. **failed** – Batch failed due to validation or processing errors (e.g., invalid endpoint, authentication issues).
6. **expired** – Batch exceeded 24-hour completion window; partial results returned.

### Result Availability

Output files remain accessible for **30 days** after batch completion. (Source: [Batch API | OpenAI API](https://developers.openai.com/api/docs/guides/batch))

---

## 3. API Mechanics & Workflow

### Three Core Operations

The Batch API workflow centers on three main endpoints and supporting file operations:

#### 3.1 Upload Input File (Files API)

**Endpoint:** `POST /v1/files`

**Purpose:** Upload a JSONL file with batch requests.

**Parameters:**
- `file` (required): The JSONL file containing batch requests.
- `purpose` (required): Must be `"batch"` (not `"fine-tune"`).

**Response:** Returns a file object with an `id` field (e.g., `file-abc123...`).

#### 3.2 Create Batch

**Endpoint:** `POST /v1/batches`

**Parameters:**
- `input_file_id` (required): The file ID from the upload step.
- `endpoint` (required): The API endpoint to call for each request (e.g., `"/v1/chat/completions"`).
- `completion_window` (required): Currently only `"24h"` is supported.

**Request Example:**
```json
{
  "input_file_id": "file-abc123...",
  "endpoint": "/v1/chat/completions",
  "completion_window": "24h"
}
```

**Response:** Returns a batch object with:
- `id` – Unique batch identifier.
- `object` – Always `"batch"`.
- `status` – Initial status (usually `"validating"`).
- `input_file_id` – References the input file.
- `output_file_id` – (Set after completion) References the output file.
- `error_file_id` – (If errors exist) References a file with failed request details.
- `created_at`, `started_at`, `completed_at` – Timestamps.

#### 3.3 Retrieve Batch & Poll Status

**Endpoint:** `GET /v1/batches/{batch_id}`

**Response:** Returns the batch object with current status and result file IDs.

**Polling Pattern:** Check status periodically (e.g., every 30–60 seconds) until status is `"completed"`, `"failed"`, or `"expired"`.

#### 3.4 Download Results (Files API)

**Endpoint:** `GET /v1/files/{file_id}/content`

**Purpose:** Download the output JSONL file once batch completes.

---

### Input JSONL Format

Each line is a JSON object representing one API request:

```jsonl
{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-5.4", "messages": [{"role": "user", "content": "What is 2+2?"}], "temperature": 1}}
{"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-5.4", "messages": [{"role": "user", "content": "Explain photosynthesis"}], "temperature": 0.7}}
```

**Required Fields Per Request:**
- `custom_id` (string, unique): Your identifier for this request; appears in results for mapping.
- `method` (string): HTTP method—typically `"POST"` (required for Chat Completions, Embeddings, etc.).
- `url` (string): API endpoint path (e.g., `"/v1/chat/completions"`).
- `body` (object): Payload matching the endpoint's standard API parameters (e.g., `model`, `messages`, `temperature`).

**Notes:**
- Each line must be valid JSON (no comments, no trailing commas).
- `custom_id` must be unique within the batch and used to match results.
- Results in the output file are not guaranteed to be in the same order as input—always use `custom_id` for mapping.

---

### Output JSONL Format

One line per successful request:

```jsonl
{"custom_id": "request-1", "result": {"status_code": 200, "request_id": "req-...", "body": {"id": "chatcmpl-...", "object": "chat.completion", "created": 1234567890, "model": "gpt-5.4", "choices": [...], "usage": {...}}}}
{"custom_id": "request-2", "result": {"status_code": 200, "request_id": "req-...", "body": {...}}}
```

**Error Handling:**
- Successful requests: `"status_code": 200` in the `result` object.
- Failed requests (e.g., 4xx/5xx errors): Appear in a separate error file referenced by `error_file_id`.
- Error file format includes `custom_id` and error details for matching to input.

---

### Supported Endpoints

The Batch API supports the following OpenAI endpoints:

- `/v1/chat/completions` – Chat Completions (including vision, JSON mode, function calling)
- `/v1/completions` – Legacy Completions
- `/v1/embeddings` – Text Embeddings
- `/v1/moderations` – Content Moderation
- `/v1/images/generations` – DALL·E Image Generation
- `/v1/images/edits` – Image Editing
- `/v1/videos` – Video Generation (Sora, if available)
- `/v1/responses` – Responses API (reusable prompts and caching)

(Source: [Batch API | OpenAI API](https://developers.openai.com/api/docs/guides/batch))

---

## 4. Limits & Constraints

### Per-Batch Limits

- **Max Requests per Batch:** 50,000 requests per batch.
- **Max Input File Size:** 200 MB.
- **Max Embeddings per Batch:** If using `/v1/embeddings`, max 50,000 embedding inputs across all requests (consistent with general per-batch limit).

### Rate Limits

- **Batch Creation Rate:** Up to 2,000 batches created per hour.
- **Enqueued Prompt Tokens:** Each model has a limit on the maximum number of enqueued prompt tokens for batch processing (viewable in [Platform Settings](https://platform.openai.com/account/billing/limits)).
- **Output Tokens:** No per-batch or per-hour limits on output tokens.
- **Separate Pool:** Batch API rate limits are **completely separate** from standard per-model rate limits, meaning batch requests don't count against your interactive API quotas.

(Source: [Batch API | OpenAI API](https://developers.openai.com/api/docs/guides/batch), [Rate limits | OpenAI API](https://developers.openai.com/api/docs/guides/rate-limits))

---

## 5. Authentication & Environment Variables

### API Key

The Batch API uses the same OpenAI API key as standard endpoints.

**Environment Variable:**
```bash
export OPENAI_API_KEY="sk-..."
```

Or set in code:
```python
import openai
openai.api_key = "sk-..."
```

### Project-Level Organization

If using OpenAI's project-based access controls:
- Requests include the standard `OpenAI-Organization` header (if you belong to multiple organizations).
- Project context is handled via the API key's associated project.

---

## 6. LiteLLM Integration

### Supported Custom LLM Provider

LiteLLM supports Batch API operations for OpenAI via:

```python
custom_llm_provider = "openai"
```

### Core Functions

LiteLLM provides async-first wrappers for batch operations:

#### 6.1 Create File

```python
import litellm

file_response = await litellm.acreate_file(
    file=open("batch_requests.jsonl", "rb"),
    purpose="batch",
    custom_llm_provider="openai"
)
file_id = file_response.id
```

#### 6.2 Create Batch

```python
batch_response = await litellm.acreate_batch(
    input_file_id=file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    custom_llm_provider="openai"
)
batch_id = batch_response.id
```

#### 6.3 Retrieve Batch

```python
batch_status = await litellm.aretrieve_batch(
    batch_id=batch_id,
    custom_llm_provider="openai"
)
print(batch_status.status)  # "validating", "in_progress", "completed", etc.
```

#### 6.4 Get File Content

```python
output = await litellm.afile_content(
    file_id=batch_status.output_file_id,
    custom_llm_provider="openai"
)
# Returns file content as string or bytes
```

#### 6.5 List Batches

```python
batches = await litellm.list_batches(
    custom_llm_provider="openai"
)
```

### Multi-Account Support

LiteLLM encodes model and provider information in file/batch IDs, enabling automatic credential routing without explicit parameter passing in multi-tenant scenarios. Use `config.yaml` to specify credentials per model.

(Source: [/batches | liteLLM](https://docs.litellm.ai/docs/batches), [Batching Completion() | liteLLM](https://docs.litellm.ai/docs/completion/batching))

---

## 7. Typical Python Workflow for Building a Batch Library

Here's a high-level pattern for a reusable Python library:

```python
import asyncio
import json
from typing import List, Dict, Any
import litellm

class OpenAIBatchClient:
    def __init__(self, api_key: str = None):
        if api_key:
            litellm.api_key = api_key
    
    async def submit_batch(
        self, 
        requests: List[Dict[str, Any]],
        endpoint: str = "/v1/chat/completions"
    ) -> str:
        """
        Submit a batch and return batch_id.
        
        Args:
            requests: List of dicts with custom_id, method, url, body
            endpoint: OpenAI API endpoint
            
        Returns:
            batch_id: Unique identifier for polling
        """
        # Create JSONL file
        jsonl_content = "\n".join(json.dumps(req) for req in requests)
        
        # Upload file
        file_response = await litellm.acreate_file(
            file=("batch.jsonl", jsonl_content.encode()),
            purpose="batch",
            custom_llm_provider="openai"
        )
        file_id = file_response.id
        
        # Create batch
        batch_response = await litellm.acreate_batch(
            input_file_id=file_id,
            endpoint=endpoint,
            completion_window="24h",
            custom_llm_provider="openai"
        )
        return batch_response.id
    
    async def poll_batch(self, batch_id: str, poll_interval: int = 30) -> Dict:
        """Poll batch status until completion."""
        while True:
            batch = await litellm.aretrieve_batch(
                batch_id=batch_id,
                custom_llm_provider="openai"
            )
            if batch.status in ["completed", "failed", "expired"]:
                return batch
            await asyncio.sleep(poll_interval)
    
    async def get_results(self, batch_id: str) -> List[Dict]:
        """Retrieve and parse output file."""
        batch = await litellm.aretrieve_batch(
            batch_id=batch_id,
            custom_llm_provider="openai"
        )
        
        if not batch.output_file_id:
            return []
        
        content = await litellm.afile_content(
            file_id=batch.output_file_id,
            custom_llm_provider="openai"
        )
        
        results = []
        for line in content.decode().strip().split("\n"):
            results.append(json.loads(line))
        return results
```

---

## 8. Key Considerations & Gotchas

1. **No Batch Size Discount Variability**: The 50% discount is flat. There's no penalty for small batches or bonus for large batches—all tokens are discounted equally.

2. **24h SLA is Hard**: The completion window is fixed at 24 hours. OpenAI does not currently offer 12h, 48h, or variable windows.

3. **Result Ordering**: Output JSONL is **not guaranteed to match input order**. Always rely on `custom_id` for mapping results to requests.

4. **Separate Rate Limits**: Batch API has its own rate limit pool, so you can run batches without impacting interactive API quotas.

5. **File Retention**: Output files expire after 30 days. Download results promptly or store them externally.

6. **Error Files**: Failed requests appear in a separate error file. Always check `error_file_id` after batch completion.

7. **Custom ID Uniqueness**: Ensure `custom_id` is unique within the batch. Duplicates may cause unpredictable behavior.

---

## References

- [Batch API | OpenAI API](https://developers.openai.com/api/docs/guides/batch) – Official guide and endpoint documentation
- [Batch API FAQ | OpenAI Help Center](https://help.openai.com/en/articles/9197833-batch-api-faq) – FAQs on limits, pricing, and operations
- [Pricing | OpenAI API](https://developers.openai.com/api/docs/pricing) – Current Batch API pricing (50% discount)
- [Rate limits | OpenAI API](https://developers.openai.com/api/docs/guides/rate-limits) – Rate limit details for Batch API
- [Batch Processing with the Batch API](https://developers.openai.com/cookbook/examples/batch_processing) – Code examples and workflow walkthrough
- [/batches | liteLLM](https://docs.litellm.ai/docs/batches) – LiteLLM Batch API wrapper documentation
- [Batching Completion() | liteLLM](https://docs.litellm.ai/docs/completion/batching) – LiteLLM batch completion patterns

---

---

## Batch Data Storage, Retention & Cleanup

### 1. Storage Cost

**Finding:** OpenAI **does not charge a per-file or per-GB-per-day storage fee** for batch input files, output files, or error files. The OpenAI pricing documentation ([Pricing | OpenAI API](https://developers.openai.com/api/docs/pricing)) lists no storage charges for Files API or batch-related file operations. 

**Important Note:** OpenAI does charge **$0.10/GB/day for vector storage** in the context of Assistants File Search and vector stores ([Assistants File Search | OpenAI API](https://platform.openai.com/docs/assistants/tools/file-search)), but this is a separate feature unrelated to batch job files. Batch files themselves incur no storage costs.

**Account Storage Quota:** Each project has a **2.5 TB file storage limit** across all uploaded files ([File storage quota errors for API users | OpenAI Help Center](https://help.openai.com/en/articles/8724843-file-storage-quota-errors-for-api-users)). This is a quota on the total number of files stored, not a cost-based limit—but exceeding the quota will block new file uploads.

---

### 2. Retention & Auto-Deletion Policy

**Output Files:** Output files are **automatically deleted 30 days after batch completion**. ([Batch API | OpenAI API](https://developers.openai.com/api/docs/guides/batch))

**Input Files:** Files uploaded with `purpose="batch"` have a **default expiration of 30 days**. ([File retention behavior](https://developers.openai.com/api/docs/api-reference/files)) You can customize this expiration window using the `expires_after` parameter when uploading, specifying a duration between **1 hour (3600 seconds) and 30 days (2,592,000 seconds)**.

**Error Files:** The documentation does **not explicitly specify a retention policy for error files** generated during batch processing. They are likely subject to the same 30-day default as other batch-related files, but this should be verified with OpenAI support if strict compliance is required.

**Batch Object Metadata:** The batch object itself (status, metadata, file IDs) appears to be retained indefinitely based on the ability to retrieve batch status via `GET /v1/batches/{batch_id}` at any time, but OpenAI's documentation does not specify a hard retention window for batch metadata.

---

### 3. Explicit Cleanup: Delete or Don't Delete?

**Manual Deletion Available:** Yes. You can explicitly delete files using `DELETE /v1/files/{file_id}` ([Delete file | OpenAI API Reference](https://developers.openai.com/api/reference/resources/files/methods/delete)). 

**Is Cleanup Recommended?**
- **Quota Management:** OpenAI strongly recommends implementing cleanup jobs to avoid hitting the 2.5 TB project storage quota. Files count toward this quota even after batch completion, so **deleting unused batch files is best practice to avoid quota exhaustion** ([File storage quota errors](https://help.openai.com/en/articles/8724843-file-storage-quota-errors-for-api-users)).
- **Cost Impact:** Cleanup is **not required for cost reasons** (no storage fees), only for quota management.
- **Data Retention Policy:** Organizations with Zero Data Retention (ZDR) enabled should note that ZDR does **not apply to the Batch API** ([Data controls in the OpenAI platform](https://developers.openai.com/api/docs/guides/your-data)). Batch files follow standard retention regardless of ZDR settings.

**Recommended Cleanup Pattern:**
1. After downloading batch results, explicitly delete the input file: `DELETE /v1/files/{input_file_id}`.
2. After archiving or processing the output, delete the output file: `DELETE /v1/files/{output_file_id}`.
3. If an error file was generated, delete it: `DELETE /v1/files/{error_file_id}`.
4. This ensures the 2.5 TB quota is preserved for future batch submissions.

---

### 4. Summary & Recommendation for Library Authors

**Does it cost money?** No—batch files have **zero storage fees**. Costs are solely for token consumption.

**Is cleanup required?** Not for billing, but **yes for practical quota management**. The 2.5 TB/project limit can be exhausted if files accumulate across many batches.

**Recommendation:** **Add an explicit cleanup step** (`delete_batch_files()` or similar) to your batch processing library. After retrieving results and verifying success:
1. Delete the input file immediately (no longer needed).
2. Delete the output and error files after storing results externally (e.g., database, cloud storage).
3. Provide users with the option to skip deletion (e.g., for debugging) but default to cleanup.

This prevents quota exhaustion, improves account hygiene, and follows OpenAI's implicit best practice of managing file storage proactively.

---

---

## Batch Cancellation: Cost, Latency, Tooling

### 1. Can You Cancel an In-Progress Batch?

**Yes.** OpenAI's Batch API supports batch cancellation via a dedicated endpoint:

**Endpoint:** `POST /v1/batches/{batch_id}/cancel`

**State Transitions:** When you initiate cancellation, the batch transitions through two states:
1. `cancelling` – Initial state after cancellation is requested.
2. `cancelled` – Final state after all in-flight requests are safely terminated.

According to the [Batch API documentation](https://developers.openai.com/docs/api-reference/batch/cancel), "The batch will be in status `cancelling` for up to 10 minutes, before changing to `cancelled`." This means **the batch can remain in the `cancelling` state for up to 10 minutes** while OpenAI gracefully stops in-flight API calls.

(Source: [Cancel a batch | OpenAI API Reference](https://developers.openai.com/docs/api-reference/batch/cancel))

---

### 2. Cancellation Latency

**Up to 10 minutes.** According to OpenAI's official documentation, after you submit a cancellation request, the batch status transitions to `cancelling` and remains in that state for **up to 10 minutes** while in-flight requests are terminated. After that period, the status becomes `cancelled`.

**Practical Implication:** If you submit a batch optimistically and want to cancel it to re-run interactively before a deadline, you need to account for up to a 10-minute latency window. If your deadline is tight (e.g., <10 minutes), batch cancellation may not meet your latency requirements.

(Source: [Batch API | OpenAI API](https://developers.openai.com/api/docs/guides/batch))

---

### 3. Billing When Cancelling

**Principle: You pay only for completed work.** When you cancel a batch:

- **Completed requests are charged.** Any requests that finished before cancellation is initiated will be billed at the 50% batch discount rate.
- **In-flight/cancelled requests are NOT charged.** Requests that were in progress or not yet started when cancellation was requested will not incur any token charges.
- **No cancellation fee.** There is no additional charge, penalty, or fee for initiating cancellation—you are only charged for tokens from successfully completed requests.

This is explicitly stated in the OpenAI documentation: "You will be charged for tokens consumed from any completed requests." Unfinished requests receive an error and do not incur charges.

(Source: [Batch API | OpenAI API](https://developers.openai.com/api/docs/guides/batch))

---

### 4. Abandoned/Expired Batch Cost

**Same principle: pay for completed work only.** If you do not explicitly cancel a batch and it expires after 24 hours, the result is identical from a billing perspective:

- **Completed requests are charged.** Requests that finished before the 24-hour window closed are billed at the batch 50% discount rate.
- **Expired requests are NOT charged.** Requests that did not finish within the 24-hour SLA window receive an error ("This request could not be executed before the completion window expired") and are not billed.

**No cost difference between cancellation and expiration.** Whether you proactively cancel or let a batch expire naturally, you pay only for work that actually completed.

(Source: [Batch API | OpenAI API](https://developers.openai.com/api/docs/guides/batch))

---

### 5. Storage Cost During Wait

**Zero storage cost.** OpenAI **does not charge** a per-file or per-GB-per-day storage fee for batch input files, output files, or error files while they sit in queue or in-progress. 

Each project has a **2.5 TB file storage quota** (a limit on total stored files, not a cost-based quota), but quota exhaustion is the only practical concern—not billing. As documented in the research section [Batch Data Storage, Retention & Cleanup](#batch-data-storage-retention--cleanup), batch files incur no storage charges.

(Source: [Pricing | OpenAI API](https://developers.openai.com/api/docs/pricing); [File storage quota errors | OpenAI Help Center](https://help.openai.com/en/articles/8724843-file-storage-quota-errors-for-api-users))

---

### 6. LiteLLM Batch Cancellation Support

**Yes, fully supported.** LiteLLM exposes batch cancellation via two functions (sync and async) available at the top level:

#### Async Function (Recommended)

```python
import litellm

response = await litellm.acancel_batch(
    batch_id="batch-abc123...",
    custom_llm_provider="openai"  # or "azure", "vertex_ai"
)
```

**Signature:**
```python
async def acancel_batch(
    batch_id: str,
    model: Optional[str] = None,
    custom_llm_provider: Literal["openai", "azure", "vertex_ai"] = "openai",
    metadata: Optional[Dict[str, str]] = None,
    **kwargs
) -> CancelBatchResponse
```

#### Sync Function (Alternative)

```python
response = litellm.cancel_batch(
    batch_id="batch-abc123...",
    custom_llm_provider="openai"
)
```

**Signature:**
```python
def cancel_batch(
    batch_id: str,
    model: Optional[str] = None,
    custom_llm_provider: Union[Literal["openai", "azure", "vertex_ai"], str] = "openai",
    metadata: Optional[Dict[str, str]] = None,
    **kwargs
) -> CancelBatchResponse
```

**LiteLLM Equivalent:** Both functions execute `POST /v1/batches/{batch_id}/cancel` (as documented in the source: "LiteLLM Equivalent of POST https://api.openai.com/v1/batches/{batch_id}/cancel").

**Multi-Provider Support:** LiteLLM also supports batch cancellation for Azure and Vertex AI via the same interface—just change the `custom_llm_provider` parameter.

(Source: [litellm/batches/main.py](https://github.com/BerriAI/litellm/blob/main/litellm/batches/main.py), lines 898–1103; [LiteLLM Batches Documentation](https://docs.litellm.ai/docs/batches))

---

### Summary: Is Submit-Then-Cancel Essentially Free?

**Yes, with important caveats:**

1. **Cost:** Cancellation itself has no fee. You pay only for requests that completed before cancellation—any partially processed or queued requests are free. For a "submit then cancel immediately" pattern, your cost is negligible (roughly the cost of validation/early processing only).

2. **Latency:** Cancellation takes **up to 10 minutes** to fully take effect (time for the batch to transition from `cancelling` to `cancelled`). If your deadline is tighter than 10 minutes, this pattern may not work. However, you can query the batch status immediately after requesting cancellation and proceed with alternative logic once cancellation is confirmed.

3. **LiteLLM Support:** Yes, `litellm.acancel_batch()` and `litellm.cancel_batch()` are fully supported with simple function signatures. Use `custom_llm_provider="openai"` for the OpenAI Batch API.

**Practical Recommendation:** The pattern is suitable if your deadline buffer is >10 minutes. Combine it with polling (`aretrieve_batch()`) to detect cancellation completion and trigger fallback logic (e.g., interactive request processing) as soon as the batch is cancelled. This allows you to "optimistically submit, then gracefully exit if time runs out" with minimal cost penalty.

---

**Document Generated:** June 8, 2026  
**Research Scope:** OpenAI Batch API (Delayed/Asynchronous Processing) + Cancellation  
**Confidence Level:** High—based on official OpenAI documentation, OpenAI API Reference, and litellm source code
