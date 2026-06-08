# Google Gemini / Vertex AI Batch Prediction Research

## Executive Summary

Gemini batch prediction offers a **flat 50% discount** across all models, regardless of batch size or number of requests—the discount is **not tied to volume**. The `vertex_ai/` model prefix is required for batch operations in litellm; the direct `gemini/` prefix (using `GEMINI_API_KEY`) does not support batch processing. Batch jobs target 24-hour completion with no formal SLA, making them suitable for cost-sensitive, non-urgent workloads.

---

## 1. Pricing & Discount Model

### Discount Structure

- **Flat 50% discount** on all billable tokens (input and output)
- Applies uniformly across all Gemini models (2.5 Flash, 2.5 Pro, 3, etc.)
- **Not tied to batch size, number of requests, or volume** — the discount is applied per token, not per job
  
**Example (Gemini 2.5 Flash on Vertex AI):**
- Standard interactive: $0.30 input / $2.50 output per 1M tokens
- Batch mode: $0.15 input / $1.25 output per 1M tokens

**Example (Gemini 2.5 Pro on Vertex AI):**
- Standard interactive: $1.25 input / $10.00 output per 1M tokens
- Batch mode: $0.625 input / $5.00 output per 1M tokens

### Discount Precedence

- If using **context caching** with high cache hit rates (90% discount), the cache discount **takes precedence over the batch discount**; they do not stack
- Batch discount applies to all non-cached tokens in a batch request

**Source:** [Gemini Developer API pricing](https://ai.google.dev/gemini-api/docs/pricing), [Vertex AI pricing](https://cloud.google.com/vertex-ai/pricing)

### Known Billing Issues

As of the research date, some users have reported being charged at standard interactive rates instead of batch discounts despite using the batch API correctly. This appears to be an account-specific or model-specific billing issue rather than a universal problem. If you observe discrepancies, contact Google Cloud Billing Support.

**Source:** [Batch API charged at regular API prices discussion](https://discuss.ai.google.dev/t/batch-api-charged-at-regular-api-prices-instead-of-discounted-batch-pricing/109621)

---

## 2. Turnaround Time & SLA

### Service Level Objective (SLO)

- **Target completion window:** 24 hours
- **Typical reality:** Many jobs complete significantly faster (hours) depending on job size and system load
- **No formal SLA:** Batch inference is explicitly excluded from Gemini Enterprise's Service Level Agreement

### Use Case Implications

Batch mode is designed for:
- Large-scale, non-urgent workloads (data preprocessing, evaluations, bulk inference)
- Scenarios where 24-hour turnaround is acceptable
- NOT suitable for latency-sensitive or real-time applications

**Source:** [Batch API docs](https://ai.google.dev/gemini-api/docs/batch-api), [Gemini Enterprise SLA](https://cloud.google.com/terms/gemini-enterprise/sla)

---

## 3. API Mechanics & Request Flow

### Two API Paths

#### A. Gemini Developer API (ai.google.dev)
- Endpoint: Direct REST API to Google's Gemini service
- Authentication: `GEMINI_API_KEY` (environment variable)
- Batch support: **Limited** — create and retrieve batch operations, but **litellm does NOT support batch through the `gemini/` prefix**
- Use case: Simple, key-based access; suited for individual developers and prototyping

#### B. Vertex AI (cloud.google.com)
- Endpoint: Vertex AI's batchPredictionJobs REST API
- Authentication: GCP service account (via `GOOGLE_APPLICATION_CREDENTIALS`)
- Batch support: **Full** — complete batch workflow with file management, job creation, status polling, and result retrieval
- Use case: Enterprise deployments, full GCP integration, cost tracking

**Key distinction:** Batch processing in production typically routes through Vertex AI, not the direct Gemini API.

### Batch Request Workflow (Vertex AI)

#### Step 1: Prepare JSONL File
```
Each line = JSON object with user-defined key + GenerateContentRequest
{
  "custom_id": "request-id-1",
  "request": {
    "model": "gemini-2.5-flash",
    "contents": [{"role": "user", "parts": [{"text": "What is 2+2?"}]}]
  }
}
```

#### Step 2: Upload File
- Via litellm: `files.create(file_path, purpose="batch", custom_llm_provider="vertex_ai")`
- File storage: Stored in configured GCS bucket
- Maximum file size: **2 GB**

#### Step 3: Create Batch Job
- Via litellm: `batches.create(input_file_id)`
- Begins validation phase immediately

#### Step 4: Poll Batch Status
- Via litellm: `batches.retrieve(batch_id)`
- Status transitions: `validating` → `in_progress` → `completed`
- Poll frequency: Recommended every few minutes for larger batches

#### Step 5: Retrieve Results
- Via litellm: `files.content(output_file_id)`
- Returns JSONL with same structure as input, but with `result` field containing `ContentResponse`

**Source:** [Batch API docs](https://ai.google.dev/gemini-api/docs/batch-api), [Vertex AI batch prediction docs](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-gemini), [litellm Vertex Batch docs](https://docs.litellm.ai/docs/providers/vertex_batch)

---

## 4. Input Formats & Limits

### Input Methods

| Method | Max Size | Use Case |
|--------|----------|----------|
| **Inline requests** | < 20 MB | Small batches, quick testing |
| **JSONL files** | 2 GB max | Large-scale batch jobs |

### Request Structure

Both inline and JSONL use the standard `GenerateContentRequest` format:
```json
{
  "model": "gemini-2.5-flash",
  "contents": [...],
  "generation_config": {...},
  "system_instruction": "..."
}
```

### Batch Job Limits

- No documented hard limit on requests per batch, but very large batches (millions of requests) should be split for faster intermediate results
- Individual requests must conform to standard API rate limits (input/output token limits per model)

**Source:** [File input methods docs](https://ai.google.dev/gemini-api/docs/file-input-methods), [Batch API docs](https://ai.google.dev/gemini-api/docs/batch-api)

---

## 5. Authentication & Authorization

### Gemini API Path (Direct, Limited Batch Support)

**Environment variable:**
```bash
export GEMINI_API_KEY="<your-api-key>"
```

**Limitations in litellm:**
- Model prefix: `gemini/gemini-2.5-flash`
- **Batch operations NOT supported** through litellm
- Use direct REST client (e.g., `google.generativeai` library) for batch if using Gemini API key directly

### Vertex AI Path (Recommended for Batch)

**Service account setup:**
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
export VERTEXAI_PROJECT="your-gcp-project"
export VERTEXAI_LOCATION="us-central1"  # or your region
export GCS_BUCKET_NAME="your-batch-storage-bucket"
```

**GCS bucket permissions:**
- Service account must have `storage.buckets.get`, `storage.objects.create`, `storage.objects.get` on the batch GCS bucket
- This is where batch input/output files are stored

**litellm model prefix:** `vertex_ai/gemini-2.5-flash`

**Source:** [litellm Vertex AI docs](https://docs.litellm.ai/docs/providers/vertex), [litellm Vertex Batch docs](https://docs.litellm.ai/docs/providers/vertex_batch)

---

## 6. litellm Batch Support

### Supported Providers & Prefixes

| Provider | Model Prefix | Batch Support | Auth |
|----------|--------------|---|---|
| **Vertex AI** | `vertex_ai/` | ✅ Full | GCP service account |
| **Gemini API** | `gemini/` | ❌ No (in litellm) | `GEMINI_API_KEY` |

### Available Methods (litellm)

All methods use the OpenAI-compatible client interface:

```python
from litellm import OpenAI

client = OpenAI(
    api_key="anything",  # ignored
    base_url="http://localhost:8000"  # litellm proxy
)

# Upload batch file
response = client.files.create(
    file=open("batch.jsonl", "rb"),
    purpose="batch",
    custom_llm_provider="vertex_ai"
)
input_file_id = response.id

# Create batch job
batch = client.batches.create(
    input_file_id=input_file_id,
    endpoint="/batches",
    custom_llm_provider="vertex_ai"
)
batch_id = batch.id

# Poll status
status = client.batches.retrieve(
    batch_id=batch_id,
    custom_llm_provider="vertex_ai"
)

# Get results
output = client.files.content(
    file_id=status.output_file_id,
    custom_llm_provider="vertex_ai"
)
```

### Cost Tracking

litellm automatically tracks costs for Vertex AI Gemini models using the correct pricing tier based on `response.usageMetadata.trafficType` (standard or batch). No manual configuration required.

**Source:** [litellm Vertex Batch docs](https://docs.litellm.ai/docs/providers/vertex_batch), [litellm Vertex AI docs](https://docs.litellm.ai/docs/providers/vertex)

---

## 7. Quotas & System Limits

### Per-Batch Limits
- **Max input file size:** 2 GB
- **Max requests per batch:** Not formally documented; recommend splitting very large batches (millions of requests) for faster intermediate processing

### Per-Request Limits
- Standard model token limits apply (Gemini 2.5 Flash: 8k context)
- Rate limits follow standard interactive API quotas

### GCS Storage Limits
- Batch jobs store input/output in Cloud Storage (GCS)
- Ensure sufficient bucket quota and permissions

**Source:** [Vertex AI quotas and limits](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/quotas)

---

## 8. Key Findings & Summary

### Discount Model (User's Critical Question)

✅ **Confirmed:** The Gemini batch discount is **NOT tied to batch size or number of requests per job**. It is a **flat 50% per-token discount** that applies regardless of whether you submit 1 request or 1 million requests in a single batch job.

### Batch Support in litellm

✅ **Confirmed:** litellm supports Gemini batch processing **only via the `vertex_ai/` model prefix** using Vertex AI's batch API. The direct `gemini/` prefix (which uses `GEMINI_API_KEY`) does **not** have batch support in litellm. For batch workloads, you must use Vertex AI with GCP service account authentication.

### Two-Path Architecture

1. **Gemini API (ai.google.dev + GEMINI_API_KEY):** Simple, key-based, but limited batch support in client libraries
2. **Vertex AI (cloud.google.com + GCP service account):** Full-featured batch workflow, cost tracking, enterprise-grade, required for litellm batch operations

---

## 9. Batch Data Storage, Retention & Cleanup

### Gemini API Path (ai.google.dev + GEMINI_API_KEY)

#### Where Data Is Stored
- Input and output files are stored in **Google-managed Files API storage**, not in your GCS bucket
- Storage is ephemeral and tied to the Files API quota (20 GB per project maximum)

#### Retention & Auto-Deletion
- **Files automatically expire and are deleted after 48 hours** ([Files API documentation](https://ai.google.dev/gemini-api/docs/files))
- No manual cleanup required; Google handles deletion automatically
- During the 48-hour window, you can retrieve file metadata and content via `files.content()`
- After 48 hours, files are no longer accessible

#### Storage Cost
- **Free**: The Files API is available at no cost in all regions where the Gemini API is available ([Gemini API Pricing](https://ai.google.dev/gemini-api/docs/pricing))
- No storage charges regardless of file size (up to 2 GB per file, 20 GB aggregate per project)

#### Cleanup Recommendation
- **No explicit cleanup needed** for batch operations; the 48-hour auto-expiry is sufficient
- Optional: You can manually delete files earlier using `files.delete()` if desired to free quota space
- **Best practice for developers**: Retrieve results promptly after batch completion (typically within 24 hours) before files auto-expire

---

### Vertex AI Path (cloud.google.com + GCP Service Account)

#### Where Data Is Stored
- Input and output files are stored in **Google Cloud Storage (GCS) buckets that you specify and own** ([Batch Predict with Gemini using GCS data](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/samples/generativeaionvertexai-batch-predict-gemini-createjob-gcs))
- You provide the GCS bucket path (e.g., `gs://your-bucket/input.jsonl`)
- Output results are written to a GCS location you specify (e.g., `gs://your-bucket/output/`)
- Files persist in your bucket indefinitely until you delete them

#### Retention & Auto-Deletion
- **No automatic deletion**: Input and output files remain in your GCS bucket indefinitely
- Google does not clean up batch-related files after job completion
- Data persists at standard GCS retention rates until you manually delete it
- Important consideration: If you enable Cloud Storage soft delete, deleted objects are retained in soft-delete state, potentially increasing storage costs; **Google recommends disabling soft delete for cost optimization** ([Batch predictions documentation](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/maas/capabilities/batch-prediction))

#### Storage Cost
- **User pays standard GCS rates** for all batch input/output data stored in the bucket
- Costs include:
  - **Storage cost**: Charged per GB per month based on GCS storage class
  - **Operation costs**: Reads, writes, and deletes are charged separately
- GCS billing is to the GCP project that owns the bucket, visible on the project's Cloud Billing invoice
- **No automatic cleanup means no automatic cost reduction**: Forgotten files continue accruing storage charges

#### Cleanup Recommendation
- **Explicit cleanup is essential** for cost control, especially for large batch jobs or frequent runs
- After retrieving results, explicitly delete input files: `gsutil rm gs://your-bucket/input.jsonl`
- Delete output files once processed: `gsutil rm gs://your-bucket/output/results.jsonl` (or archive if needed)
- For library implementations (like litellm): **Add an optional cleanup hook** that deletes GCS objects after batch job completion and result retrieval
- **Best practice for library authors**: Provide a `cleanup_after_batch` flag (default: False for safety) that explicitly removes files after retrieving results, preventing unexpected GCS charges

---

### Cost and Cleanup Summary

| Path | Storage Location | Auto-Deleted? | Cost | Manual Cleanup Needed? |
|------|------------------|---|---|---|
| **Gemini API** | Google-managed Files API | ✅ Yes (48h) | Free | No |
| **Vertex AI** | User's GCS bucket | ❌ No | User pays GCS rates | ✅ Yes (recommended) |

---

## 10. Batch Cancellation: Cost, Latency, Tooling

### Cancellation API & State Transitions

**API Method:** `projects.locations.batchPredictionJobs.cancel`
- **Endpoint:** POST `https://{service-endpoint}/v1/{name}:cancel`
- **Parameter:** Resource name in format `projects/{project}/locations/{location}/batchPredictionJobs/{batchPredictionJob}`
- **Request body:** Empty JSON object
- **Response:** Empty JSON object on success

**State Transitions:**
- On cancellation request: job transitions to `JOB_STATE_CANCELLING` 
- On successful cancellation: job transitions to `CANCELLED`
- **Important caveat:** Server makes "best effort" to cancel — **success is not guaranteed**. If a job completes before cancellation takes effect, it will complete normally rather than being cancelled. Clients must check job state via `GetBatchPredictionJob` to verify actual cancellation status.

**Sources:** [Method: batchPredictionJobs.cancel](https://docs.cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.batchPredictionJobs/cancel), [Cancel a batch prediction job sample](https://docs.cloud.google.com/vertex-ai/docs/samples/aiplatform-cancel-batch-prediction-job-sample)

### Cancellation Latency

**Documented latency:** No specific latency bound documented. The cancellation is **asynchronous**, meaning the API returns immediately, but the job's actual state transition to CANCELLED may take additional time server-side.

**Practical guidance:** Clients should poll `GetBatchPredictionJob` periodically (every few minutes) to detect state transitions. No SLA or guaranteed upper bound is published.

**Implication for "submit then cancel if deadline nears" pattern:** You must account for unknown cancellation latency when deciding when to issue the cancellation request — a job might complete between your cancellation request and the state transition actually taking effect.

**Sources:** [Method: batchPredictionJobs.cancel](https://docs.cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.batchPredictionJobs/cancel)

### Billing on Cancellation

**Pay-for-what-was-processed model:** You are **charged only for completed requests** (those returning HTTP 200 status). Unprocessed or failed requests (4xx, 5xx) are **not charged**.

**On cancellation:** When you cancel a batch job, any remaining work is cancelled, and you pay **only for completed requests** that have already been processed up to the cancellation point. You are **not charged** for:
- Requests queued but not yet processed
- Requests that were in-flight when cancellation took effect
- Any "speculative" or failed processing

**Per-token model:** For Gemini models on Vertex batch, billing is per-token (input and output tokens) for successfully completed requests. The 50% batch discount applies to all tokens in completed requests.

**No cancellation fee:** Google Cloud does not charge an additional fee or penalty for cancelling a job.

**Implications:**
- ✅ Cancellation itself is free (no API fee)
- ✅ You pay only for what was actually processed
- ⚠️ If a job nearly completes (due to latency in cancellation taking effect), you may pay for nearly all requests — cancellation is not guaranteed to stop processing mid-batch
- ⚠️ For a deadline-based pattern, you must cancel *significantly* before your deadline to account for unknown cancellation latency

**Sources:** [Batch inference with Gemini](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-gemini), [Batch Predictions Overview](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/maas/capabilities/batch-prediction)

### Abandoned Job Cost

If you submit a batch job and let it run to completion (never cancel), you pay for **all successfully processed requests** at the per-token rate (with batch discount applied). Incomplete jobs that exceed 24 hours are automatically cancelled by Google, and you still pay only for what was completed before the auto-cancellation.

### GCS Storage Cost During Batch Wait

**Storage location (Vertex AI path):** Input and output files are stored in your own GCS bucket that you provide.

**Cost:** You pay standard GCS storage rates (typically **$0.020/GB/month** for Standard tier in US regions) for the duration the files sit in the bucket.

**Negligibility for small batches:** For a small batch (e.g., 5 MB JSONL input + 10 MB results = 15 MB total) queued for 1 day:
- 15 MB = 0.015 GB
- Cost = 0.015 GB × ($0.020 / month) × (1 day / 30 days) ≈ **$0.00001** (0.001 cents)
- This is **negligible** for short turnaround times (hours to a day)

**For longer waits:** Storage cost becomes material only if files remain in the bucket for weeks or months without cleanup. For example, 100 GB sitting for 30 days = $2 storage cost, plus GCS operation costs (writes, reads, deletes).

**Best practice:** Delete input/output files from GCS after job completion and result retrieval to avoid unnecessary storage charges.

**Sources:** [Batch Data Storage section (earlier in this document)](#9-batch-data-storage-retention--cleanup), [Google Cloud Storage Pricing](https://cloud.google.com/storage/pricing)

### litellm Support for Batch Cancellation

**Status:** litellm **has** a `cancel_batch()` method, **but it does NOT support the `vertex_ai` provider**.

**Current support:** The `cancel_batch()` function signature includes a type hint for `vertex_ai`, but the implementation only handles `"openai"` and `"azure"` providers (OpenAI-compatible APIs).

```python
# litellm signature (supports openai, azure only)
cancel_batch(
    batch_id: str,
    model: str = "",
    custom_llm_provider: Union[Literal["openai", "azure", "vertex_ai"], str] = "openai",
    ...
)
# Note: vertex_ai is in the type hint but not implemented
```

**Workaround:** To cancel a Vertex AI batch via litellm, you have two options:

1. **Use litellm's passthrough endpoint:**
   ```bash
   curl -X POST "http://localhost:4000/v1/projects/your-project/locations/us-central1/batchPredictionJobs/job-id:cancel" \
     -H "Authorization: Bearer your-api-key"
   ```
   This bypasses the `cancel_batch()` method and hits the Vertex AI REST API directly through litellm's proxy.

2. **Use Vertex AI SDK directly:**
   ```python
   from google.cloud import aiplatform
   client = aiplatform.gapic.JobServiceClient()
   client.cancel_batch_prediction_job(name="projects/.../batchPredictionJobs/job-id")
   ```
   Drop down to the native Vertex AI client library to cancel jobs.

**Sources:** [litellm batches/main.py](https://github.com/BerriAI/litellm/blob/main/litellm/batches/main.py), [litellm Vertex Batch APIs](https://docs.litellm.ai/docs/providers/vertex_batch), [liteLLM passthrough endpoint](https://docs.litellm.ai/docs/vertex_batch_passthrough)

### Summary: Is "Submit Then Cancel If Deadline Nears" Essentially Free?

**Answer: YES, with important caveats on timing and cost structure.**

**Costs:**
- ✅ **Cancellation itself:** Free (no API fee or penalty)
- ✅ **GCS storage:** Negligible for small batches held for hours/days (sub-cent)
- ⚠️ **Processed tokens:** You pay for what was already processed before cancellation took effect — this is not free, but it's bounded by "actual work done"

**Latency & Guarantee:**
- ⚠️ **No latency bound:** Cancellation is asynchronous with no documented completion time. A job can race to completion before cancellation takes effect, resulting in full processing charges.
- ⚠️ **Best-effort only:** Server may not cancel the job if it completes first.

**Recommendation for deadline-based pattern:**
1. **Set a cancellation deadline well ahead of your hard deadline** (e.g., if you need results by T+12h, issue cancellation request at T+10h to account for cancellation latency)
2. **Poll job state after cancellation** to verify state transitioned to CANCELLED
3. **Accept that you'll pay for tokens processed up to the actual cancellation point** — the cost is not zero, but it is bounded by "work in progress"
4. **For litellm users:** Use Vertex AI SDK directly or the passthrough endpoint; `litellm.cancel_batch()` does not support vertex_ai

**Bottom line:** The pattern is operationally viable if you can tolerate uncertainty in cancellation latency and accept paying for partial batch processing. The storage cost is negligible for typical batch durations. The main risk is misjudging when to cancel (job might complete anyway), not hidden charges.

**Sources:** [Method: batchPredictionJobs.cancel](https://docs.cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.batchPredictionJobs/cancel), [Batch inference with Gemini](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-gemini), [litellm batches/main.py](https://github.com/BerriAI/litellm/blob/main/litellm/batches/main.py)

---

## References

- [Gemini API Batch API Documentation](https://ai.google.dev/gemini-api/docs/batch-api)
- [Gemini API Pricing](https://ai.google.dev/gemini-api/docs/pricing)
- [Gemini Files API Documentation](https://ai.google.dev/gemini-api/docs/files)
- [Vertex AI Batch Prediction with Gemini](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-gemini)
- [Vertex AI Batch Predict with Gemini using GCS](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/samples/generativeaionvertexai-batch-predict-gemini-createjob-gcs)
- [Batch Predictions Overview](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/maas/capabilities/batch-prediction)
- [Vertex AI Pricing](https://cloud.google.com/vertex-ai/pricing)
- [litellm Vertex Batch APIs](https://docs.litellm.ai/docs/providers/vertex_batch)
- [litellm Vertex AI Provider Documentation](https://docs.litellm.ai/docs/providers/vertex)
- [Gemini Enterprise SLA](https://cloud.google.com/terms/gemini-enterprise/sla)
- [Zero Data Retention (ZDR) Documentation](https://ai.google.dev/gemini-api/docs/zdr)
- [Batch API Pricing Issue Discussion](https://discuss.ai.google.dev/t/batch-api-charged-at-regular-api-prices-instead-of-discounted-batch-pricing/109621)
- [Retention Period Discussion - Batch Mode](https://discuss.ai.google.dev/t/retention-period-and-storage-region-of-input-output-data-in-gemini-api-batch-mode/112404)
- [Method: batchPredictionJobs.cancel](https://docs.cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.batchPredictionJobs/cancel)
- [Cancel a batch prediction job sample](https://docs.cloud.google.com/vertex-ai/docs/samples/aiplatform-cancel-batch-prediction-job-sample)
- [Google Cloud Storage Pricing](https://cloud.google.com/storage/pricing)
- [litellm batches/main.py source](https://github.com/BerriAI/litellm/blob/main/litellm/batches/main.py)
