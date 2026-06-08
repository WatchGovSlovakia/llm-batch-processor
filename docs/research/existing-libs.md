# Existing LLM Batch Processing Libraries: Research Report

## Executive Summary

**Does an equivalent off-the-shelf solution exist?** No. While several libraries and frameworks handle individual aspects of LLM batch processing, **none combine all seven design points**: transparent interactive↔batch switching, per-job persistent state with file/database backends, deadline-aware fallback, multi-step resumable pipelines, provider-agnostic batch support (OpenAI + Anthropic + Vertex), centralized job polling, and provider-side data cleanup.

**Build vs. Adopt recommendation:** **Build, but borrow selectively.** The litellm foundation is solid (existing batch wrappers across providers), but you will need custom layers for state persistence, deadline logic, and transparent mode switching. LangGraph's checkpointing patterns are worth studying for the persistence layer.

---

## Comparison Table: Our 7 Design Points vs. Existing Libraries

| Library | Multi-Provider Batch | Persistent Job State | Idempotent Job.execute() | Deadline-Aware Fallback | Resumable Pipelines | Central Poller | Provider Data Cleanup |
|---------|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| **Our Library (proposed)** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **litellm** | ✓ | ✗ | ✗ | ✗ | ✗ | Partial (polling loop, no state) | ✗ |
| **openai-batch** (Parasail) | Partial (OpenAI, HF) | ✓ (resume by ID) | ✗ | ✗ | ✗ | ✗ | ✗ |
| **openbatch** | ✓ (OpenAI) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **anthropic-batch-kit** | ✓ (Anthropic) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **ai-batch** (batchata) | Partial (Anthropic, OpenAI planned) | ✓ (JSON state) | ✗ | ✗ | ✗ | ✗ | ✗ |
| **LangChain .batch()** | ✓ (via LiteLLM) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **LangGraph** | ✓ (via LiteLLM) | ✓ (checkpointer) | ✗ | ✗ | ✓ | ✗ | ✗ |
| **Ray Data** | ✓ (via Ray LLM) | ✓ (checkpoint) | ✗ | ✗ | ✓ | ✗ | ✗ |
| **llmq** | ✓ (self-hosted) | ✓ (durable queue) | ✗ | ✗ | Partial (multi-stage) | ✗ | ✗ |
| **Mirascope** | ✓ (via async) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **instructor** | ✓ (via async) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **Marvin** | ✓ (OpenAI) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **Prefect** | ✗ (not LLM-specific) | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ |
| **Dagster** | ✗ (not LLM-specific) | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ |

---

## Per-Library Analysis

### 1. **litellm** 
**Links:** [Docs](https://docs.litellm.ai/docs/batches) | [GitHub](https://github.com/BerriAI/litellm)

**What it does:**  
An AI gateway supporting 100+ LLM APIs with a unified OpenAI-compatible interface. Includes batch endpoints for OpenAI, Azure, Vertex, Bedrock, vLLM, and more.

**Covers:**
- Multi-provider batch support ✓
- Built-in polling loop (timeout/retry logic) - partial

**Gaps:**
- No persistent job state; polling is in-memory with configurable MAX_WAIT_TIME (300s) and POLL_INTERVAL
- No deadline-aware fallback; no seamless sync→batch switching
- No multi-step pipeline abstraction
- No idempotent job execution across restarts
- No data cleanup

**Verdict:**  
Excellent foundation. Use litellm as the low-level batch submission layer, but you must wrap it with your own state persistence and deadline logic.

---

### 2. **openai-batch** (Parasail)
**Links:** [PyPI](https://pypi.org/project/openai-batch/) | [Blog: openbatch](https://www.daniel-gomm.com/blog/2025/openbatch/)

**What it does:**  
Helper library simplifying OpenAI Batch API workflows: JSONL prep, job submission, polling, result download, and request↔response mapping via `custom_id`.

**Covers:**
- Handles OpenAI and HuggingFace models (limited multi-provider)
- Resume by batch ID (can reconnect to interrupted jobs)

**Gaps:**
- Resume is manual (call with `batch_id`), not idempotent
- No persistent state file or database backend
- No deadline fallback or transparent batch↔sync
- Single-provider focus

**Verdict:**  
Good reference for batch submission patterns, but does not handle state persistence or multi-provider orchestration.

---

### 3. **openbatch**
**Links:** [GitHub/Blog](https://www.daniel-gomm.com/blog/2025/openbatch/) | [PyPI (inferred)](https://pypi.org/project/openbatch/)

**What it does:**  
Lightweight Python library mirroring the OpenAI Python client API for easier batch file generation. Drop-in replacement for structured outputs.

**Covers:**
- OpenAI API coverage
- Works as familiar interface

**Gaps:**
- Focused on batch file prep, not orchestration
- No state persistence, polling, or deadline logic
- No multi-provider support

**Verdict:**  
Convenience layer only; not relevant for job orchestration.

---

### 4. **anthropic-batch-kit**
**Links:** [GitHub (MukundaKatta)](https://github.com/MukundaKatta/anthropic-batch-kit)

**What it does:**  
Wraps the Anthropic Python SDK's batch endpoints with a simpler submit→poll→retrieve pattern. Handles JSONL parsing.

**Covers:**
- Anthropic Message Batches (10,000 requests per batch, 50% discount)
- Polling convenience

**Gaps:**
- Single-provider (Anthropic only)
- No persistent state across restarts
- No deadline fallback or transparent sync↔batch

**Verdict:**  
Thin wrapper; suitable for Anthropic-only workloads but not part of a multi-provider solution.

---

### 5. **ai-batch** / **batchata**
**Links:** [GitHub](https://github.com/agamm/ai-batch)

**What it does:**  
Simplifies batch processing with optional structured output for Anthropic and OpenAI (OpenAI support planned). Includes cost tracking and auto-resume via JSON state files.

**Covers:**
- Anthropic batch processing (native)
- JSON state file for resumability
- Cost tracking

**Gaps:**
- OpenAI support is planned, not implemented
- State resumption is JSON-file-based but lacks deadline awareness
- No transparent batch↔sync switching
- No centralized poller; no deadline fallback

**Verdict:**  
Closest single-provider match to your use case, but incomplete multi-provider support and no deadline logic.

---

### 6. **LangChain** (.batch() / LCEL)
**Links:** [Docs: Batch Processing](https://apxml.com/courses/langchain-production-llm/chapter-6-optimizing-scaling-langchain/batch-processing-offline) | [GitHub](https://github.com/langchain-ai/langchain)

**What it does:**  
Expression language (LCEL) with .batch(), .map(), and async methods (abatch, ainvoke) for parallel LLM calls. Supports max_concurrency control.

**Covers:**
- Multi-provider (delegates to LiteLLM or native clients)
- Concurrent/parallel execution

**Gaps:**
- Batch here = concurrent in-process execution, not async batch APIs (OpenAI, Anthropic)
- No persistent state or checkpointing at the batch call level
- No deadline fallback
- No transparent sync↔batch mode

**Verdict:**  
Orthogonal use case (concurrency, not batch submission). Useful for request parallelization but not batch job orchestration.

---

### 7. **LangGraph**
**Links:** [Docs: Persistence](https://deepchecks.com/langchain-vs-langgraph/) | [GitHub](https://github.com/langchain-ai/langgraph)

**What it does:**  
Stateful graph-based workflow engine for agents. Includes checkpointing (SQLite, PostgreSQL, Redis) to persist state between node executions. Resumes from last checkpoint on failure.

**Covers:**
- Resumable multi-step pipelines ✓
- Persistent state with multiple backends ✓
- Fault tolerance via checkpoints ✓

**Gaps:**
- Not LLM-batch-specific; no awareness of batch APIs
- No deadline-aware fallback to sync mode
- No centralized job poller
- No transparent batch↔sync switching (designed for agent workflows, not cost optimization)

**Verdict:**  
Excellent reference for the checkpointing/state persistence layer. Borrow the checkpoint pattern but would need significant adaptation for batch-specific use cases.

---

### 8. **Ray Data**
**Links:** [Docs: Batch Inference](https://docs.ray.io/en/latest/data/batch_inference.html) | [Anyscale Blog](https://www.anyscale.com/blog/batch-llm-inference-announcement)

**What it does:**  
Distributed batch inference framework. Supports checkpoint-based resume: if a job fails, re-running skips already-processed records. Built on Ray for fault tolerance and GPU scaling.

**Covers:**
- Resumable pipelines via checkpoints ✓
- Multi-GPU / distributed execution
- Partial state persistence (checkpoint store)

**Gaps:**
- Not transparently switchable between batch and sync (designed for scale, not cost switching)
- No deadline-aware fallback
- No centralized poller for external batch APIs (OpenAI, Anthropic) — Ray itself is the executor
- Requires Ray infrastructure; not lightweight

**Verdict:**  
Excellent for large-scale distributed inference but not for transparent cost-switching or external batch API orchestration. Study checkpoint patterns.

---

### 9. **llmq**
**Links:** [GitHub](https://github.com/iPieter/llmq) | [Blog](https://pieter.ai/introducing-llmq/)

**What it does:**  
Distributed LLM inference scheduler. Submits jobs to RabbitMQ workers running vLLM. Supports multi-stage pipelines, horizontal scaling, and durable result queues.

**Covers:**
- Multi-stage pipelines ✓
- Resumable result retrieval (durable queue) ✓
- Distributed execution

**Gaps:**
- Designed for self-hosted models (vLLM), not external batch APIs (OpenAI, Anthropic)
- No deadline-aware fallback
- No transparent batch↔sync (no interactive mode)
- Infrastructure-heavy (RabbitMQ, worker cluster)

**Verdict:**  
Wrong use case; for managed LLM services (OpenAI, Anthropic), not self-hosted models.

---

### 10. **Mirascope**
**Links:** [Docs: Async](https://mirascope.com/learn/async/) | [GitHub](https://github.com/Mirascope/mirascope)

**What it does:**  
LLM "anti-framework" for structured prompting, chaining, and async execution. Supports asyncio.gather() for concurrent LLM calls.

**Covers:**
- Multi-provider (via plugin architecture)
- Async concurrency

**Gaps:**
- Concurrency, not batch API submission
- No persistent state, polling, or deadline awareness
- No transparent sync↔batch

**Verdict:**  
Useful for prompt engineering and async concurrency but orthogonal to batch job orchestration.

---

### 11. **instructor**
**Links:** [Docs](https://python.useinstructor.com/) | [GitHub](https://github.com/instructor-ai/instructor) | [Blog: Async](https://python.useinstructor.com/blog/2023/11/13/learn-async/)

**What it does:**  
Structured output library using OpenAI function calling or Anthropic tool use. Supports async/batch processing with asyncio.gather(), asyncio.as_completed(), and semaphore-based rate limiting.

**Covers:**
- Multi-provider structured outputs
- Async concurrency patterns

**Gaps:**
- Structured output focus, not batch API orchestration
- Async concurrency, not persistent batch job state
- No deadline fallback or transparent mode switching

**Verdict:**  
Orthogonal; good for response validation but not batch job management.

---

### 12. **Marvin** (AI Engineering Framework)
**Links:** [GitHub](https://github.com/PrefectHQ/marvin)

**What it does:**  
LLM framework by Prefect emphasizing reliability. Supports async operations and model classification/extraction tasks.

**Covers:**
- OpenAI-focused
- Async execution

**Gaps:**
- No batch API support
- No persistent state
- Limited multi-provider support

**Verdict:**  
Focused on prompt engineering, not batch orchestration.

---

### 13. **Prefect** (Workflow Engine)
**Links:** [Docs](https://docs.prefect.io/) | [GitHub](https://github.com/PrefectHQ/prefect)

**What it does:**  
General-purpose workflow/task orchestration engine. Supports persistent task state, retries, and cron scheduling. Not LLM-specific.

**Covers:**
- Resumable workflows ✓
- Persistent state ✓
- Central scheduler/poller ✓
- Idempotent task execution ✓

**Gaps:**
- No LLM-specific batch API awareness
- No deadline-aware fallback between batch and interactive
- No transparent batch↔sync switching
- Requires additional wrapper layer for LLM batch APIs

**Verdict:**  
Could be used as the orchestration foundation, but would still need custom LLM-batch wrappers for deadline logic and transparent mode switching.

---

### 14. **Dagster** (Data Orchestration)
**Links:** [Docs](https://docs.dagster.io/) | [GitHub](https://github.com/dagster-io/dagster)

**What it does:**  
Data pipeline orchestration engine with op/asset DAGs, state persistence, and cron scheduling. Supports Celery executor for distributed execution.

**Covers:**
- Resumable pipelines ✓
- Persistent state ✓
- Central scheduler ✓
- Idempotent execution ✓
- Multi-step workflows ✓

**Gaps:**
- No LLM-specific batch awareness
- No deadline-aware fallback
- No transparent batch↔sync switching
- Would require wrapping with custom batch logic

**Verdict:**  
Similar to Prefect; excellent workflow foundation but lacks LLM-batch-specific features.

---

## Gaps Our Library Fills

1. **Transparent Interactive ↔ Batch Switching**
   - No existing library offers seamless, idempotent Job.execute() that automatically chooses batch or interactive based on deadline.
   - All competitors require explicit batch job submission or async task spawning, not automatic fallback.

2. **Multi-Provider Batch Unification**
   - litellm provides low-level wrappers but no unified job state, deadline logic, or provider data cleanup.
   - Single-provider libraries (openai-batch, anthropic-batch-kit) don't solve the multi-provider problem.

3. **Persistent Job State with Deadline Awareness**
   - LangGraph and Ray Data have checkpointing but no deadline-based mode switching.
   - Prefect and Dagster are general-purpose and don't include LLM batch-specific deadline logic.

4. **Idempotent Execution Across Restarts**
   - Only Prefect and Dagster (general-purpose) and LangGraph (agent-specific) support true idempotency.
   - No existing library combines this with transparent batch↔sync and LLM-batch-specific state.

5. **Provider-Side Data Cleanup**
   - No library explicitly handles deletion of OpenAI input files or Vertex GCS objects after batch completion.
   - This is a cost/quota concern specific to managed batch APIs.

6. **Multi-Step Resumable Pipelines with Unified Batch Logic**
   - Prefect, Dagster, and Ray Data support multi-step pipelines, but with no awareness of batch API state or deadline fallback.
   - llmq and LangGraph support multi-step workflows but lack multi-provider or deadline features.

---

## Ideas Worth Borrowing

1. **LangGraph's Checkpointing Pattern**
   - Use SQLite/PostgreSQL checkpointer abstraction for the job state backend.
   - Study BaseCheckpointSaver interface for pluggable storage.

2. **litellm's Multi-Provider Abstraction**
   - Use litellm's existing batch wrappers for OpenAI, Anthropic, Vertex, Bedrock.
   - Extend with unified job ID mapping and state persistence.

3. **Ray Data's Resumable Checkpoint Config**
   - Checkpoint-on-completion pattern is solid; adapt for per-job JSON state.
   - Built-in fault tolerance model is worth studying.

4. **Dagster/Prefect's Central Scheduler**
   - Cron-based job poller architecture is clean; use as reference for your Poller.

5. **ai-batch's JSON State Persistence**
   - Simple, git-friendly state format; borrow the idea (even if implementation is basic).

6. **instructor/Mirascope's Async Patterns**
   - For the interactive (sync) path within your library, leverage async/await patterns for concurrency.

---

## Conclusion & Build vs. Adopt Decision

**Should you build?** **Yes.** 

No existing library combines transparent interactive↔batch switching, multi-provider unification, deadline-aware fallback, persistent resumable state, and provider data cleanup. The closest candidates are:

- **litellm** (batch submission) + **LangGraph** (state persistence) + **Prefect/Dagster** (orchestration) would require significant custom integration.
- **ai-batch** (state persistence) but single-provider and missing deadline logic.
- **Ray Data** (pipelines + checkpoints) but heavyweight and not designed for external batch API orchestration.

**Recommended approach:**

1. **Foundation:** Use litellm for low-level batch API calls (OpenAI, Anthropic, Vertex, Bedrock).
2. **State Persistence:** Build a custom checkpointer inspired by LangGraph, using JSON files (or Postgres) to store per-job state.
3. **Deadline Logic:** Implement the transparent fallback: if time to deadline < batch SLA, use interactive; else submit batch.
4. **Central Poller:** Build a simple cron-friendly poller (or reuse Prefect/Dagster for ops/assets if you adopt them).
5. **Data Cleanup:** Add post-completion hooks to delete OpenAI files and Vertex objects.
6. **Multi-Step Pipelines:** Layer on top of state persistence with step markers (e.g., "step_1_job_id", "step_2_job_id").

**License & Attribution Concerns:** Be sure to review litellm's license (MIT) and any other libraries you integrate. Your library can be built on these foundations without licensing conflicts.

---

## Sources

- [LiteLLM Batches Documentation](https://docs.litellm.ai/docs/batches)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm/)
- [OpenAI Batch API Guide](https://developers.openai.com/api/docs/guides/batch)
- [OpenAI-Batch PyPI](https://pypi.org/project/openai-batch/)
- [OpenBatch Blog](https://www.daniel-gomm.com/blog/2025/openbatch/)
- [Anthropic Message Batches API](https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/resources/beta/messages/batches.py)
- [ai-batch GitHub](https://github.com/agamm/ai-batch)
- [LangChain Batch Processing](https://apxml.com/courses/langchain-production-llm/chapter-6-optimizing-scaling-langchain/batch-processing-offline)
- [LangGraph Persistence & Checkpointing](https://deepchecks.com/langchain-vs-langgraph/)
- [Ray Data Batch Inference](https://docs.ray.io/en/latest/data/batch_inference.html)
- [Anyscale Batch LLM Inference](https://www.anyscale.com/blog/batch-llm-inference-announcement)
- [llmq GitHub](https://github.com/iPieter/llmq)
- [llmq Blog](https://pieter.ai/introducing-llmq/)
- [Mirascope Async Guide](https://mirascope.com/learn/async/)
- [Instructor Documentation](https://python.useinstructor.com/)
- [Prefect Documentation](https://docs.prefect.io/)
- [Dagster Documentation](https://docs.dagster.io/)
