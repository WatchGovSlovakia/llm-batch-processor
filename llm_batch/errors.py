"""Exceptions for llm_batch."""


class BatchJobFailed(Exception):
    """Raised when a batch job reaches a terminal failure state.

    `execute()` on a FAILED job re-raises this rather than silently retrying.
    """

    def __init__(self, message: str, batch_job_id: str | None = None):
        super().__init__(message)
        self.batch_job_id = batch_job_id


class TransientProviderError(Exception):
    """A retryable provider error (network blip, timeout, HTTP 429/5xx).

    Raised by the provider layer on submit. The Job leaves its state unchanged
    so the next run retries; nothing is marked FAILED.
    """


class OpenRouterAPIError(Exception):
    """A non-2xx answer from the OpenRouter Batch API.

    `status_code` drives the classification in `providers._is_transient`:
    429/5xx are retried, while 401/402/4xx stop the run and leave the job's
    state unchanged (e.g. top up credits, then re-run).
    """

    def __init__(self, status_code: int | None, message: str):
        super().__init__(f"OpenRouter {status_code}: {message}")
        self.status_code = status_code
