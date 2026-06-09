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
