"""llm_batch — transparent, resumable interactive/batch LLM processing.

Public API:
    Job, Pipeline, Poller            — core
    FileStorage, PostgresStorage     — storage backends
    BatchJobFailed, TransientProviderError, OpenRouterAPIError  — errors
"""

from llm_batch.errors import BatchJobFailed, OpenRouterAPIError, TransientProviderError
from llm_batch.job import Job
from llm_batch.pipeline import Pipeline
from llm_batch.poller import Poller
from llm_batch.storage.file import FileStorage
from llm_batch.storage.postgres import PostgresStorage

__all__ = [
    "Job",
    "Pipeline",
    "Poller",
    "FileStorage",
    "PostgresStorage",
    "BatchJobFailed",
    "TransientProviderError",
    "OpenRouterAPIError",
]
