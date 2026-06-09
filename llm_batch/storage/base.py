"""Storage protocols.

`Storage` is the per-job handle a Job holds (load/save/get/finish).
`StorageScanner` is the backend-level enumeration the Poller needs.

A backend object (FileStorage, PostgresStorage) is neither, exactly — it is a
*factory* that mints per-job Storage handles via `for_job(key)` / `for_meta()`
and enumerates them via `iter_jobs()`.
"""

from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable


@runtime_checkable
class Storage(Protocol):
    """Per-job state handle."""

    def load(self) -> bool:
        """Load existing state. Returns True if state existed."""
        ...

    def save(self, data: dict) -> None:
        """Persist the full job state dict."""
        ...

    def get(self, key: str, default=None):
        """Read one field from the loaded state."""
        ...

    def snapshot(self) -> dict:
        """Return a copy of the full loaded state dict."""
        ...

    def finish(self) -> None:
        """Delete / mark this job finished."""
        ...


@runtime_checkable
class StorageScanner(Protocol):
    """Backend-level enumeration used by the Poller."""

    def iter_jobs(self) -> Iterator[Storage]:
        """Yield a loaded per-job Storage handle for every persisted job."""
        ...
