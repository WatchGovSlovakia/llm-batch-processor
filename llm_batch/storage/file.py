"""FileStorage — one JSON file per job, atomic and git-friendly.

A `FileStorage(dir)` is a directory-scoped backend. It mints per-job handles
(`for_job`, `for_meta`) and enumerates them (`iter_jobs`). Each handle owns one
JSON file path and implements the per-job `Storage` protocol.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Iterator

META_FILENAME = "pipeline.json"


class _FileJobHandle:
    """Per-job Storage handle bound to a single JSON file."""

    def __init__(self, path: str):
        self._path = path
        self._data: dict = {}

    def load(self) -> bool:
        if not os.path.exists(self._path):
            return False
        with open(self._path, encoding="utf-8") as f:
            self._data = json.load(f)
        return True

    def save(self, data: dict) -> None:
        self._data = data
        directory = os.path.dirname(os.path.abspath(self._path))
        os.makedirs(directory, exist_ok=True)
        # Deterministic, merge-friendly serialization for clean git diffs.
        payload = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        # Atomic write: temp file in the same dir + os.replace.
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
            os.replace(tmp, self._path)
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def snapshot(self) -> dict:
        return dict(self._data)

    def finish(self) -> None:
        if os.path.exists(self._path):
            os.remove(self._path)


class FileStorage:
    """Directory-scoped backend that mints per-job handles and scans them."""

    def __init__(self, directory: str):
        self._dir = directory

    def for_job(self, key: str) -> _FileJobHandle:
        # Auto-load so a freshly-minted handle reflects any persisted state
        # (a re-read is cheap and keeps callers from forgetting to load()).
        handle = _FileJobHandle(os.path.join(self._dir, f"{key}.json"))
        handle.load()
        return handle

    def for_meta(self) -> _FileJobHandle:
        return _FileJobHandle(os.path.join(self._dir, META_FILENAME))

    def iter_jobs(self) -> Iterator[_FileJobHandle]:
        """Walk the tree, yielding a loaded handle for every job file.

        Skips pipeline.json meta files and in-flight temp files.
        """
        for root, _dirs, files in os.walk(self._dir):
            for name in sorted(files):
                if name == META_FILENAME or not name.endswith(".json"):
                    continue
                if name.startswith(".tmp-"):
                    continue
                handle = _FileJobHandle(os.path.join(root, name))
                handle.load()
                yield handle
