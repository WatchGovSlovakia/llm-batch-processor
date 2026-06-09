"""PostgresStorage — JSONB-backed per-job state, namespaced for many pipelines.

Required schema (created on demand by `ensure_schema`):

    CREATE TABLE IF NOT EXISTS llm_batch_jobs (
        namespace  TEXT NOT NULL,
        key        TEXT NOT NULL,
        data       JSONB NOT NULL,
        state      TEXT,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (namespace, key)
    );
"""

from __future__ import annotations

import json
from typing import Iterator

TABLE = "llm_batch_jobs"
# Reserved key for the pipeline meta row, kept out of iter_jobs().
_META_KEY = "__pipeline_meta__"


class _PgJobHandle:
    def __init__(self, conn, namespace: str, key: str):
        self._conn = conn
        self._namespace = namespace
        self._key = key
        self._data: dict = {}

    def load(self) -> bool:
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT data FROM {TABLE} WHERE namespace = %s AND key = %s",
                (self._namespace, self._key))
            row = cur.fetchone()
        if not row:
            return False
        raw = row[0]
        self._data = raw if isinstance(raw, dict) else json.loads(raw)
        return True

    def save(self, data: dict) -> None:
        self._data = data
        payload = json.dumps(data, sort_keys=True)
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {TABLE} (namespace, key, data, state)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (namespace, key) DO UPDATE SET
                    data = EXCLUDED.data,
                    state = EXCLUDED.state,
                    updated_at = NOW()
                """,
                (self._namespace, self._key, payload, data.get("state")))
        self._conn.commit()

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def snapshot(self) -> dict:
        return dict(self._data)

    def finish(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {TABLE} WHERE namespace = %s AND key = %s",
                (self._namespace, self._key))
        self._conn.commit()


class PostgresStorage:
    """Namespace-scoped backend that mints per-job handles over one table."""

    def __init__(self, conn, namespace: str):
        self._conn = conn
        self._namespace = namespace

    def ensure_schema(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {TABLE} (
                    namespace  TEXT NOT NULL,
                    key        TEXT NOT NULL,
                    data       JSONB NOT NULL,
                    state      TEXT,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (namespace, key)
                )
                """)
        self._conn.commit()

    def for_job(self, key: str) -> _PgJobHandle:
        return _PgJobHandle(self._conn, self._namespace, key)

    def for_meta(self) -> _PgJobHandle:
        return _PgJobHandle(self._conn, self._namespace, _META_KEY)

    def iter_jobs(self) -> Iterator[_PgJobHandle]:
        # Exclude the reserved meta row at the SQL level so iter_jobs only ever
        # yields real jobs.
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT data FROM {TABLE} WHERE namespace = %s AND key <> %s",
                (self._namespace, _META_KEY))
            rows = cur.fetchall()
        for (raw,) in rows:
            data = raw if isinstance(raw, dict) else json.loads(raw)
            handle = _PgJobHandle(self._conn, self._namespace, data.get("key"))
            handle._data = data
            yield handle
