import json

import pytest

from llm_batch.storage.base import Storage, StorageScanner
from llm_batch.storage.postgres import PostgresStorage


class FakeCursor:
    """Minimal cursor over an in-memory dict table keyed by (namespace, key)."""

    def __init__(self, store):
        self._store = store
        self._result = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=()):
        sql_norm = " ".join(sql.split()).lower()
        if sql_norm.startswith("create table"):
            self._result = None
        elif sql_norm.startswith("insert"):
            namespace, key, data_json, state = params
            self._store[(namespace, key)] = (data_json, state)
            self._result = None
        elif "where namespace = %s and key = %s" in sql_norm and sql_norm.startswith("select"):
            namespace, key = params
            row = self._store.get((namespace, key))
            self._result = [(row[0],)] if row else []
        elif "key <> %s" in sql_norm and sql_norm.startswith("select"):
            namespace, exclude = params
            self._result = [(d,) for (ns, k), (d, s) in self._store.items()
                            if ns == namespace and k != exclude]
        elif sql_norm.startswith("delete"):
            namespace, key = params
            self._store.pop((namespace, key), None)
            self._result = None

    def fetchone(self):
        return self._result[0] if self._result else None

    def fetchall(self):
        return list(self._result or [])


class FakeConn:
    def __init__(self):
        self.store = {}
        self.committed = 0

    def cursor(self):
        return FakeCursor(self.store)

    def commit(self):
        self.committed += 1


@pytest.fixture
def conn():
    return FakeConn()


def test_for_job_round_trip(conn):
    backend = PostgresStorage(conn, namespace="doc1")
    handle = backend.for_job("ocr")
    assert handle.load() is False
    handle.save({"key": "ocr", "state": "sent", "model": "gpt-4o"})

    fresh = PostgresStorage(conn, namespace="doc1").for_job("ocr")
    assert fresh.load() is True
    assert fresh.get("state") == "sent"
    assert fresh.snapshot()["model"] == "gpt-4o"


def test_get_default(conn):
    handle = PostgresStorage(conn, namespace="d").for_job("missing")
    handle.load()
    assert handle.get("state", "not_sent") == "not_sent"


def test_finish_deletes_row(conn):
    handle = PostgresStorage(conn, namespace="d").for_job("j")
    handle.save({"key": "j", "state": "completed"})
    handle.finish()
    assert PostgresStorage(conn, namespace="d").for_job("j").load() is False


def test_iter_jobs_scoped_to_namespace(conn):
    PostgresStorage(conn, namespace="doc1").for_job("a").save({"key": "a", "state": "sent"})
    PostgresStorage(conn, namespace="doc1").for_job("b").save({"key": "b", "state": "completed"})
    PostgresStorage(conn, namespace="doc2").for_job("a").save({"key": "a", "state": "sent"})

    handles = list(PostgresStorage(conn, namespace="doc1").iter_jobs())
    assert sorted(h.get("state") for h in handles) == ["completed", "sent"]
    assert isinstance(handles[0], Storage)


def test_meta_uses_reserved_key(conn):
    backend = PostgresStorage(conn, namespace="doc1")
    backend.for_meta().save({"pipeline_started_at": "t"})
    # meta is NOT returned by iter_jobs
    assert list(backend.iter_jobs()) == []
    assert backend.for_meta().load() is True


def test_is_scanner(conn):
    assert isinstance(PostgresStorage(conn, namespace="d"), StorageScanner)
