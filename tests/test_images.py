"""Image hosting: data-URI rewrite, the Mongo and S3 hosts, and the Job wiring
that hosts images for OpenRouter batches and deletes them once the batch has ended."""

import base64
from datetime import datetime, timedelta, timezone

import pytest

from llm_batch import providers
from llm_batch.images import MongoImageHost, S3ImageHost, externalize_images
from llm_batch.job import Job
from llm_batch.poller import Poller
from llm_batch.storage.file import FileStorage
from tests.test_openrouter import FakeOpenRouter, _batch, _err, _result

mongomock = pytest.importorskip("mongomock")

MODEL = "openrouter/anthropic/claude-haiku-5.5"
PNG = b"\x89PNG\r\n\x1a\nfake"
PNG_URI = "data:image/png;base64," + base64.b64encode(PNG).decode()


class FakeHost:
    def __init__(self, fail_after=None):
        self.images = {}
        self.deleted = []
        self._fail_after = fail_after

    def put(self, data, content_type):
        if self._fail_after is not None and len(self.images) >= self._fail_after:
            raise RuntimeError("host down")
        image_id = f"img{len(self.images)}"
        self.images[image_id] = (data, content_type)
        return image_id, f"https://img.example/{image_id}"

    def delete(self, image_id):
        self.deleted.append(image_id)
        self.images.pop(image_id, None)

    def delete_orphans(self, live_ids, older_than_days):
        gone = [i for i in self.images if i not in live_ids]
        for i in gone:
            self.delete(i)
        return gone


def _image_msgs(*urls):
    return [{"role": "user", "content": [{"type": "text", "text": "read"}]
             + [{"type": "image_url", "image_url": {"url": u}} for u in urls]}]


# --- externalize_images ------------------------------------------------------------

def test_externalize_replaces_data_uris_and_keeps_the_input():
    host = FakeHost()
    original = [_image_msgs(PNG_URI, "https://already.public/x.png"),
                [{"role": "user", "content": [{"type": "image_url", "image_url": PNG_URI}]}],
                [{"role": "user", "content": "plain text"}]]

    out, ids = externalize_images(original, host)

    assert ids == ["img0"]  # the same image is stored once
    assert host.images["img0"] == (PNG, "image/png")
    assert out[0][0]["content"][1]["image_url"]["url"] == "https://img.example/img0"
    assert out[0][0]["content"][2]["image_url"]["url"] == "https://already.public/x.png"
    assert out[1][0]["content"][0]["image_url"] == "https://img.example/img0"
    assert out[2] == original[2]
    assert original[0][0]["content"][1]["image_url"]["url"] == PNG_URI  # not mutated


def test_externalize_deletes_stored_images_when_a_put_fails():
    host = FakeHost(fail_after=1)
    other = "data:image/jpeg;base64," + base64.b64encode(b"jpeg").decode()
    with pytest.raises(RuntimeError):
        externalize_images([_image_msgs(PNG_URI, other)], host)
    assert host.deleted == ["img0"]


# --- MongoImageHost ---------------------------------------------------------------

def test_mongo_host_stores_and_deletes():
    db = mongomock.MongoClient().db
    host = MongoImageHost(db, "https://wd.example/api/v1/llm-images/")

    image_id, url = host.put(PNG, "image/png")

    assert image_id.endswith(".png") and len(image_id) > 30
    assert url == f"https://wd.example/api/v1/llm-images/{image_id}"
    doc = db.llm_images.find_one({"_id": image_id})
    assert (doc["content_type"], bytes(doc["data"])) == ("image/png", PNG)

    host.delete(image_id)
    host.delete(image_id)  # missing: no error
    assert db.llm_images.count_documents({}) == 0


def test_mongo_host_rejects_oversized_images():
    host = MongoImageHost(mongomock.MongoClient().db, "https://wd.example/i")
    with pytest.raises(ValueError):
        host.put(b"x" * (MongoImageHost.MAX_BYTES + 1), "image/png")


def test_mongo_host_ttl_index():
    db = mongomock.MongoClient().db
    MongoImageHost(db, "https://wd.example/i").ensure_indexes(ttl_days=10)
    [ttl] = [i for i in db.llm_images.index_information().values() if i["key"] == [("created_at", 1)]]
    assert ttl["expireAfterSeconds"] == 864000


def test_mongo_host_deletes_old_orphans_only():
    db = mongomock.MongoClient().db
    host = MongoImageHost(db, "https://wd.example/i")
    old_live, _ = host.put(PNG, "image/png")
    old_orphan, _ = host.put(PNG, "image/png")
    fresh, _ = host.put(PNG, "image/png")
    long_ago = datetime.now(timezone.utc) - timedelta(days=5)
    db.llm_images.update_many({"_id": {"$in": [old_live, old_orphan]}},
                              {"$set": {"created_at": long_ago}})

    assert host.delete_orphans({old_live}, older_than_days=3) == [old_orphan]
    assert {d["_id"] for d in db.llm_images.find()} == {old_live, fresh}


# --- S3ImageHost ------------------------------------------------------------------

class FakeS3:
    def __init__(self):
        self.objects = {}

    def put_object(self, Bucket, Key, Body, ContentType):
        self.objects[(Bucket, Key)] = {"Body": Body, "ContentType": ContentType,
                                       "LastModified": datetime.now(timezone.utc)}

    def delete_object(self, Bucket, Key):
        self.objects.pop((Bucket, Key), None)

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        s3 = self

        class _Paginator:
            def paginate(self, Bucket, Prefix):
                yield {"Contents": [{"Key": k, "LastModified": v["LastModified"]}
                                    for (b, k), v in s3.objects.items()
                                    if b == Bucket and k.startswith(Prefix)]}
        return _Paginator()


def test_s3_host_puts_public_objects_and_sweeps_orphans():
    s3 = FakeS3()
    host = S3ImageHost(s3, "bucket", "https://asset.example")

    image_id, url = host.put(PNG, "image/png")
    assert url == f"https://asset.example/llm-images/{image_id}"
    assert s3.objects[("bucket", f"llm-images/{image_id}")]["ContentType"] == "image/png"

    orphan, _ = host.put(PNG, "image/png")
    for key in (f"llm-images/{image_id}", f"llm-images/{orphan}"):
        s3.objects[("bucket", key)]["LastModified"] -= timedelta(days=5)
    assert host.delete_orphans({image_id}, older_than_days=3) == [orphan]

    host.delete(image_id)
    assert s3.objects == {}


# --- Job wiring -------------------------------------------------------------------

@pytest.fixture
def openrouter(monkeypatch, litellm_mock):
    fake = FakeOpenRouter()
    monkeypatch.setattr(providers, "_openrouter_http", fake)
    return fake


def _job(tmp_path, host, model=MODEL, **kw):
    return Job(FileStorage(str(tmp_path)).for_job("k"), "k",
               messages=_image_msgs(PNG_URI), model=model, image_host=host, **kw)


def test_openrouter_batch_gets_hosted_urls_and_deletes_them_when_done(tmp_path, openrouter):
    host = FakeHost()
    openrouter.routes[("POST", "/batches")] = {"id": "batch-1", "status": "validating"}
    openrouter.routes[("DELETE", "/batches/batch-1")] = {}

    job = _job(tmp_path, host)
    assert job.execute() is None
    body = openrouter.calls[0][2]
    assert body["requests"][0]["body"]["messages"][0]["content"][1]["image_url"]["url"] \
        == "https://img.example/img0"
    assert job.state == Job.SENT
    assert host.images and not host.deleted

    openrouter.routes[("GET", "/batches/batch-1")] = _batch("completed", [_result("req-0", "ok")])
    # A later run (e.g. a cron re-run) resumes from storage with the same host.
    resumed = Job(FileStorage(str(tmp_path)).for_job("k"), "k", messages=_image_msgs(PNG_URI),
                  model=MODEL, image_host=host)
    assert resumed.execute() == "ok"
    assert host.deleted == ["img0"]


def test_failed_batch_deletes_hosted_images(tmp_path, openrouter):
    host = FakeHost()
    openrouter.routes[("POST", "/batches")] = {"id": "batch-1", "status": "validating"}
    openrouter.routes[("DELETE", "/batches/batch-1")] = {}
    openrouter.routes[("GET", "/batches/batch-1")] = _batch("expired")
    job = _job(tmp_path, host)
    job.execute()
    with pytest.raises(Exception):
        job.execute()
    assert job.state == Job.FAILED
    assert host.deleted == ["img0"]


@pytest.mark.parametrize("error, raises", [(_err(429), False), (_err(400), True)])
def test_failed_submit_deletes_hosted_images(tmp_path, openrouter, error, raises):
    host = FakeHost()
    openrouter.routes[("POST", "/batches")] = error
    job = _job(tmp_path, host)
    if raises:
        with pytest.raises(providers.OpenRouterAPIError):
            job.execute()
    else:
        assert job.execute() is None
    assert job.state == Job.NOT_SENT
    assert host.images == {} and host.deleted == ["img0"]


def test_other_providers_and_interactive_keep_inline_images(tmp_path, litellm_mock):
    host = FakeHost()
    seen = []
    litellm_mock.completion = lambda **kw: (seen.append(kw["messages"]) or _completion())
    job = _job(tmp_path, host, mode="interactive")
    assert job.execute() == "ok"
    assert seen[0][0]["content"][1]["image_url"]["url"] == PNG_URI
    assert host.images == {}
    assert providers.batch_needs_hosted_images("vertex_ai") is False


def _completion():
    from tests.conftest import make_obj
    return make_obj(choices=[make_obj(message=make_obj(content="ok"), finish_reason="stop")],
                    usage=make_obj(prompt_tokens=1, completion_tokens=1, total_tokens=2,
                                   prompt_tokens_details=None, cost=None),
                    _hidden_params={})


def test_poller_releases_images_and_sweeps_orphans(tmp_path, openrouter):
    host = FakeHost()
    openrouter.routes[("POST", "/batches")] = {"id": "batch-1", "status": "validating"}
    openrouter.routes[("DELETE", "/batches/batch-1")] = {}
    _job(tmp_path, host).execute()
    host.images["stray"] = (PNG, "image/png")  # e.g. left by a crash before submit
    openrouter.routes[("GET", "/batches")] = {"data": [], "has_more": False}

    poller = Poller(FileStorage(str(tmp_path)), run_cmd=lambda argv: None, image_host=host)
    assert poller.sweep_orphans(older_than_days=3)["images"] == ["stray"]  # img0 is live

    openrouter.routes[("GET", "/batches/batch-1")] = _batch("completed", [_result("req-0", "ok")])
    poller.poll_all()
    assert "img0" in host.deleted
