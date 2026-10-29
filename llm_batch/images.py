"""Image hosting for batch APIs that reject inline images.

OpenRouter's Batch API rejects base64/data-URI images and only takes public
https URLs, which the upstream provider fetches when it runs the batch (maybe
hours later). An `ImageHost` stores the bytes and returns such a URL; the Job
swaps every data-URI image for one before submit and deletes them once the
batch has ended.

Backends:
    MongoImageHost — a MongoDB collection; something else (e.g. your API)
                     serves `<public_base_url>/<id>` from it.
    S3ImageHost    — a publicly readable S3 bucket (or a CDN in front of it).

Ids are random and unguessable (the URL is the only access control), with the
image's extension so upstreams that sniff the URL see the type.
"""

from __future__ import annotations

import base64
import copy
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Protocol, runtime_checkable

_DATA_URI = re.compile(r"^data:(?P<type>[^;,]*)(?:;[^;,]*)*;base64,(?P<data>.*)$", re.DOTALL)
_EXTENSIONS = {"image/png": ".png", "image/jpeg": ".jpg", "image/gif": ".gif",
               "image/webp": ".webp"}


@runtime_checkable
class ImageHost(Protocol):
    """Stores image bytes behind public URLs."""

    def put(self, data: bytes, content_type: str) -> tuple[str, str]:
        """Store one image; returns (id, public https URL)."""
        ...

    def delete(self, image_id: str) -> None:
        """Delete one image. A missing id is not an error."""
        ...

    def delete_orphans(self, live_ids: set, older_than_days: float) -> list[str]:
        """Delete images older than the cutoff whose id is not in live_ids; returns the deleted ids."""
        ...


def new_image_id(content_type: str) -> str:
    return secrets.token_urlsafe(24) + _EXTENSIONS.get(content_type, "")


def _cutoff(older_than_days: float) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=older_than_days)


def _parse_data_uri(url: str) -> tuple[bytes, str] | None:
    match = _DATA_URI.match(url)
    if not match:
        return None
    return base64.b64decode(match["data"]), match["type"] or "application/octet-stream"


def externalize_images(requests: list, host: ImageHost) -> tuple[list, list[str]]:
    """Return a copy of `requests` (message lists) with every data-URI `image_url`
    replaced by a hosted URL, and the ids of the hosted images.

    Both content-part shapes are handled: `{"image_url": {"url": ...}}` and
    `{"image_url": "..."}`. Identical images within one call are stored once. If
    a put fails, the images already stored are deleted and the error propagates.
    """
    out = copy.deepcopy(requests)
    ids: list[str] = []
    by_uri: dict[str, str] = {}

    def hosted(url: str) -> str:
        if url in by_uri:
            return by_uri[url]
        parsed = _parse_data_uri(url)
        if parsed is None:
            return url
        image_id, public_url = host.put(*parsed)
        ids.append(image_id)
        by_uri[url] = public_url
        return public_url

    try:
        for messages in out:
            for message in messages:
                content = message.get("content") if isinstance(message, dict) else None
                if not isinstance(content, list):
                    continue
                for part in content:
                    if not isinstance(part, dict) or "image_url" not in part:
                        continue
                    image = part["image_url"]
                    if isinstance(image, str) and image.startswith("data:"):
                        part["image_url"] = hosted(image)
                    elif isinstance(image, dict) and str(image.get("url", "")).startswith("data:"):
                        image["url"] = hosted(image["url"])
    except BaseException:
        release_images(host, ids)
        raise
    return out, ids


def release_images(host: ImageHost, ids) -> None:
    """Best-effort delete of hosted images; never raises."""
    for image_id in ids or ():
        try:
            host.delete(image_id)
        except Exception:
            pass


class MongoImageHost:
    """Images as documents of a pymongo collection (default `llm_images`):

        {_id: <image id>, content_type: str, data: Binary, created_at: datetime}

    `public_base_url` is where your server serves the collection, e.g.
    `https://example.org/api/v1/llm-images`; the URL is `<public_base_url>/<id>`.
    An image must fit one document (16 MiB); batch providers cap images far lower.
    `ensure_indexes(ttl_days)` adds a TTL index as a backstop for orphans.
    """

    MAX_BYTES = 15 * 1024 * 1024  # under the 16 MiB BSON document limit

    def __init__(self, db, public_base_url: str, *, collection: str = "llm_images"):
        self._coll = db[collection]
        self._base = public_base_url.rstrip("/")

    def ensure_indexes(self, ttl_days: float | None = 10) -> None:
        """TTL index on created_at. Keep ttl_days above the longest batch turnaround
        (OpenRouter via Vertex can take 96 h); None skips it."""
        if ttl_days is not None:
            self._coll.create_index("created_at", expireAfterSeconds=int(ttl_days * 86400))

    def put(self, data: bytes, content_type: str) -> tuple[str, str]:
        if len(data) > self.MAX_BYTES:
            raise ValueError(f"image of {len(data)} bytes exceeds {self.MAX_BYTES}")
        image_id = new_image_id(content_type)
        self._coll.insert_one({"_id": image_id, "content_type": content_type, "data": data,
                               "created_at": datetime.now(timezone.utc)})
        return image_id, f"{self._base}/{image_id}"

    def delete(self, image_id: str) -> None:
        self._coll.delete_one({"_id": image_id})

    def delete_orphans(self, live_ids: set, older_than_days: float) -> list[str]:
        old = [doc["_id"] for doc in self._coll.find(
            {"created_at": {"$lt": _cutoff(older_than_days)}}, {"_id": 1})]
        orphans = [i for i in old if i not in live_ids]
        if orphans:
            self._coll.delete_many({"_id": {"$in": orphans}})
        return orphans


class S3ImageHost:
    """Images as objects `<prefix><id>` in a publicly readable S3 bucket.

    `client` is a boto3 S3 client (any S3-compatible store, e.g. MinIO, via its
    `endpoint_url`). The URL is `<public_base_url>/<prefix><id>`.
    """

    def __init__(self, client, bucket: str, public_base_url: str, *, prefix: str = "llm-images/"):
        self._client = client
        self._bucket = bucket
        self._base = public_base_url.rstrip("/")
        self._prefix = prefix

    def put(self, data: bytes, content_type: str) -> tuple[str, str]:
        image_id = new_image_id(content_type)
        self._client.put_object(Bucket=self._bucket, Key=self._prefix + image_id,
                                Body=data, ContentType=content_type)
        return image_id, f"{self._base}/{self._prefix}{image_id}"

    def delete(self, image_id: str) -> None:
        self._client.delete_object(Bucket=self._bucket, Key=self._prefix + image_id)

    def delete_orphans(self, live_ids: set, older_than_days: float) -> list[str]:
        cutoff, deleted = _cutoff(older_than_days), []
        pages = self._client.get_paginator("list_objects_v2").paginate(
            Bucket=self._bucket, Prefix=self._prefix)
        for page in pages:
            for obj in page.get("Contents", []):
                image_id = obj["Key"][len(self._prefix):]
                if image_id in live_ids or obj["LastModified"] >= cutoff:
                    continue
                try:
                    self.delete(image_id)
                    deleted.append(image_id)
                except Exception:
                    pass
        return deleted
