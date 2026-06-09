import io
import json
import types

import pytest


@pytest.fixture
def litellm_mock(monkeypatch):
    """A fake litellm module patched into llm_batch.providers.

    Tests configure return values by assigning to the returned object's
    attributes (each is a plain function or a list of canned objects).
    """
    fake = types.SimpleNamespace()

    # Default no-op stubs; individual tests override these.
    def _not_configured(*a, **k):  # pragma: no cover - guard
        raise AssertionError("litellm function called but not configured by the test")

    fake.create_file = _not_configured
    fake.create_batch = _not_configured
    fake.retrieve_batch = _not_configured
    fake.file_content = _not_configured
    fake.completion = _not_configured
    fake.file_delete = lambda *a, **k: None
    fake.cost_per_token = lambda **k: (0.0, 0.0)

    import llm_batch.providers as providers
    monkeypatch.setattr(providers, "litellm", fake)
    return fake


def make_obj(**attrs):
    """Build a simple attribute-bag object (stand-in for litellm response objects)."""
    return types.SimpleNamespace(**attrs)


def batch_output_jsonl(texts, usage=None):
    """Build the raw JSONL body that litellm.file_content returns for a completed batch.

    `texts` is a list aligned to custom_id req-0..req-N.
    """
    usage = usage or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    lines = []
    for i, text in enumerate(texts):
        lines.append(json.dumps({
            "custom_id": f"req-{i}",
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [{"message": {"content": text}}],
                    "usage": usage,
                },
            },
        }))
    body = "\n".join(lines) + "\n"
    return make_obj(text=body, content=body.encode())


def completion_response(text, usage=None):
    """Build a stand-in for a litellm.completion(...) response."""
    usage = usage or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    message = make_obj(content=text)
    choice = make_obj(message=message)
    return make_obj(choices=[choice], usage=usage)
