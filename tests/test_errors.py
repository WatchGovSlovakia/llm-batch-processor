import pytest

from llm_batch.errors import BatchJobFailed, TransientProviderError


def test_batch_job_failed_carries_batch_id():
    err = BatchJobFailed("provider rejected batch", batch_job_id="batches/abc")
    assert err.batch_job_id == "batches/abc"
    assert "provider rejected batch" in str(err)


def test_batch_job_failed_batch_id_optional():
    err = BatchJobFailed("no id available")
    assert err.batch_job_id is None


def test_transient_provider_error_is_exception():
    assert issubclass(TransientProviderError, Exception)
    with pytest.raises(TransientProviderError):
        raise TransientProviderError("429 rate limited")
