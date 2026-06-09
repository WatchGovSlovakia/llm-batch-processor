import pytest

from llm_batch import cli as cli_module


def test_poll_command_invokes_poller(monkeypatch, tmp_path):
    calls = {"poll_all": 0, "sweep": None, "max_age": None, "state_dir": None}

    class FakePoller:
        def __init__(self, backend, *, max_batch_age_days=3, run_cmd=None):
            calls["max_age"] = max_batch_age_days
            calls["state_dir"] = backend._dir       # FileStorage stores dir in _dir
        def poll_all(self):
            calls["poll_all"] += 1
            return {"advanced": []}
        def sweep_orphans(self, older_than_days):
            calls["sweep"] = older_than_days

    monkeypatch.setattr(cli_module, "Poller", FakePoller)
    rc = cli_module.main(["poll", str(tmp_path)])
    assert rc == 0
    assert calls["poll_all"] == 1
    assert calls["sweep"] is None              # not requested
    assert calls["max_age"] == 3
    assert calls["state_dir"] == str(tmp_path)


def test_poll_command_with_sweep_orphans(monkeypatch, tmp_path):
    swept = {}

    class FakePoller:
        def __init__(self, backend, *, max_batch_age_days=3, run_cmd=None):
            pass
        def poll_all(self):
            return {"advanced": []}
        def sweep_orphans(self, older_than_days):
            swept["days"] = older_than_days

    monkeypatch.setattr(cli_module, "Poller", FakePoller)
    rc = cli_module.main(["poll", str(tmp_path), "--sweep-orphans", "5"])
    assert rc == 0
    assert swept["days"] == 5


def test_poll_command_custom_max_age(monkeypatch, tmp_path):
    seen = {}

    class FakePoller:
        def __init__(self, backend, *, max_batch_age_days=3, run_cmd=None):
            seen["max_age"] = max_batch_age_days
        def poll_all(self):
            return {"advanced": []}
        def sweep_orphans(self, older_than_days):
            pass

    monkeypatch.setattr(cli_module, "Poller", FakePoller)
    cli_module.main(["poll", str(tmp_path), "--max-batch-age-days", "7"])
    assert seen["max_age"] == 7


def test_no_command_errors(monkeypatch):
    with pytest.raises(SystemExit):
        cli_module.main([])
