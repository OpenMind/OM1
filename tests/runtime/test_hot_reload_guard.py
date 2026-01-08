import pytest


class _StopLoop(Exception):
    """Internal exception to stop the infinite watcher loop in tests."""


@pytest.mark.asyncio
async def test_skip_reentrant_reload_when_already_reloading(monkeypatch, tmp_path):
    import runtime.single_mode.cortex as cortex_mod
    from runtime.single_mode.cortex import CortexRuntime

    cortex = CortexRuntime.__new__(CortexRuntime)

    cortex.check_interval = 0.0
    cortex.config_path = str(tmp_path / "config.json5")
    cortex.last_modified = 1.0
    cortex._is_reloading = True

    monkeypatch.setattr(cortex_mod.os.path, "exists", lambda _p: True)
    monkeypatch.setattr(cortex, "_get_file_mtime", lambda: 2.0)

    calls = {"n": 0}

    async def fake_reload():
        calls["n"] += 1

    monkeypatch.setattr(cortex, "_reload_config", fake_reload)

    sleep_calls = {"n": 0}

    async def fake_sleep(_):
        sleep_calls["n"] += 1
        if sleep_calls["n"] >= 2:
            raise _StopLoop()

    monkeypatch.setattr(cortex_mod.asyncio, "sleep", fake_sleep)

    with pytest.raises(_StopLoop):
        await cortex._check_config_changes()

    assert calls["n"] == 0
