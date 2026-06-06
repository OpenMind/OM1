"""Tests for expose.config — server configuration from env vars."""

import pytest

from expose.config import ServerConfig


class TestDefaults:
    def test_defaults_when_no_env_vars(self, monkeypatch):
        for var in ("OM1_WEBSIM_HOST", "OM1_WEBSIM_PORT", "OM1_LOG_LEVEL"):
            monkeypatch.delenv(var, raising=False)
        cfg = ServerConfig.from_env()
        assert cfg.websim_host == "127.0.0.1"
        assert cfg.websim_port == 8000
        assert cfg.log_level == "WARNING"


class TestEnvOverrides:
    def test_host_override(self, monkeypatch):
        monkeypatch.setenv("OM1_WEBSIM_HOST", "0.0.0.0")
        assert ServerConfig.from_env().websim_host == "0.0.0.0"

    def test_port_override_parses_int(self, monkeypatch):
        monkeypatch.setenv("OM1_WEBSIM_PORT", "9000")
        assert ServerConfig.from_env().websim_port == 9000

    def test_log_level_override(self, monkeypatch):
        monkeypatch.setenv("OM1_LOG_LEVEL", "DEBUG")
        assert ServerConfig.from_env().log_level == "DEBUG"


class TestValidation:
    def test_invalid_port_raises(self, monkeypatch):
        monkeypatch.setenv("OM1_WEBSIM_PORT", "not-a-number")
        with pytest.raises(ValueError):
            ServerConfig.from_env()
