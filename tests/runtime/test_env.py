import os
from unittest.mock import patch

from runtime.env import EnvLoader


class TestLoadEnvVars:
    """Test cases for EnvLoader.load_env_vars."""

    def test_simple(self):
        with patch.dict(os.environ, {"API_KEY": "secret123"}):
            result = EnvLoader.load_env_vars({"api_key": "${API_KEY}"})
            assert result == {"api_key": "secret123"}

    def test_default_when_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            result = EnvLoader.load_env_vars({"url": "${BASE_URL:-http://localhost}"})
            assert result == {"url": "http://localhost"}

    def test_env_overrides_default(self):
        with patch.dict(os.environ, {"BASE_URL": "http://prod.example.com"}):
            result = EnvLoader.load_env_vars({"url": "${BASE_URL:-http://localhost}"})
            assert result == {"url": "http://prod.example.com"}

    def test_nested_dict(self):
        with patch.dict(os.environ, {"K": "v1", "N": "v2"}):
            config = {"outer": {"inner": "${K}", "deep": {"leaf": "${N}"}}}
            result = EnvLoader.load_env_vars(config)
            assert result == {"outer": {"inner": "v1", "deep": {"leaf": "v2"}}}

    def test_list_in_dict(self):
        with patch.dict(os.environ, {"A": "x", "B": "y"}):
            result = EnvLoader.load_env_vars({"items": ["${A}", "${B}", "literal"]})
            assert result == {"items": ["x", "y", "literal"]}

    def test_mixed_list_in_dict(self):
        with patch.dict(os.environ, {"VAR": "replaced"}):
            result = EnvLoader.load_env_vars({"items": ["${VAR}", "static", 42]})
            assert result == {"items": ["replaced", "static", 42]}

    def test_primitives_unchanged(self):
        config = {"count": 42, "rate": 3.14, "flag": True, "empty": None}
        assert EnvLoader.load_env_vars(config) == config

    def test_none_value_unchanged(self):
        result = EnvLoader.load_env_vars({"key": None})
        assert result == {"key": None}

    def test_mixed_string(self):
        with patch.dict(os.environ, {"HOST": "example.com"}):
            result = EnvLoader.load_env_vars({"url": "https://${HOST}/api"})
            assert result == {"url": "https://example.com/api"}

    def test_multiple_vars_in_one_string(self):
        with patch.dict(os.environ, {"HOST": "example.com", "PORT": "8080"}):
            result = EnvLoader.load_env_vars({"addr": "${HOST}:${PORT}"})
            assert result == {"addr": "example.com:8080"}

    def test_default_with_trailing_path(self):
        with patch.dict(os.environ, {}, clear=True):
            result = EnvLoader.load_env_vars(
                {"url": "${EXAMPLE_URL:-http://example.local:8860}/v1"}
            )
            assert result == {"url": "http://example.local:8860/v1"}

    def test_env_overrides_default_with_trailing_path(self):
        with patch.dict(os.environ, {"EXAMPLE_URL": "http://prod.example.com:9999"}):
            result = EnvLoader.load_env_vars(
                {"url": "${EXAMPLE_URL:-http://example.local:8860}/v1"}
            )
            assert result == {"url": "http://prod.example.com:9999/v1"}

    def test_missing_var_keeps_pattern(self):
        with patch.dict(os.environ, {}, clear=True):
            result = EnvLoader.load_env_vars({"key": "${MISSING_VAR}"})
            assert result == {"key": "${MISSING_VAR}"}

    def test_missing_var_logs_warning(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            with patch.dict(os.environ, {}, clear=True):
                EnvLoader.load_env_vars({"key": "${MISSING_VAR}"})
        assert "MISSING_VAR" in caplog.text
        assert "no default was provided" in caplog.text

    def test_empty_env_var_used(self):
        with patch.dict(os.environ, {"EMPTY_VAR": ""}):
            result = EnvLoader.load_env_vars({"key": "${EMPTY_VAR:-fallback}"})
            assert result == {"key": ""}


class TestLoadValue:
    """Test cases for EnvLoader.load_value."""

    def test_no_pattern(self):
        assert EnvLoader.load_value("plain text") == "plain text"

    def test_dollar_without_braces(self):
        assert EnvLoader.load_value("$NOT_A_PATTERN") == "$NOT_A_PATTERN"

    def test_empty_default(self):
        with patch.dict(os.environ, {}, clear=True):
            assert EnvLoader.load_value("${MISSING:-}") == ""
