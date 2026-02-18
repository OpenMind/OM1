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

    def test_list(self):
        with patch.dict(os.environ, {"A": "x", "B": "y"}):
            result = EnvLoader.load_env_vars(["${A}", "${B}", "literal"])
            assert result == ["x", "y", "literal"]

    def test_mixed_list_in_dict(self):
        with patch.dict(os.environ, {"VAR": "replaced"}):
            result = EnvLoader.load_env_vars({"items": ["${VAR}", "static", 42]})
            assert result == {"items": ["replaced", "static", 42]}

    def test_primitives_unchanged(self):
        config = {"count": 42, "rate": 3.14, "flag": True, "empty": None}
        assert EnvLoader.load_env_vars(config) == config

    def test_none_input(self):
        assert EnvLoader.load_env_vars(None) is None

    def test_mixed_string(self):
        with patch.dict(os.environ, {"HOST": "example.com"}):
            result = EnvLoader.load_env_vars("https://${HOST}/api")
            assert result == "https://example.com/api"

    def test_multiple_vars_in_one_string(self):
        with patch.dict(os.environ, {"HOST": "example.com", "PORT": "8080"}):
            result = EnvLoader.load_env_vars("${HOST}:${PORT}")
            assert result == "example.com:8080"

    def test_missing_var_keeps_pattern(self):
        with patch.dict(os.environ, {}, clear=True):
            result = EnvLoader.load_env_vars({"key": "${MISSING_VAR}"})
            assert result == {"key": "${MISSING_VAR}"}

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


class TestFindMissing:
    """Test cases for EnvLoader._find_missing."""

    def test_all_present(self):
        with patch.dict(os.environ, {"A": "1", "B": "2"}):
            assert EnvLoader._find_missing({"x": "${A}", "y": "${B}"}) == []

    def test_missing_reported(self):
        with patch.dict(os.environ, {}, clear=True):
            result = EnvLoader._find_missing({"x": "${ZEBRA}", "y": "${ALPHA}"})
            assert result == ["ALPHA", "ZEBRA"]

    def test_defaults_not_reported(self):
        with patch.dict(os.environ, {}, clear=True):
            config = {"x": "${HAS_DEFAULT:-val}", "y": "${NO_DEFAULT}"}
            assert EnvLoader._find_missing(config) == ["NO_DEFAULT"]

    def test_nested(self):
        with patch.dict(os.environ, {}, clear=True):
            config = {"level1": {"level2": ["${DEEP_VAR}"]}}
            assert EnvLoader._find_missing(config) == ["DEEP_VAR"]

    def test_non_string_ignored(self):
        config = {"num": 42, "flag": True, "items": [1, 2, None]}
        assert EnvLoader._find_missing(config) == []
