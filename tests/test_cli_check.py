"""
Tests for the CLI check (diagnostics) command.

These tests verify that the OM1 setup diagnostics work correctly.
"""

import os
import sys
from unittest.mock import patch

import pytest


# Import the check helper functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
from cli import (
    _check_api_key_env,
    _check_config_directory,
    _check_os,
    _check_package,
    _check_python_version,
    _check_uv_installed,
)


class TestPythonVersionCheck:
    """Tests for Python version compatibility check."""

    def test_python_version_pass(self):
        """Python 3.10+ should pass."""
        status, message, _ = _check_python_version()
        # We're running on 3.10+, so it should pass
        assert status == "pass"
        assert "Python" in message

    @pytest.mark.skip(reason="Mocking sys.version_info tuple attributes is complex")
    def test_python_version_fail(self):
        """Python 3.8 should fail."""
        # This test is skipped because sys.version_info is a special namedtuple
        # that's difficult to mock correctly
        pass


class TestUvCheck:
    """Tests for uv package manager check."""

    def test_uv_installed(self):
        """uv should be detected if installed."""
        status, message, _ = _check_uv_installed()
        # If uv is installed, it should pass
        if status == "pass":
            assert "uv" in message
        else:
            assert status == "fail"
            assert "not found" in message.lower()


class TestOsCheck:
    """Tests for OS detection."""

    def test_os_detected(self):
        """OS should be detected."""
        status, message, _ = _check_os()
        assert status == "pass"
        # Should contain the OS name
        assert any(
            os_name in message for os_name in ["Darwin", "Linux", "Windows"]
        )


class TestConfigDirectoryCheck:
    """Tests for config directory check."""

    def test_config_directory_exists(self):
        """Config directory should exist in the repo."""
        status, message, _ = _check_config_directory()
        assert status == "pass"
        assert "config files" in message.lower()


class TestApiKeyCheck:
    """Tests for API key check."""

    @patch.dict(os.environ, {"OM_API_KEY": "test_key_123"})
    def test_api_key_configured(self):
        """API key should be detected when set."""
        status, message, _ = _check_api_key_env()
        assert status == "pass"
        assert "configured" in message.lower()

    @patch.dict(os.environ, {"OM_API_KEY": ""}, clear=True)
    def test_api_key_not_configured(self):
        """Missing API key should warn."""
        # Need to also clear any existing key
        with patch.dict(os.environ, {}, clear=True):
            status, message, hint = _check_api_key_env()
            assert status == "warn"
            assert "portal.openmind.org" in hint


class TestPackageCheck:
    """Tests for package installation check."""

    def test_check_installed_package(self):
        """Installed packages should be detected."""
        # pytest is definitely installed since we're running it
        status, message, _ = _check_package("pytest", "pytest")
        assert status == "pass"

    def test_check_missing_package(self):
        """Missing packages should fail."""
        status, message, hint = _check_package(
            "nonexistent_package_12345", "nonexistent"
        )
        assert status == "fail"
        assert "not installed" in message.lower()
        assert "pip install" in hint


class TestCheckCommandIntegration:
    """Integration tests for the check command."""

    def test_check_command_runs(self):
        """Check command should run without errors."""
        from typer.testing import CliRunner

        from cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["check"])
        # Should complete (exit 0) or warn (still exit 0)
        # Only fails (exit 1) if there are critical issues
        assert result.exit_code in [0, 1]
        assert "OM1 Setup Diagnostics" in result.output

    def test_check_verbose_flag(self):
        """Check command should accept --verbose flag."""
        from typer.testing import CliRunner

        from cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["check", "--verbose"])
        assert "OM1 Setup Diagnostics" in result.output
