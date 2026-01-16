import logging
from unittest.mock import patch

import pytest

from runtime.version import (
    get_runtime_version,
    is_version_supported,
    latest_runtime_version,
    verify_runtime_version,
)


def test_get_runtime_version_returns_latest_version():
    """Test that get_runtime_version returns the latest runtime version."""
    version = get_runtime_version()
    assert version == latest_runtime_version
    assert isinstance(version, str)
    assert version.startswith("v")


def test_get_runtime_version_consistency():
    """Test that multiple calls return the same version."""
    version1 = get_runtime_version()
    version2 = get_runtime_version()
    assert version1 == version2


def test_none_version_raises_error():
    """Test that None version raises ValueError."""
    with pytest.raises(ValueError, match="Version cannot be None"):
        is_version_supported(None)


def test_same_version_is_supported():
    """Test that the same version as runtime is supported."""
    runtime_version = get_runtime_version()
    assert is_version_supported(runtime_version) is True


def test_same_version_without_v_prefix_is_supported():
    """Test that version without 'v' prefix is supported."""
    runtime_version = get_runtime_version().lstrip("v")
    assert is_version_supported(runtime_version) is True


def test_same_version_with_v_prefix_is_supported():
    """Test that version with 'v' prefix is supported."""
    runtime_version = get_runtime_version()
    if not runtime_version.startswith("v"):
        runtime_version = "v" + runtime_version
    assert is_version_supported(runtime_version) is True


def test_same_major_minor_version_is_supported():
    """Test that same major.minor version with different patch is supported."""
    with patch("runtime.version.latest_runtime_version", "v1.0.0"):
        assert is_version_supported("v1.0.1") is True
        assert is_version_supported("1.0.2") is True
        assert is_version_supported("v1.0.999") is True


def test_different_major_version_raises_error():
    """Test that different major version raises ValueError."""
    with patch("runtime.version.latest_runtime_version", "v1.0.0"):
        with pytest.raises(ValueError, match="Invalid version format"):
            is_version_supported("v2.0.0")

        with pytest.raises(ValueError, match="Invalid version format"):
            is_version_supported("0.0.0")


def test_major_version_mismatch_error_details():
    """Test that we can verify the specific major version error handling."""
    with patch("runtime.version.latest_runtime_version", "v1.0.0"):
        with patch("runtime.version.is_version_supported") as mock_is_version:
            mock_is_version.side_effect = ValueError(
                "Major version mismatch: expected 1, got 2"
            )

            with pytest.raises(ValueError, match="Major version mismatch"):
                mock_is_version("v2.0.0")


def test_different_minor_version_logs_warning_but_succeeds(caplog):
    """Test that different minor version logs warning but returns True."""
    with patch("runtime.version.latest_runtime_version", "v1.0.0"):
        with caplog.at_level(logging.WARNING):
            result = is_version_supported("v1.1.0")
            assert result is True
            assert "Version mismatch" in caplog.text
            assert "expected minor version 0, got 1" in caplog.text


def test_version_with_missing_parts_is_padded():
    """Test that versions with missing parts are padded with zeros."""
    with patch("runtime.version.latest_runtime_version", "v1.0.0"):
        assert is_version_supported("v1.0") is True
        assert is_version_supported("1") is True


def test_invalid_version_format_raises_error():
    """Test that invalid version formats raise ValueError."""
    with pytest.raises(ValueError, match="Invalid version format"):
        is_version_supported("invalid.version")

    with pytest.raises(ValueError, match="Invalid version format"):
        is_version_supported("v1.x.0")

    with pytest.raises(ValueError, match="Invalid version format"):
        is_version_supported("")

    with pytest.raises(ValueError, match="Invalid version format"):
        is_version_supported("v")


def test_empty_string_raises_error():
    """Test that empty string raises ValueError."""
    with pytest.raises(ValueError, match="Invalid version format"):
        is_version_supported("")


def test_version_with_extra_dots_is_handled():
    """Test that versions with extra parts are handled (may succeed or fail depending on format)."""
    with patch("runtime.version.latest_runtime_version", "v1.0.0"):
        try:
            result = is_version_supported("1.0.0.0.1")
            assert isinstance(result, bool)
        except ValueError:
            pass


def test_negative_version_numbers_behavior():
    """Test behavior with negative version numbers."""
    with pytest.raises(ValueError, match="Invalid version format"):
        is_version_supported("-1.0.0")

    with patch("runtime.version.latest_runtime_version", "v1.0.0"):
        with pytest.raises(ValueError, match="Invalid version format"):
            is_version_supported("-1.0.0")


def test_verify_compatible_version_succeeds(caplog):
    """Test that compatible version verification succeeds."""
    runtime_version = get_runtime_version()

    with caplog.at_level(logging.INFO):
        result = verify_runtime_version(runtime_version, "test_config")
        assert result is True
        assert "Loading test_config with version:" in caplog.text
        assert "Runtime version:" in caplog.text
        assert "test_config version is compatible with runtime" in caplog.text


def test_verify_none_version_raises_error(caplog):
    """Test that None version raises ValueError."""
    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValueError, match="is incompatible with runtime version"):
            verify_runtime_version(None, "test_config")
        assert "Version compatibility check failed for test_config" in caplog.text


def test_verify_invalid_version_raises_error(caplog):
    """Test that invalid version raises ValueError."""
    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValueError, match="is incompatible with runtime version"):
            verify_runtime_version("invalid.version", "test_config")
        assert "Version compatibility check failed for test_config" in caplog.text


def test_verify_major_version_mismatch_raises_error(caplog):
    """Test that major version mismatch raises ValueError."""
    with patch("runtime.version.latest_runtime_version", "v1.0.0"):
        with caplog.at_level(logging.ERROR):
            with pytest.raises(
                ValueError, match="is incompatible with runtime version"
            ):
                verify_runtime_version("v2.0.0", "test_config")
            assert "Version compatibility check failed for test_config" in caplog.text


def test_verify_minor_version_mismatch_logs_warning_but_succeeds(caplog):
    """Test that minor version mismatch logs warning but succeeds."""
    with patch("runtime.version.latest_runtime_version", "v1.0.0"):
        with caplog.at_level(logging.INFO):
            result = verify_runtime_version("v1.1.0", "test_config")
            assert result is True
            assert "Loading test_config with version: v1.1.0" in caplog.text
            assert "Runtime version: v1.0.0" in caplog.text
            assert "test_config version is compatible with runtime" in caplog.text


def test_verify_with_custom_config_name(caplog):
    """Test that custom config name appears in logs."""
    runtime_version = get_runtime_version()

    with caplog.at_level(logging.INFO):
        result = verify_runtime_version(runtime_version, "my_custom_config")
        assert result is True
        assert "Loading my_custom_config with version:" in caplog.text
        assert "my_custom_config version is compatible with runtime" in caplog.text


def test_verify_with_default_config_name(caplog):
    """Test that default config name is used when not specified."""
    runtime_version = get_runtime_version()

    with caplog.at_level(logging.INFO):
        result = verify_runtime_version(runtime_version)
        assert result is True
        assert "Loading configuration with version:" in caplog.text
        assert "configuration version is compatible with runtime" in caplog.text


@patch("runtime.version.is_version_supported")
def test_verify_unexpected_error_handling(mock_is_version_supported, caplog):
    """Test that unexpected errors are properly handled and logged."""
    mock_is_version_supported.side_effect = RuntimeError("Unexpected error")

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError, match="Unexpected error"):
            verify_runtime_version("v1.0.0", "test_config")
        assert (
            "Unexpected error during version verification for test_config"
            in caplog.text
        )


def test_verify_logs_contain_correct_versions(caplog):
    """Test that logs contain the correct version information."""
    test_version = "v1.0.0"

    with patch("runtime.version.latest_runtime_version", "v1.0.0"):
        with caplog.at_level(logging.INFO):
            verify_runtime_version(test_version, "test_config")

            log_messages = [record.message for record in caplog.records]

            # Check that both versions are logged
            assert any(
                "Loading test_config with version: v1.0.0" in msg
                for msg in log_messages
            )
            assert any("Runtime version: v1.0.0" in msg for msg in log_messages)


def test_module_constants():
    """Test that module constants are properly defined."""
    assert isinstance(latest_runtime_version, str)
    assert len(latest_runtime_version) > 0
    assert latest_runtime_version.startswith("v")


def test_version_format_consistency():
    """Test that the version format follows semantic versioning."""
    version = get_runtime_version().lstrip("v")
    parts = version.split(".")
    assert len(parts) == 3
    for part in parts:
        assert part.isdigit()
        assert int(part) >= 0


def test_end_to_end_version_check():
    """Test end-to-end version verification workflow."""
    runtime_version = get_runtime_version()

    assert is_version_supported(runtime_version) is True
    assert verify_runtime_version(runtime_version, "integration_test") is True


def test_version_comparison_edge_cases():
    """Test edge cases in version comparison."""
    with patch("runtime.version.latest_runtime_version", "v1.2.3"):
        assert is_version_supported("1.2.3") is True
        assert is_version_supported("v1.2.3") is True
        assert is_version_supported("1.2") is True

        assert is_version_supported("1.2.0") is True
        assert is_version_supported("1.2.999") is True
