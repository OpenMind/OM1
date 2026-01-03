"""
Comprehensive tests for ConfigProvider security features.

Tests cover:
- API key authentication
- Schema validation
- Backup and rollback mechanisms
- Error handling
- Security against timing attacks
"""

import json
import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import json5
import pytest

from providers.config_provider import ConfigProvider
from zenoh_msgs import String


@pytest.fixture
def temp_config_dir():
    """Create a temporary directory for config files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_zenoh_session():
    """Mock Zenoh session."""
    session = MagicMock()
    session.declare_publisher.return_value = MagicMock()
    session.declare_subscriber.return_value = MagicMock()
    return session


@pytest.fixture
def valid_single_mode_config():
    """Valid single-mode configuration."""
    return {
        "version": "1.0.0",
        "hertz": 1.0,
        "name": "test_config",
        "api_key": "test_api_key_12345",
        "system_prompt_base": "You are a helpful assistant.",
        "system_governance": "Follow ethical guidelines.",
        "system_prompt_examples": "Example interactions.",
        "agent_inputs": [{"type": "mock_input", "config": {}}],
        "cortex_llm": {"type": "openai_llm", "config": {"agent_name": "test"}},
        "agent_actions": [
            {
                "name": "speak",
                "llm_label": "speak",
                "connector": "mock_connector",
                "config": {},
            }
        ],
    }


@pytest.fixture
def valid_multi_mode_config():
    """Valid multi-mode configuration."""
    return {
        "version": "1.0.0",
        "default_mode": "mode1",
        "api_key": "test_api_key_12345",
        "system_governance": "Follow ethical guidelines.",
        "cortex_llm": {"type": "openai_llm", "config": {"agent_name": "test"}},
        "modes": {
            "mode1": {
                "display_name": "Mode 1",
                "description": "Test mode",
                "system_prompt_base": "You are a helpful assistant.",
                "hertz": 1.0,
                "agent_inputs": [{"type": "mock_input", "config": {}}],
                "agent_actions": [
                    {
                        "name": "speak",
                        "llm_label": "speak",
                        "connector": "mock_connector",
                        "config": {},
                    }
                ],
            }
        },
    }


@pytest.fixture
def config_provider_with_auth(temp_config_dir, mock_zenoh_session):
    """Create ConfigProvider with authentication enabled."""
    with patch(
        "providers.config_provider.open_zenoh_session", return_value=mock_zenoh_session
    ):
        with patch.dict(os.environ, {"OM_API_KEY": "authorized_key_12345"}):
            provider = ConfigProvider.__new__(ConfigProvider)
            provider.session = mock_zenoh_session
            provider.config_response_publisher = MagicMock()
            provider.config_request_subscriber = MagicMock()
            provider.running = True

            # Override paths to use temp directory
            memory_dir = os.path.join(temp_config_dir, "memory")
            os.makedirs(memory_dir, exist_ok=True)
            provider.config_path = os.path.join(memory_dir, ".runtime.json5")
            provider.backup_dir = os.path.join(memory_dir, "backups")
            os.makedirs(provider.backup_dir, exist_ok=True)
            provider.max_backups = 10
            provider._authorized_api_key = "authorized_key_12345"

            return provider


@pytest.fixture
def config_provider_no_auth(temp_config_dir, mock_zenoh_session):
    """Create ConfigProvider without authentication."""
    with patch(
        "providers.config_provider.open_zenoh_session", return_value=mock_zenoh_session
    ):
        with patch.dict(os.environ, {}, clear=True):
            provider = ConfigProvider.__new__(ConfigProvider)
            provider.session = mock_zenoh_session
            provider.config_response_publisher = MagicMock()
            provider.config_request_subscriber = MagicMock()
            provider.running = True

            # Override paths to use temp directory
            memory_dir = os.path.join(temp_config_dir, "memory")
            os.makedirs(memory_dir, exist_ok=True)
            provider.config_path = os.path.join(memory_dir, ".runtime.json5")
            provider.backup_dir = os.path.join(memory_dir, "backups")
            os.makedirs(provider.backup_dir, exist_ok=True)
            provider.max_backups = 10
            provider._authorized_api_key = None

            return provider


class TestAuthentication:
    """Test API key authentication."""

    def test_verify_valid_api_key(self, config_provider_with_auth):
        """Test that valid API key is accepted."""
        assert config_provider_with_auth._verify_api_key("authorized_key_12345") is True

    def test_verify_invalid_api_key(self, config_provider_with_auth):
        """Test that invalid API key is rejected."""
        assert config_provider_with_auth._verify_api_key("wrong_key") is False

    def test_verify_missing_api_key(self, config_provider_with_auth):
        """Test that missing API key is rejected."""
        assert config_provider_with_auth._verify_api_key(None) is False
        assert config_provider_with_auth._verify_api_key("") is False

    def test_verify_no_authorized_key_configured(self, config_provider_no_auth):
        """Test that updates are rejected when no authorized key is set."""
        assert config_provider_no_auth._verify_api_key("any_key") is False
        assert config_provider_no_auth._verify_api_key(None) is False

    def test_constant_time_compare_equal(self, config_provider_with_auth):
        """Test constant-time comparison with equal strings."""
        a = b"test_string"
        b = b"test_string"
        assert config_provider_with_auth._constant_time_compare(a, b) is True

    def test_constant_time_compare_different(self, config_provider_with_auth):
        """Test constant-time comparison with different strings."""
        a = b"test_string"
        b = b"different_string"
        assert config_provider_with_auth._constant_time_compare(a, b) is False

    def test_constant_time_compare_different_lengths(self, config_provider_with_auth):
        """Test constant-time comparison with different lengths."""
        a = b"short"
        b = b"much_longer_string"
        assert config_provider_with_auth._constant_time_compare(a, b) is False

    def test_constant_time_compare_timing_attack_resistance(
        self, config_provider_with_auth
    ):
        """Test that comparison is resistant to timing attacks."""
        import time

        # Test that comparison time is similar for equal and different strings
        # This is a basic test - full timing attack resistance requires more sophisticated testing
        a = b"a" * 1000
        b_equal = b"a" * 1000
        b_different = b"b" * 1000

        # Measure time for equal comparison
        start = time.perf_counter()
        for _ in range(1000):
            config_provider_with_auth._constant_time_compare(a, b_equal)
        equal_time = time.perf_counter() - start

        # Measure time for different comparison
        start = time.perf_counter()
        for _ in range(1000):
            config_provider_with_auth._constant_time_compare(a, b_different)
        different_time = time.perf_counter() - start

        # Times should be similar (within 50% - this is a basic check)
        # In production, you'd want tighter bounds
        assert abs(equal_time - different_time) / max(equal_time, different_time) < 0.5


class TestSchemaValidation:
    """Test schema validation."""

    def test_validate_valid_single_mode_config(
        self, config_provider_with_auth, valid_single_mode_config
    ):
        """Test validation of valid single-mode config."""
        is_valid, error = config_provider_with_auth._validate_config_schema(
            valid_single_mode_config
        )
        assert is_valid is True
        assert error == ""

    def test_validate_valid_multi_mode_config(
        self, config_provider_with_auth, valid_multi_mode_config
    ):
        """Test validation of valid multi-mode config."""
        is_valid, error = config_provider_with_auth._validate_config_schema(
            valid_multi_mode_config
        )
        assert is_valid is True
        assert error == ""

    def test_validate_missing_required_field(
        self, config_provider_with_auth, valid_single_mode_config
    ):
        """Test validation fails when required field is missing."""
        invalid_config = valid_single_mode_config.copy()
        del invalid_config["version"]
        is_valid, error = config_provider_with_auth._validate_config_schema(
            invalid_config
        )
        assert is_valid is False
        assert "version" in error.lower() or "required" in error.lower()

    def test_validate_invalid_type(
        self, config_provider_with_auth, valid_single_mode_config
    ):
        """Test validation fails when field has wrong type."""
        invalid_config = valid_single_mode_config.copy()
        invalid_config["hertz"] = "not_a_number"
        is_valid, error = config_provider_with_auth._validate_config_schema(
            invalid_config
        )
        assert is_valid is False
        assert "hertz" in error.lower() or "number" in error.lower()

    def test_detect_config_type_single_mode(
        self, config_provider_with_auth, valid_single_mode_config
    ):
        """Test detection of single-mode config."""
        assert (
            config_provider_with_auth._detect_config_type(valid_single_mode_config)
            == "single_mode"
        )

    def test_detect_config_type_multi_mode(
        self, config_provider_with_auth, valid_multi_mode_config
    ):
        """Test detection of multi-mode config."""
        assert (
            config_provider_with_auth._detect_config_type(valid_multi_mode_config)
            == "multi_mode"
        )


class TestBackupAndRollback:
    """Test backup and rollback functionality."""

    def test_create_backup(self, config_provider_with_auth, valid_single_mode_config):
        """Test that backup is created successfully."""
        # Create initial config file
        with open(config_provider_with_auth.config_path, "w") as f:
            json.dump(valid_single_mode_config, f, indent=2)

        backup_path = config_provider_with_auth._create_backup()
        assert backup_path is not None
        assert os.path.exists(backup_path)

        # Verify backup content matches original
        with open(backup_path, "r") as f:
            backup_config = json5.load(f)
        assert backup_config == valid_single_mode_config

    def test_backup_not_created_when_no_config(self, config_provider_with_auth):
        """Test that backup is not created when config file doesn't exist."""
        backup_path = config_provider_with_auth._create_backup()
        assert backup_path is None

    def test_rollback_config(self, config_provider_with_auth, valid_single_mode_config):
        """Test rollback to previous backup."""
        # Create initial config
        with open(config_provider_with_auth.config_path, "w") as f:
            json.dump(valid_single_mode_config, f, indent=2)

        # Create backup
        backup_path = config_provider_with_auth._create_backup()
        assert backup_path is not None

        # Modify config
        modified_config = valid_single_mode_config.copy()
        modified_config["name"] = "modified_config"
        with open(config_provider_with_auth.config_path, "w") as f:
            json.dump(modified_config, f, indent=2)

        # Rollback
        config_provider_with_auth._rollback_config(backup_path)

        # Verify original config is restored
        with open(config_provider_with_auth.config_path, "r") as f:
            restored_config = json5.load(f)
        assert restored_config["name"] == "test_config"

    def test_cleanup_old_backups(
        self, config_provider_with_auth, valid_single_mode_config
    ):
        """Test that old backups are cleaned up."""
        # Create initial config
        with open(config_provider_with_auth.config_path, "w") as f:
            json.dump(valid_single_mode_config, f, indent=2)

        # Set max_backups to 3 for testing
        config_provider_with_auth.max_backups = 3

        # Create 5 backups
        for i in range(5):
            with open(config_provider_with_auth.config_path, "w") as f:
                config = valid_single_mode_config.copy()
                config["name"] = f"config_{i}"
                json.dump(config, f, indent=2)
            config_provider_with_auth._create_backup()

        # Check that only 3 backups remain
        backup_files = [
            f
            for f in os.listdir(config_provider_with_auth.backup_dir)
            if f.startswith(".runtime_backup_") and f.endswith(".json5")
        ]
        assert len(backup_files) == 3


class TestConfigUpdate:
    """Test config update functionality with security checks."""

    def test_update_config_success(
        self, config_provider_with_auth, valid_single_mode_config
    ):
        """Test successful config update with valid config and API key."""
        # Create initial config
        with open(config_provider_with_auth.config_path, "w") as f:
            json.dump(valid_single_mode_config, f, indent=2)

        # Update config with valid API key
        updated_config = valid_single_mode_config.copy()
        updated_config["name"] = "updated_config"
        updated_config["api_key"] = "authorized_key_12345"

        request_id = String("test_request_123")
        config_str = json5.dumps(updated_config)

        config_provider_with_auth._handle_set_config(request_id, config_str)

        # Verify config was updated
        with open(config_provider_with_auth.config_path, "r") as f:
            saved_config = json5.load(f)
        assert saved_config["name"] == "updated_config"

    def test_update_config_invalid_api_key(
        self, config_provider_with_auth, valid_single_mode_config
    ):
        """Test that config update is rejected with invalid API key."""
        # Create initial config
        with open(config_provider_with_auth.config_path, "w") as f:
            json.dump(valid_single_mode_config, f, indent=2)

        # Try to update with wrong API key
        updated_config = valid_single_mode_config.copy()
        updated_config["api_key"] = "wrong_key"

        request_id = String("test_request_123")
        config_str = json5.dumps(updated_config)

        config_provider_with_auth._handle_set_config(request_id, config_str)

        # Verify config was NOT updated
        with open(config_provider_with_auth.config_path, "r") as f:
            saved_config = json5.load(f)
        assert saved_config["name"] == "test_config"  # Original name unchanged

    def test_update_config_missing_api_key(
        self, config_provider_with_auth, valid_single_mode_config
    ):
        """Test that config update is rejected when API key is missing."""
        # Create initial config
        with open(config_provider_with_auth.config_path, "w") as f:
            json.dump(valid_single_mode_config, f, indent=2)

        # Try to update without API key
        updated_config = valid_single_mode_config.copy()
        del updated_config["api_key"]

        request_id = String("test_request_123")
        config_str = json5.dumps(updated_config)

        config_provider_with_auth._handle_set_config(request_id, config_str)

        # Verify config was NOT updated
        with open(config_provider_with_auth.config_path, "r") as f:
            saved_config = json5.load(f)
        assert saved_config["name"] == "test_config"  # Original name unchanged

    def test_update_config_invalid_schema(
        self, config_provider_with_auth, valid_single_mode_config
    ):
        """Test that config update is rejected with invalid schema."""
        # Create initial config
        with open(config_provider_with_auth.config_path, "w") as f:
            json.dump(valid_single_mode_config, f, indent=2)

        # Try to update with invalid schema (missing required field)
        invalid_config = valid_single_mode_config.copy()
        invalid_config["api_key"] = "authorized_key_12345"
        del invalid_config["version"]  # Remove required field

        request_id = String("test_request_123")
        config_str = json5.dumps(invalid_config)

        config_provider_with_auth._handle_set_config(request_id, config_str)

        # Verify config was NOT updated
        with open(config_provider_with_auth.config_path, "r") as f:
            saved_config = json5.load(f)
        assert saved_config["name"] == "test_config"  # Original name unchanged

    def test_update_config_invalid_json5(
        self, config_provider_with_auth, valid_single_mode_config
    ):
        """Test that config update is rejected with invalid JSON5."""
        # Create initial config
        with open(config_provider_with_auth.config_path, "w") as f:
            json.dump(valid_single_mode_config, f, indent=2)

        # Try to update with invalid JSON5
        invalid_json5 = "{ invalid json5 syntax }"

        request_id = String("test_request_123")

        config_provider_with_auth._handle_set_config(request_id, invalid_json5)

        # Verify config was NOT updated
        with open(config_provider_with_auth.config_path, "r") as f:
            saved_config = json5.load(f)
        assert saved_config["name"] == "test_config"  # Original name unchanged

    def test_update_config_creates_backup(
        self, config_provider_with_auth, valid_single_mode_config
    ):
        """Test that backup is created before config update."""
        # Create initial config
        with open(config_provider_with_auth.config_path, "w") as f:
            json.dump(valid_single_mode_config, f, indent=2)

        # Count existing backups
        initial_backups = len(
            [
                f
                for f in os.listdir(config_provider_with_auth.backup_dir)
                if f.startswith(".runtime_backup_")
            ]
        )

        # Update config
        updated_config = valid_single_mode_config.copy()
        updated_config["api_key"] = "authorized_key_12345"
        request_id = String("test_request_123")
        config_str = json5.dumps(updated_config)

        config_provider_with_auth._handle_set_config(request_id, config_str)

        # Verify backup was created
        final_backups = len(
            [
                f
                for f in os.listdir(config_provider_with_auth.backup_dir)
                if f.startswith(".runtime_backup_")
            ]
        )
        assert final_backups == initial_backups + 1

    def test_update_config_rollback_on_failure(
        self, config_provider_with_auth, valid_single_mode_config
    ):
        """Test that config is rolled back on write failure."""
        # Create initial config
        with open(config_provider_with_auth.config_path, "w") as f:
            json.dump(valid_single_mode_config, f, indent=2)

        # Simulate write failure by making directory read-only
        # (This is a simplified test - in practice you'd mock the file operations)
        updated_config = valid_single_mode_config.copy()
        updated_config["api_key"] = "authorized_key_12345"
        request_id = String("test_request_123")
        config_str = json5.dumps(updated_config)

        # Mock os.rename to raise an exception
        with patch("os.rename", side_effect=OSError("Permission denied")):
            config_provider_with_auth._handle_set_config(request_id, config_str)

        # Verify config was rolled back (original config still exists)
        with open(config_provider_with_auth.config_path, "r") as f:
            saved_config = json5.load(f)
        assert saved_config["name"] == "test_config"  # Original name unchanged


class TestConfigHash:
    """Test config hash calculation for audit logging."""

    def test_calculate_config_hash(
        self, config_provider_with_auth, valid_single_mode_config
    ):
        """Test that config hash is calculated correctly."""
        hash1 = config_provider_with_auth._calculate_config_hash(
            valid_single_mode_config
        )
        assert isinstance(hash1, str)
        assert len(hash1) == 16  # First 16 chars of SHA256

    def test_config_hash_consistency(
        self, config_provider_with_auth, valid_single_mode_config
    ):
        """Test that same config produces same hash."""
        hash1 = config_provider_with_auth._calculate_config_hash(
            valid_single_mode_config
        )
        hash2 = config_provider_with_auth._calculate_config_hash(
            valid_single_mode_config
        )
        assert hash1 == hash2

    def test_config_hash_different_for_different_configs(
        self, config_provider_with_auth, valid_single_mode_config
    ):
        """Test that different configs produce different hashes."""
        config1 = valid_single_mode_config.copy()
        config2 = valid_single_mode_config.copy()
        config2["name"] = "different_name"

        hash1 = config_provider_with_auth._calculate_config_hash(config1)
        hash2 = config_provider_with_auth._calculate_config_hash(config2)
        assert hash1 != hash2


class TestErrorHandling:
    """Test error handling in various scenarios."""

    def test_handle_set_config_sends_error_response_on_failure(
        self, config_provider_with_auth, valid_single_mode_config
    ):
        """Test that error response is sent on failure."""
        # Create initial config
        with open(config_provider_with_auth.config_path, "w") as f:
            json.dump(valid_single_mode_config, f, indent=2)

        # Try to update with invalid API key
        updated_config = valid_single_mode_config.copy()
        updated_config["api_key"] = "wrong_key"

        request_id = String("test_request_123")
        config_str = json5.dumps(updated_config)

        config_provider_with_auth._handle_set_config(request_id, config_str)

        # Verify error response was sent
        assert config_provider_with_auth.config_response_publisher.put.called
        # The last call should be an error response
        call_args = config_provider_with_auth.config_response_publisher.put.call_args
        assert call_args is not None
