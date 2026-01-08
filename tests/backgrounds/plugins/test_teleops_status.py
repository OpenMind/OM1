import os
import sys
import time
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.modules.setdefault(
    "providers.context_provider", SimpleNamespace(ContextProvider=object)
)
sys.modules.setdefault(
    "providers.config_provider", SimpleNamespace(ConfigProvider=object)
)

from backgrounds.plugins.teleops_status import (
    TeleopsStatus,
    TeleopsStatusConfig,
)


@pytest.fixture
def mock_config_provider():
    with patch("backgrounds.plugins.teleops_status.ConfigProvider") as mock:
        mock_instance = MagicMock()
        mock_instance.get.return_value = "test_api_key"
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def mock_teleops_status_provider():
    with patch("backgrounds.plugins.teleops_status.TeleopsStatusProvider") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock, mock_instance


class TestTeleopsStatusConfig:
    def test_default_config(self):
        """Test default configuration values."""
        config = TeleopsStatusConfig()
        assert config.machine_name == "desktop_device"
        assert config.update_interval == 10

    def test_custom_config(self):
        """Test custom configuration values."""
        config = TeleopsStatusConfig(
            machine_name="my_device",
            update_interval=5
        )
        assert config.machine_name == "my_device"
        assert config.update_interval == 5


class TestTeleopsStatus:
    def test_init_with_config_provider(self, mock_config_provider, mock_teleops_status_provider):
        """Test initialization with API key from config provider."""
        mock_provider, mock_instance = mock_teleops_status_provider
        config = TeleopsStatusConfig(machine_name="test_device", update_interval=10)

        background = TeleopsStatus(config)

        assert background.machine_name == "test_device"
        assert background.update_interval == 10
        assert background.running is True
        mock_provider.assert_called_once_with(api_key="test_api_key")

        background.stop()

    def test_init_with_env_variable(self, mock_teleops_status_provider):
        """Test initialization with API key from environment variable."""
        mock_provider, mock_instance = mock_teleops_status_provider

        with patch("backgrounds.plugins.teleops_status.ConfigProvider") as mock_config:
            mock_config.side_effect = Exception("No config")
            with patch.dict(os.environ, {"OM_API_KEY": "env_api_key"}):
                config = TeleopsStatusConfig()
                background = TeleopsStatus(config)

                mock_provider.assert_called_once_with(api_key="env_api_key")
                background.stop()

    def test_init_no_api_key(self):
        """Test initialization fails gracefully without API key."""
        with patch("backgrounds.plugins.teleops_status.ConfigProvider") as mock_config:
            mock_config.side_effect = Exception("No config")
            with patch.dict(os.environ, {}, clear=True):
                if "OM_API_KEY" in os.environ:
                    del os.environ["OM_API_KEY"]

                config = TeleopsStatusConfig()
                background = TeleopsStatus(config)

                assert not hasattr(background, "status_provider")

    def test_stop(self, mock_config_provider, mock_teleops_status_provider):
        """Test stopping the background loop."""
        config = TeleopsStatusConfig()
        background = TeleopsStatus(config)

        assert background.running is True
        background.stop()
        assert background.running is False

    def test_status_loop_sends_status(self, mock_config_provider, mock_teleops_status_provider):
        """Test that status loop sends status updates."""
        mock_provider, mock_instance = mock_teleops_status_provider
        config = TeleopsStatusConfig(machine_name="test_device", update_interval=1)

        background = TeleopsStatus(config)

        # Wait for at least one status update
        time.sleep(1.5)
        background.stop()

        # Verify status was shared at least once
        assert mock_instance.share_status.called

    def test_status_loop_handles_errors(self, mock_config_provider, mock_teleops_status_provider):
        """Test that status loop handles errors gracefully."""
        mock_provider, mock_instance = mock_teleops_status_provider
        mock_instance.share_status.side_effect = Exception("Network error")

        config = TeleopsStatusConfig(update_interval=1)
        background = TeleopsStatus(config)

        # Wait for status update attempt
        time.sleep(1.5)
        background.stop()

        # Background should still be running despite error
        assert mock_instance.share_status.called

    def test_thread_is_daemon(self, mock_config_provider, mock_teleops_status_provider):
        """Test that the background thread is a daemon thread."""
        config = TeleopsStatusConfig()
        background = TeleopsStatus(config)

        assert background.thread.daemon is True
        background.stop()
