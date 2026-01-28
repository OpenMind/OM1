from unittest.mock import MagicMock, patch

import pytest

from backgrounds.plugins.agent_teleops_status import (
    AgentTeleopsStatusBackground,
    AgentTeleopsStatusConfig,
)
from providers import BatteryStatus, TeleopsStatus


class TestAgentTeleopsStatusConfig:
    """Test cases for AgentTeleopsStatusConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AgentTeleopsStatusConfig()
        assert config.machine_name == "agent_teleops_status_reporter"

    def test_custom_machine_name(self):
        """Test custom machine name configuration."""
        config = AgentTeleopsStatusConfig(machine_name="custom_machine")
        assert config.machine_name == "custom_machine"


class TestAgentTeleopsStatusBackground:
    """Test cases for AgentTeleopsStatusBackground."""

    @patch("backgrounds.plugins.agent_teleops_status.TeleopsStatusProvider")
    def test_initialization(self, mock_provider_class):
        """Test background initialization."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = AgentTeleopsStatusConfig(machine_name="test_machine")
        background = AgentTeleopsStatusBackground(config)

        assert background.config == config
        assert background.teleops_status_provider == mock_provider
        mock_provider_class.assert_called_once()

    @patch("backgrounds.plugins.agent_teleops_status.TeleopsStatusProvider")
    def test_initialization_logging(self, mock_provider_class, caplog):
        """Test that initialization logs the correct message."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = AgentTeleopsStatusConfig()
        with caplog.at_level("INFO"):
            AgentTeleopsStatusBackground(config)

        assert "Initiated Teleops Status Provider in background" in caplog.text

    @patch("backgrounds.plugins.agent_teleops_status.time.sleep")
    @patch("backgrounds.plugins.agent_teleops_status.time.time")
    @patch("backgrounds.plugins.agent_teleops_status.TeleopsStatusProvider")
    def test_run_shares_status(self, mock_provider_class, mock_time, mock_sleep):
        """Test that run method shares status with correct parameters."""
        mock_time.return_value = 1234567890.0
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = AgentTeleopsStatusConfig(machine_name="test_machine")
        background = AgentTeleopsStatusBackground(config)

        background.run()

        mock_provider.share_status.assert_called_once()

        call_args = mock_provider.share_status.call_args
        status = call_args[0][0]

        assert isinstance(status, TeleopsStatus)
        assert status.machine_name == "test_machine"
        assert status.update_time == str(mock_time.return_value)
        assert isinstance(status.battery_status, BatteryStatus)

        mock_sleep.assert_called_once_with(60)

    @patch("backgrounds.plugins.agent_teleops_status.time.sleep")
    @patch("backgrounds.plugins.agent_teleops_status.time.time")
    @patch("backgrounds.plugins.agent_teleops_status.TeleopsStatusProvider")
    def test_run_battery_status_from_empty_dict(
        self, mock_provider_class, mock_time, mock_sleep
    ):
        """Test that battery status is created from empty dict."""
        mock_time.return_value = 1234567890.0
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = AgentTeleopsStatusConfig()
        background = AgentTeleopsStatusBackground(config)
        background.run()

        call_args = mock_provider.share_status.call_args
        status = call_args[0][0]

        assert status.battery_status.battery_level == 0.0
        assert status.battery_status.charging_status is False
        assert status.battery_status.temperature == 0.0
        assert status.battery_status.voltage == 0.0

    @patch("backgrounds.plugins.agent_teleops_status.time.sleep")
    @patch("backgrounds.plugins.agent_teleops_status.TeleopsStatusProvider")
    def test_run_provider_exception_handling(self, mock_provider_class, mock_sleep):
        """Test that exceptions from provider are propagated."""
        mock_provider = MagicMock()
        mock_provider.share_status.side_effect = Exception("Provider error")
        mock_provider_class.return_value = mock_provider

        config = AgentTeleopsStatusConfig()
        background = AgentTeleopsStatusBackground(config)

        with pytest.raises(Exception, match="Provider error"):
            background.run()

    @patch("backgrounds.plugins.agent_teleops_status.TeleopsStatusProvider")
    def test_inherits_from_background(self, mock_provider_class):
        """Test that AgentTeleopsStatusBackground inherits from Background."""
        from backgrounds.base import Background

        config = AgentTeleopsStatusConfig()
        background = AgentTeleopsStatusBackground(config)

        assert isinstance(background, Background)

    @patch("backgrounds.plugins.agent_teleops_status.TeleopsStatusProvider")
    def test_config_type_annotation(self, mock_provider_class):
        """Test that the class correctly uses AgentTeleopsStatusConfig."""
        config = AgentTeleopsStatusConfig(machine_name="typed_test")
        background = AgentTeleopsStatusBackground(config)

        assert type(background.config) is AgentTeleopsStatusConfig
        assert background.config.machine_name == "typed_test"
