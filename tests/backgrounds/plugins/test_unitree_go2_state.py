from unittest.mock import MagicMock, patch

import pytest

from backgrounds.plugins.unitree_go2_state import (
    UnitreeGo2State,
    UnitreeGo2StateConfig,
)


class TestUnitreeGo2StateConfig:
    """Test cases for UnitreeGo2StateConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = UnitreeGo2StateConfig()
        assert config.unitree_ethernet is None

    def test_custom_unitree_ethernet(self):
        """Test custom unitree_ethernet configuration."""
        config = UnitreeGo2StateConfig(unitree_ethernet="eth0")
        assert config.unitree_ethernet == "eth0"


class TestUnitreeGo2State:
    """Test cases for UnitreeGo2State background plugin."""

    @patch("backgrounds.plugins.unitree_go2_state.UnitreeGo2StateProvider")
    def test_initialization(self, mock_provider_class):
        """Test background initialization with valid ethernet."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = UnitreeGo2StateConfig(unitree_ethernet="eth0")
        background = UnitreeGo2State(config)

        assert background.config is config
        assert background.unitree_go2_state_provider == mock_provider
        mock_provider_class.assert_called_once()

    @patch("backgrounds.plugins.unitree_go2_state.UnitreeGo2StateProvider")
    def test_initialization_logging(self, mock_provider_class, caplog):
        """Test that initialization logs the correct message."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = UnitreeGo2StateConfig(unitree_ethernet="eth0")
        with caplog.at_level("INFO"):
            UnitreeGo2State(config)

        assert "Unitree Go2 State Provider initialized in background" in caplog.text

    @patch("backgrounds.plugins.unitree_go2_state.UnitreeGo2StateProvider")
    def test_initialization_with_none_ethernet_raises_value_error(
        self, mock_provider_class
    ):
        """Test that None ethernet raises ValueError."""
        config = UnitreeGo2StateConfig()

        with pytest.raises(
            ValueError,
            match="Unitree Go2 Ethernet channel must be specified",
        ):
            UnitreeGo2State(config)

        mock_provider_class.assert_not_called()

    @patch("backgrounds.plugins.unitree_go2_state.UnitreeGo2StateProvider")
    def test_initialization_with_none_ethernet_error_log(
        self, mock_provider_class, caplog
    ):
        """Test that None ethernet logs an error before raising."""
        config = UnitreeGo2StateConfig()

        with caplog.at_level("ERROR"):
            with pytest.raises(ValueError):
                UnitreeGo2State(config)

        assert (
            "Unitree Go2 Ethernet channel is not set in the configuration"
            in caplog.text
        )

    @patch("backgrounds.plugins.unitree_go2_state.UnitreeGo2StateProvider")
    def test_initialization_with_empty_string_raises_value_error(
        self, mock_provider_class
    ):
        """Test that empty string ethernet raises ValueError."""
        config = UnitreeGo2StateConfig(unitree_ethernet="")

        with pytest.raises(ValueError):
            UnitreeGo2State(config)

        mock_provider_class.assert_not_called()

    @patch("backgrounds.plugins.unitree_go2_state.UnitreeGo2StateProvider")
    def test_config_stored(self, mock_provider_class):
        """Test that config is stored correctly."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = UnitreeGo2StateConfig(unitree_ethernet="eth0")
        background = UnitreeGo2State(config)

        assert background.config is config
        assert background.config.unitree_ethernet == "eth0"
