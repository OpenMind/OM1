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
        assert config.api_key is None
        assert config.use_sim is False

    def test_custom_unitree_ethernet(self):
        """Test custom unitree_ethernet configuration."""
        config = UnitreeGo2StateConfig(unitree_ethernet="eth0")
        assert config.unitree_ethernet == "eth0"

    def test_config_with_api_key(self):
        """Test configuration with API key."""
        config = UnitreeGo2StateConfig(api_key="test-api-key", unitree_ethernet="eth0")
        assert config.api_key == "test-api-key"
        assert config.unitree_ethernet == "eth0"

    def test_config_with_use_sim(self):
        """Test configuration with use_sim enabled."""
        config = UnitreeGo2StateConfig(use_sim=True)
        assert config.use_sim is True
        assert config.unitree_ethernet is None

    def test_config_with_all_parameters(self):
        """Test configuration with all parameters set."""
        config = UnitreeGo2StateConfig(api_key="test-key", use_sim=True, unitree_ethernet="eth0")
        assert config.api_key == "test-key"
        assert config.use_sim is True
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
    def test_initialization_with_none_ethernet_raises_value_error(self, mock_provider_class):
        """Test that None ethernet raises ValueError."""
        config = UnitreeGo2StateConfig()

        with pytest.raises(
            ValueError,
            match="Unitree Go2 Ethernet channel must be specified",
        ):
            UnitreeGo2State(config)

        mock_provider_class.assert_not_called()

    @patch("backgrounds.plugins.unitree_go2_state.UnitreeGo2StateProvider")
    def test_initialization_with_none_ethernet_error_log(self, mock_provider_class, caplog):
        """Test that None ethernet logs an error before raising."""
        config = UnitreeGo2StateConfig()

        with caplog.at_level("ERROR"):
            with pytest.raises(ValueError):
                UnitreeGo2State(config)

        assert "Unitree Go2 Ethernet channel is not set in the configuration" in caplog.text

    @patch("backgrounds.plugins.unitree_go2_state.UnitreeGo2StateProvider")
    def test_initialization_with_empty_string_raises_value_error(self, mock_provider_class):
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

    @patch("backgrounds.plugins.unitree_go2_state.UnitreeGo2StateZenohProvider")
    def test_initialization_with_use_sim(self, mock_zenoh_provider_class):
        """Test initialization with use_sim=True uses Zenoh provider."""
        mock_provider = MagicMock()
        mock_zenoh_provider_class.return_value = mock_provider

        config = UnitreeGo2StateConfig(use_sim=True)
        background = UnitreeGo2State(config)

        assert background.unitree_go2_state_provider == mock_provider
        mock_zenoh_provider_class.assert_called_once_with(None, True)

    @patch("backgrounds.plugins.unitree_go2_state.UnitreeGo2StateZenohProvider")
    def test_initialization_with_use_sim_and_api_key(self, mock_zenoh_provider_class):
        """Test initialization with use_sim=True and api_key."""
        mock_provider = MagicMock()
        mock_zenoh_provider_class.return_value = mock_provider

        config = UnitreeGo2StateConfig(use_sim=True, api_key="test-key")
        background = UnitreeGo2State(config)

        assert background.unitree_go2_state_provider == mock_provider
        mock_zenoh_provider_class.assert_called_once_with("test-key", True)

    @patch("backgrounds.plugins.unitree_go2_state.UnitreeGo2StateZenohProvider")
    def test_initialization_with_use_sim_no_ethernet_required(self, mock_zenoh_provider_class):
        """Test that use_sim=True does not require unitree_ethernet."""
        mock_provider = MagicMock()
        mock_zenoh_provider_class.return_value = mock_provider

        # Should not raise ValueError even without unitree_ethernet
        config = UnitreeGo2StateConfig(use_sim=True)
        background = UnitreeGo2State(config)

        assert background.unitree_go2_state_provider == mock_provider
        mock_zenoh_provider_class.assert_called_once()

    @patch("backgrounds.plugins.unitree_go2_state.UnitreeGo2StateZenohProvider")
    def test_initialization_with_use_sim_logging(self, mock_zenoh_provider_class, caplog):
        """Test that use_sim initialization logs the correct message."""
        mock_provider = MagicMock()
        mock_zenoh_provider_class.return_value = mock_provider

        config = UnitreeGo2StateConfig(use_sim=True)
        with caplog.at_level("INFO"):
            UnitreeGo2State(config)

        assert "Unitree Go2 State Zenoh Provider initialized in background" in caplog.text

    @patch("backgrounds.plugins.unitree_go2_state.UnitreeGo2StateProvider")
    @patch("backgrounds.plugins.unitree_go2_state.UnitreeGo2StateZenohProvider")
    def test_use_sim_false_uses_regular_provider(self, mock_zenoh_provider_class, mock_provider_class):
        """Test that use_sim=False uses regular provider."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = UnitreeGo2StateConfig(use_sim=False, unitree_ethernet="eth0")
        background = UnitreeGo2State(config)

        mock_provider_class.assert_called_once()
        mock_zenoh_provider_class.assert_not_called()
        assert background.unitree_go2_state_provider == mock_provider

    @patch("backgrounds.plugins.unitree_go2_state.UnitreeGo2StateProvider")
    def test_initialization_with_api_key_regular_provider(self, mock_provider_class):
        """Test initialization with api_key but use_sim=False."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = UnitreeGo2StateConfig(api_key="test-key", unitree_ethernet="eth0")
        background = UnitreeGo2State(config)

        assert background.config.api_key == "test-key"
        assert background.unitree_go2_state_provider == mock_provider
        mock_provider_class.assert_called_once()
