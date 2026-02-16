from unittest.mock import MagicMock, patch

from backgrounds.plugins.unitree_go2_odom import (
    UnitreeGo2Odom,
    UnitreeGo2OdomConfig,
)


class TestUnitreeGo2OdomConfig:
    """Test cases for UnitreeGo2OdomConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = UnitreeGo2OdomConfig()
        assert config.unitree_ethernet is None

    def test_custom_unitree_ethernet(self):
        """Test custom unitree_ethernet configuration."""
        config = UnitreeGo2OdomConfig(unitree_ethernet="eth0")
        assert config.unitree_ethernet == "eth0"

    def test_config_inherits_from_background_config(self):
        """Test that config properly inherits from BackgroundConfig."""
        config = UnitreeGo2OdomConfig(unitree_ethernet="eth1")
        assert hasattr(config, "unitree_ethernet")
        assert config.unitree_ethernet == "eth1"


class TestUnitreeGo2Odom:
    """Test cases for UnitreeGo2Odom."""

    def test_initialization(self):
        """Test background initialization."""
        with patch(
            "backgrounds.plugins.unitree_go2_odom.UnitreeGo2OdomProvider"
        ) as mock_provider_class:
            mock_provider = MagicMock()
            mock_provider_class.return_value = mock_provider

            config = UnitreeGo2OdomConfig(unitree_ethernet="eth0")
            background = UnitreeGo2Odom(config)

            assert background.config == config
            assert background.odom_provider == mock_provider
            mock_provider_class.assert_called_once_with("eth0")

    def test_initialization_with_none_ethernet(self):
        """Test background initialization with None unitree_ethernet."""
        with patch(
            "backgrounds.plugins.unitree_go2_odom.UnitreeGo2OdomProvider"
        ) as mock_provider_class:
            mock_provider = MagicMock()
            mock_provider_class.return_value = mock_provider

            config = UnitreeGo2OdomConfig()
            background = UnitreeGo2Odom(config)

            assert background.config.unitree_ethernet is None
            assert background.odom_provider == mock_provider
            mock_provider_class.assert_called_once_with(None)

    def test_initialization_logging(self, caplog):
        """Test that initialization logs the correct message."""
        with patch(
            "backgrounds.plugins.unitree_go2_odom.UnitreeGo2OdomProvider"
        ) as mock_provider_class:
            mock_provider = MagicMock()
            mock_provider_class.return_value = mock_provider

            config = UnitreeGo2OdomConfig(unitree_ethernet="eth2")
            with caplog.at_level("INFO"):
                UnitreeGo2Odom(config)

            assert "Initialized Unitree Go2 Odom Provider: eth2" in caplog.text

    def test_initialization_logging_with_none(self, caplog):
        """Test that initialization logs correctly with None ethernet."""
        with patch(
            "backgrounds.plugins.unitree_go2_odom.UnitreeGo2OdomProvider"
        ) as mock_provider_class:
            mock_provider = MagicMock()
            mock_provider_class.return_value = mock_provider

            config = UnitreeGo2OdomConfig()
            with caplog.at_level("INFO"):
                UnitreeGo2Odom(config)

            assert "Initialized Unitree Go2 Odom Provider: None" in caplog.text

    def test_provider_initialization_with_correct_ethernet(self):
        """Test that provider is initialized with the correct ethernet channel."""
        with patch(
            "backgrounds.plugins.unitree_go2_odom.UnitreeGo2OdomProvider"
        ) as mock_provider_class:
            mock_provider = MagicMock()
            mock_provider_class.return_value = mock_provider

            config = UnitreeGo2OdomConfig(unitree_ethernet="enp2s0")
            UnitreeGo2Odom(config)
            mock_provider_class.assert_called_once_with("enp2s0")

    def test_config_stored_correctly(self):
        """Test that config is stored correctly in the background instance."""
        with patch(
            "backgrounds.plugins.unitree_go2_odom.UnitreeGo2OdomProvider"
        ) as mock_provider_class:
            mock_provider = MagicMock()
            mock_provider_class.return_value = mock_provider

            config = UnitreeGo2OdomConfig(unitree_ethernet="wlan0")
            background = UnitreeGo2Odom(config)

            assert background.config is config
            assert background.config.unitree_ethernet == "wlan0"

    def test_multiple_instances_with_different_ethernet(self):
        """Test that multiple instances can be created with different ethernet channels."""
        with patch(
            "backgrounds.plugins.unitree_go2_odom.UnitreeGo2OdomProvider"
        ) as mock_provider_class:
            mock_provider1 = MagicMock()
            mock_provider2 = MagicMock()
            mock_provider_class.side_effect = [mock_provider1, mock_provider2]

            config1 = UnitreeGo2OdomConfig(unitree_ethernet="eth0")
            config2 = UnitreeGo2OdomConfig(unitree_ethernet="eth1")

            background1 = UnitreeGo2Odom(config1)
            background2 = UnitreeGo2Odom(config2)

            assert background1.config.unitree_ethernet == "eth0"
            assert background2.config.unitree_ethernet == "eth1"
            assert background1.odom_provider == mock_provider1
            assert background2.odom_provider == mock_provider2

            assert mock_provider_class.call_count == 2
            mock_provider_class.assert_any_call("eth0")
            mock_provider_class.assert_any_call("eth1")
