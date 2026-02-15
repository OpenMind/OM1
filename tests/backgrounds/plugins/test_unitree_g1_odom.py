from unittest.mock import MagicMock, patch

from backgrounds.plugins.unitree_g1_odom import UnitreeG1Odom, UnitreeG1OdomConfig


class TestUnitreeG1OdomConfig:
    """Test cases for UnitreeG1OdomConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = UnitreeG1OdomConfig()
        assert config.unitree_ethernet is None

    def test_custom_unitree_ethernet(self):
        """Test custom unitree_ethernet configuration."""
        config = UnitreeG1OdomConfig(unitree_ethernet="eth0")
        assert config.unitree_ethernet == "eth0"

    def test_config_inherits_from_background_config(self):
        """Test that config properly inherits from BackgroundConfig."""
        config = UnitreeG1OdomConfig(unitree_ethernet="eth1")
        assert hasattr(config, "unitree_ethernet")
        assert config.unitree_ethernet == "eth1"


class TestUnitreeG1Odom:
    """Test cases for UnitreeG1Odom background plugin."""

    @patch("backgrounds.plugins.unitree_g1_odom.UnitreeG1OdomProvider")
    def test_initialization(self, mock_provider_class):
        """Test background initialization creates provider."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = UnitreeG1OdomConfig(unitree_ethernet="eth0")
        background = UnitreeG1Odom(config)

        assert background.config is config
        assert background.odom_provider == mock_provider
        mock_provider_class.assert_called_once_with("eth0")

    @patch("backgrounds.plugins.unitree_g1_odom.UnitreeG1OdomProvider")
    def test_initialization_with_none_ethernet(self, mock_provider_class):
        """Test background initialization with None unitree_ethernet."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = UnitreeG1OdomConfig()
        background = UnitreeG1Odom(config)

        assert background.config.unitree_ethernet is None
        assert background.odom_provider == mock_provider
        mock_provider_class.assert_called_once_with(None)

    @patch("backgrounds.plugins.unitree_g1_odom.UnitreeG1OdomProvider")
    def test_initialization_logging(self, mock_provider_class, caplog):
        """Test that initialization logs the correct message."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = UnitreeG1OdomConfig(unitree_ethernet="eth2")
        with caplog.at_level("INFO"):
            UnitreeG1Odom(config)

        assert "Initialized Unitree G1 Odom Provider: eth2" in caplog.text

    @patch("backgrounds.plugins.unitree_g1_odom.UnitreeG1OdomProvider")
    def test_initialization_logging_with_none(self, mock_provider_class, caplog):
        """Test that initialization logs correctly with None ethernet."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = UnitreeG1OdomConfig()
        with caplog.at_level("INFO"):
            UnitreeG1Odom(config)

        assert "Initialized Unitree G1 Odom Provider: None" in caplog.text

    @patch("backgrounds.plugins.unitree_g1_odom.UnitreeG1OdomProvider")
    def test_config_stored_correctly(self, mock_provider_class):
        """Test that config is stored correctly in the background instance."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = UnitreeG1OdomConfig(unitree_ethernet="wlan0")
        background = UnitreeG1Odom(config)

        assert background.config is config
        assert background.config.unitree_ethernet == "wlan0"

    @patch("backgrounds.plugins.unitree_g1_odom.UnitreeG1OdomProvider")
    def test_multiple_instances_with_different_ethernet(self, mock_provider_class):
        """Test multiple instances with different ethernet channels."""
        mock_provider1 = MagicMock()
        mock_provider2 = MagicMock()
        mock_provider_class.side_effect = [mock_provider1, mock_provider2]

        config1 = UnitreeG1OdomConfig(unitree_ethernet="eth0")
        config2 = UnitreeG1OdomConfig(unitree_ethernet="eth1")

        background1 = UnitreeG1Odom(config1)
        background2 = UnitreeG1Odom(config2)

        assert background1.config.unitree_ethernet == "eth0"
        assert background2.config.unitree_ethernet == "eth1"
        assert background1.odom_provider == mock_provider1
        assert background2.odom_provider == mock_provider2

        assert mock_provider_class.call_count == 2
        mock_provider_class.assert_any_call("eth0")
        mock_provider_class.assert_any_call("eth1")
