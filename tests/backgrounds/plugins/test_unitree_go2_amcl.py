from unittest.mock import MagicMock, patch

from backgrounds.base import BackgroundConfig
from backgrounds.plugins.unitree_go2_amcl import UnitreeGo2AMCL


class TestUnitreeGo2AMCL:
    """Test cases for UnitreeGo2AMCL background plugin."""

    @patch("backgrounds.plugins.unitree_go2_amcl.UnitreeGo2AMCLProvider")
    def test_initialization(self, mock_provider_class):
        """Test background initialization creates provider and starts it."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = BackgroundConfig()
        background = UnitreeGo2AMCL(config)

        assert background.config is config
        assert background.unitree_go2_amcl_provider == mock_provider
        mock_provider_class.assert_called_once()
        mock_provider.start.assert_called_once()

    @patch("backgrounds.plugins.unitree_go2_amcl.UnitreeGo2AMCLProvider")
    def test_initialization_logging(self, mock_provider_class, caplog):
        """Test that initialization logs the correct message."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = BackgroundConfig()
        with caplog.at_level("INFO"):
            UnitreeGo2AMCL(config)

        assert "Unitree Go2 AMCL Provider initialized in background" in caplog.text

    @patch("backgrounds.plugins.unitree_go2_amcl.UnitreeGo2AMCLProvider")
    def test_provider_attribute(self, mock_provider_class):
        """Test that provider attribute is set correctly."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = BackgroundConfig()
        background = UnitreeGo2AMCL(config)

        assert background.unitree_go2_amcl_provider is mock_provider

    @patch("backgrounds.plugins.unitree_go2_amcl.UnitreeGo2AMCLProvider")
    def test_config_stored(self, mock_provider_class):
        """Test that config is stored correctly."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = BackgroundConfig()
        background = UnitreeGo2AMCL(config)

        assert background.config is config
