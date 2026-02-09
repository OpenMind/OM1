from unittest.mock import MagicMock, patch

import pytest

from backgrounds.base import Background, BackgroundConfig
from backgrounds.plugins.unitree_go2_amcl import UnitreeGo2AMCL


class TestUnitreeGo2AMCL:
    """Test cases for UnitreeGo2AMCL background plugin."""

    @patch("backgrounds.plugins.unitree_go2_amcl.UnitreeGo2AMCLProvider")
    def test_initialization(self, mock_provider_class):
        """Test background initializes provider and calls start."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = BackgroundConfig()
        background = UnitreeGo2AMCL(config)

        mock_provider_class.assert_called_once()
        mock_provider.start.assert_called_once()
        assert background.unitree_go2_amcl_provider == mock_provider

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
    def test_inherits_from_background(self, mock_provider_class):
        """Test that UnitreeGo2AMCL inherits from Background."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = BackgroundConfig()
        background = UnitreeGo2AMCL(config)

        assert isinstance(background, Background)

    @patch("backgrounds.plugins.unitree_go2_amcl.UnitreeGo2AMCLProvider")
    def test_config_stored(self, mock_provider_class):
        """Test that configuration is stored correctly."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = BackgroundConfig()
        background = UnitreeGo2AMCL(config)

        assert background.config == config

    @patch("backgrounds.plugins.unitree_go2_amcl.UnitreeGo2AMCLProvider")
    def test_provider_exception_propagates(self, mock_provider_class):
        """Test that provider initialization errors propagate."""
        mock_provider_class.side_effect = Exception("AMCL connection failed")

        config = BackgroundConfig()
        with pytest.raises(Exception, match="AMCL connection failed"):
            UnitreeGo2AMCL(config)
