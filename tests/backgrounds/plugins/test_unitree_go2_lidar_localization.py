from unittest.mock import MagicMock, patch

import pytest

from backgrounds.base import Background, BackgroundConfig
from backgrounds.plugins.unitree_go2_lidar_localization import (
    UnitreeGo2LidarLocalization,
)


class TestUnitreeGo2LidarLocalization:
    """Test cases for UnitreeGo2LidarLocalization background plugin."""

    @patch(
        "backgrounds.plugins.unitree_go2_lidar_localization.UnitreeGo2LidarLocalizationProvider"
    )
    def test_initialization(self, mock_provider_class):
        """Test background initializes provider and calls start."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = BackgroundConfig()
        background = UnitreeGo2LidarLocalization(config)

        mock_provider_class.assert_called_once()
        mock_provider.start.assert_called_once()
        assert background.unitree_go2_lidar_localization_provider == mock_provider

    @patch(
        "backgrounds.plugins.unitree_go2_lidar_localization.UnitreeGo2LidarLocalizationProvider"
    )
    def test_initialization_logging(self, mock_provider_class, caplog):
        """Test that initialization logs the correct message."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = BackgroundConfig()
        with caplog.at_level("INFO"):
            UnitreeGo2LidarLocalization(config)

        assert (
            "Unitree Go2 Lidar Localization Provider initialized in background"
            in caplog.text
        )

    @patch(
        "backgrounds.plugins.unitree_go2_lidar_localization.UnitreeGo2LidarLocalizationProvider"
    )
    def test_inherits_from_background(self, mock_provider_class):
        """Test that UnitreeGo2LidarLocalization inherits from Background."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = BackgroundConfig()
        background = UnitreeGo2LidarLocalization(config)

        assert isinstance(background, Background)

    @patch(
        "backgrounds.plugins.unitree_go2_lidar_localization.UnitreeGo2LidarLocalizationProvider"
    )
    def test_config_stored(self, mock_provider_class):
        """Test that configuration is stored correctly."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = BackgroundConfig()
        background = UnitreeGo2LidarLocalization(config)

        assert background.config == config

    @patch(
        "backgrounds.plugins.unitree_go2_lidar_localization.UnitreeGo2LidarLocalizationProvider"
    )
    def test_provider_start_called_on_init(self, mock_provider_class):
        """Test that provider.start() is called during initialization."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = BackgroundConfig()
        UnitreeGo2LidarLocalization(config)

        mock_provider.start.assert_called_once()

    @patch(
        "backgrounds.plugins.unitree_go2_lidar_localization.UnitreeGo2LidarLocalizationProvider"
    )
    def test_provider_exception_propagates(self, mock_provider_class):
        """Test that provider initialization errors propagate."""
        mock_provider_class.side_effect = Exception("Lidar connection failed")

        config = BackgroundConfig()
        with pytest.raises(Exception, match="Lidar connection failed"):
            UnitreeGo2LidarLocalization(config)
