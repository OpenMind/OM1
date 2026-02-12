from unittest.mock import MagicMock, patch

from backgrounds.plugins.unitree_go2_locations import (
    UnitreeGo2Locations,
    UnitreeGo2LocationsConfig,
)


class TestUnitreeGo2LocationsConfig:
    """Test cases for UnitreeGo2LocationsConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = UnitreeGo2LocationsConfig()
        assert config.base_url == "http://localhost:5000/maps/locations/list"
        assert config.timeout == 5
        assert config.refresh_interval == 30

    def test_custom_base_url(self):
        """Test custom base_url configuration."""
        config = UnitreeGo2LocationsConfig(base_url="http://example.com/locations")
        assert config.base_url == "http://example.com/locations"

    def test_custom_timeout(self):
        """Test custom timeout configuration."""
        config = UnitreeGo2LocationsConfig(timeout=10)
        assert config.timeout == 10

    def test_custom_refresh_interval(self):
        """Test custom refresh_interval configuration."""
        config = UnitreeGo2LocationsConfig(refresh_interval=60)
        assert config.refresh_interval == 60

    def test_all_custom_values(self):
        """Test configuration with all custom values."""
        config = UnitreeGo2LocationsConfig(
            base_url="http://custom:8080/api",
            timeout=15,
            refresh_interval=120,
        )
        assert config.base_url == "http://custom:8080/api"
        assert config.timeout == 15
        assert config.refresh_interval == 120


class TestUnitreeGo2Locations:
    """Test cases for UnitreeGo2Locations background plugin."""

    @patch("backgrounds.plugins.unitree_go2_locations.UnitreeGo2LocationsProvider")
    def test_initialization(self, mock_provider_class):
        """Test background initialization creates provider and starts it."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = UnitreeGo2LocationsConfig()
        background = UnitreeGo2Locations(config)

        assert background.config is config
        assert background.locations_provider == mock_provider
        mock_provider_class.assert_called_once_with(
            base_url="http://localhost:5000/maps/locations/list",
            timeout=5,
            refresh_interval=30,
        )
        mock_provider.start.assert_called_once()

    @patch("backgrounds.plugins.unitree_go2_locations.UnitreeGo2LocationsProvider")
    def test_initialization_with_custom_config(self, mock_provider_class):
        """Test initialization with custom configuration parameters."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = UnitreeGo2LocationsConfig(
            base_url="http://custom:8080/api",
            timeout=15,
            refresh_interval=120,
        )
        UnitreeGo2Locations(config)

        mock_provider_class.assert_called_once_with(
            base_url="http://custom:8080/api",
            timeout=15,
            refresh_interval=120,
        )

    @patch("backgrounds.plugins.unitree_go2_locations.UnitreeGo2LocationsProvider")
    def test_initialization_logging(self, mock_provider_class, caplog):
        """Test that initialization logs the correct message."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = UnitreeGo2LocationsConfig()
        with caplog.at_level("INFO"):
            UnitreeGo2Locations(config)

        assert "Locations Provider initialized in background" in caplog.text
        assert "base_url: http://localhost:5000/maps/locations/list" in caplog.text
        assert "refresh: 30s" in caplog.text

    @patch("backgrounds.plugins.unitree_go2_locations.UnitreeGo2LocationsProvider")
    def test_provider_start_called(self, mock_provider_class):
        """Test that provider.start() is called during initialization."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = UnitreeGo2LocationsConfig()
        UnitreeGo2Locations(config)

        mock_provider.start.assert_called_once()

    @patch("backgrounds.plugins.unitree_go2_locations.UnitreeGo2LocationsProvider")
    def test_config_stored(self, mock_provider_class):
        """Test that config is stored correctly."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = UnitreeGo2LocationsConfig(timeout=10)
        background = UnitreeGo2Locations(config)

        assert background.config is config
        assert background.config.timeout == 10
