from unittest.mock import MagicMock, patch

from backgrounds.plugins.unitree_g1_locations import (
    UnitreeG1Locations,
    UnitreeG1LocationsConfig,
)


class TestUnitreeG1LocationsConfig:
    """Test cases for UnitreeG1LocationsConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = UnitreeG1LocationsConfig()
        assert config.base_url == "http://localhost:5000/maps/locations/list"
        assert config.timeout == 5
        assert config.refresh_interval == 30

    def test_custom_base_url(self):
        """Test custom base_url configuration."""
        config = UnitreeG1LocationsConfig(base_url="http://example.com/locations")
        assert config.base_url == "http://example.com/locations"

    def test_custom_timeout(self):
        """Test custom timeout configuration."""
        config = UnitreeG1LocationsConfig(timeout=10)
        assert config.timeout == 10

    def test_custom_refresh_interval(self):
        """Test custom refresh_interval configuration."""
        config = UnitreeG1LocationsConfig(refresh_interval=60)
        assert config.refresh_interval == 60

    def test_all_custom_values(self):
        """Test configuration with all custom values."""
        config = UnitreeG1LocationsConfig(
            base_url="http://custom:8080/api",
            timeout=15,
            refresh_interval=120,
        )
        assert config.base_url == "http://custom:8080/api"
        assert config.timeout == 15
        assert config.refresh_interval == 120


class TestUnitreeG1Locations:
    """Test cases for UnitreeG1Locations background plugin."""

    @patch("backgrounds.plugins.unitree_g1_locations.UnitreeG1LocationsProvider")
    def test_initialization(self, mock_provider_class):
        """Test background initialization creates provider and starts it."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = UnitreeG1LocationsConfig()
        background = UnitreeG1Locations(config)

        assert background.config is config
        assert background.locations_provider == mock_provider
        mock_provider_class.assert_called_once_with(
            base_url="http://localhost:5000/maps/locations/list",
            timeout=5,
            refresh_interval=30,
        )
        mock_provider.start.assert_called_once()

    @patch("backgrounds.plugins.unitree_g1_locations.UnitreeG1LocationsProvider")
    def test_initialization_with_custom_config(self, mock_provider_class):
        """Test initialization with custom configuration parameters."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = UnitreeG1LocationsConfig(
            base_url="http://custom:8080/api",
            timeout=15,
            refresh_interval=120,
        )
        UnitreeG1Locations(config)

        mock_provider_class.assert_called_once_with(
            base_url="http://custom:8080/api",
            timeout=15,
            refresh_interval=120,
        )

    @patch("backgrounds.plugins.unitree_g1_locations.UnitreeG1LocationsProvider")
    def test_initialization_logging(self, mock_provider_class, caplog):
        """Test that initialization logs the correct message."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = UnitreeG1LocationsConfig()
        with caplog.at_level("INFO"):
            UnitreeG1Locations(config)

        assert "G1 Locations Provider initialized in background" in caplog.text
        assert "base_url: http://localhost:5000/maps/locations/list" in caplog.text
        assert "refresh: 30s" in caplog.text

    @patch("backgrounds.plugins.unitree_g1_locations.UnitreeG1LocationsProvider")
    def test_provider_start_called(self, mock_provider_class):
        """Test that provider.start() is called during initialization."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = UnitreeG1LocationsConfig()
        UnitreeG1Locations(config)

        mock_provider.start.assert_called_once()

    @patch("backgrounds.plugins.unitree_g1_locations.UnitreeG1LocationsProvider")
    def test_config_stored(self, mock_provider_class):
        """Test that config is stored correctly."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = UnitreeG1LocationsConfig(timeout=10)
        background = UnitreeG1Locations(config)

        assert background.config is config
        assert background.config.timeout == 10
