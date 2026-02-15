from unittest.mock import MagicMock, Mock, patch

import pytest
import requests as req

from actions.gps.connector.fabric import (  # noqa: E402
    GPSFabricConfig,
    GPSFabricConnector,
)
from actions.gps.interface import GPSAction, GPSInput  # noqa: E402


@pytest.fixture
def default_config():
    """Create a default config for testing."""
    return GPSFabricConfig()


@pytest.fixture
def custom_config():
    """Create a custom config for testing."""
    return GPSFabricConfig(fabric_endpoint="http://custom:8080")


@pytest.fixture
def gps_input_share():
    """Create a GPSInput instance with share location action."""
    return GPSInput(action=GPSAction.SHARE_LOCATION)


@pytest.fixture
def gps_input_idle():
    """Create a GPSInput instance with idle action."""
    return GPSInput(action=GPSAction.IDLE)


@pytest.fixture(scope="session", autouse=True)
def mock_zenoh_modules():
    """Mock zenoh modules before imports."""
    with patch.dict(
        "sys.modules",
        {
            "zenoh": MagicMock(),
            "zenoh_msgs": MagicMock(),
        },
    ):
        yield


class TestGPSFabricConfig:
    """Test the GPS Fabric configuration class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GPSFabricConfig()
        assert config.fabric_endpoint == "http://localhost:8545"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = GPSFabricConfig(fabric_endpoint="http://custom:9000")
        assert config.fabric_endpoint == "http://custom:9000"


class TestGPSFabricConnector:
    """Test the GPS Fabric connector."""

    def test_init(self, default_config):
        """Test initialization of GPSFabricConnector."""
        with patch("actions.gps.connector.fabric.IOProvider") as mock_io_provider_class:
            mock_io_instance = Mock()
            mock_io_provider_class.return_value = mock_io_instance

            connector = GPSFabricConnector(default_config)

            mock_io_provider_class.assert_called_once()
            assert connector.io_provider is not None
            assert connector.fabric_endpoint == "http://localhost:8545"

    def test_init_with_custom_config(self, custom_config):
        """Test initialization with custom configuration."""
        with patch("actions.gps.connector.fabric.IOProvider") as mock_io_provider_class:
            mock_io_instance = Mock()
            mock_io_provider_class.return_value = mock_io_instance

            connector = GPSFabricConnector(custom_config)

            assert connector.fabric_endpoint == "http://custom:8080"

    @pytest.mark.asyncio
    async def test_connect_share_location(self, default_config, gps_input_share):
        """Test connect with share location action."""
        with patch("actions.gps.connector.fabric.IOProvider") as mock_io_provider_class:
            mock_io_instance = Mock()
            mock_io_provider_class.return_value = mock_io_instance

            connector = GPSFabricConnector(default_config)

            with patch.object(connector, "send_coordinates") as mock_send:
                await connector.connect(gps_input_share)
                mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_idle(self, default_config, gps_input_idle):
        """Test connect with idle action does not send coordinates."""
        with patch("actions.gps.connector.fabric.IOProvider") as mock_io_provider_class:
            mock_io_instance = Mock()
            mock_io_provider_class.return_value = mock_io_instance

            connector = GPSFabricConnector(default_config)

            with patch.object(connector, "send_coordinates") as mock_send:
                await connector.connect(gps_input_idle)
                mock_send.assert_not_called()

    def test_send_coordinates_success(self, default_config):
        """Test send_coordinates with successful response."""
        with (
            patch("actions.gps.connector.fabric.IOProvider") as mock_io_provider_class,
            patch("actions.gps.connector.fabric.requests") as mock_requests,
        ):
            mock_io_instance = Mock()
            mock_io_instance.get_dynamic_variable.side_effect = lambda x: {
                "latitude": 37.7749,
                "longitude": -122.4194,
                "yaw_deg": 90.0,
            }.get(x)
            mock_io_provider_class.return_value = mock_io_instance

            mock_response = Mock()
            mock_response.json.return_value = {"result": True}
            mock_requests.post.return_value = mock_response

            connector = GPSFabricConnector(default_config)
            connector.send_coordinates()

            mock_requests.post.assert_called_once()
            call_args = mock_requests.post.call_args
            assert call_args[0][0] == "http://localhost:8545"
            assert call_args[1]["json"]["method"] == "omp2p_shareStatus"

    def test_send_coordinates_no_coordinates(self, default_config):
        """Test send_coordinates when no coordinates available."""
        with (
            patch("actions.gps.connector.fabric.IOProvider") as mock_io_provider_class,
            patch("actions.gps.connector.fabric.requests") as mock_requests,
        ):
            mock_io_instance = Mock()
            mock_io_instance.get_dynamic_variable.return_value = None
            mock_io_provider_class.return_value = mock_io_instance

            connector = GPSFabricConnector(default_config)
            result = connector.send_coordinates()

            assert result is None
            mock_requests.post.assert_not_called()

    def test_send_coordinates_request_failure(self, default_config):
        """Test send_coordinates handles request exception."""
        with (
            patch("actions.gps.connector.fabric.IOProvider") as mock_io_provider_class,
            patch("actions.gps.connector.fabric.requests") as mock_requests,
        ):
            mock_io_instance = Mock()
            mock_io_instance.get_dynamic_variable.side_effect = lambda x: {
                "latitude": 37.7749,
                "longitude": -122.4194,
                "yaw_deg": 90.0,
            }.get(x)
            mock_io_provider_class.return_value = mock_io_instance

            mock_requests.post.side_effect = req.RequestException("Connection error")
            mock_requests.RequestException = req.RequestException

            connector = GPSFabricConnector(default_config)
            # Should not raise exception
            connector.send_coordinates()

            mock_requests.post.assert_called_once()
