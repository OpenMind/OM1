"""Tests for the GPS Fabric connector."""

import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

# Mock modules at module load time BEFORE any other imports
mock_zenoh = MagicMock()
mock_zenoh_msgs = MagicMock()
sys.modules["zenoh"] = mock_zenoh
sys.modules["zenoh_msgs"] = mock_zenoh_msgs

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


@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset all mock objects between tests."""
    mock_zenoh.reset_mock()
    mock_zenoh_msgs.reset_mock()
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

    @patch("actions.gps.connector.fabric.IOProvider")
    def test_init(self, mock_io_provider_class, default_config):
        """Test initialization of GPSFabricConnector."""
        mock_io_instance = Mock()
        mock_io_provider_class.return_value = mock_io_instance

        connector = GPSFabricConnector(default_config)

        mock_io_provider_class.assert_called_once()
        assert connector.io_provider is not None
        assert connector.fabric_endpoint == "http://localhost:8545"

    @patch("actions.gps.connector.fabric.IOProvider")
    def test_init_with_custom_config(self, mock_io_provider_class, custom_config):
        """Test initialization with custom configuration."""
        mock_io_instance = Mock()
        mock_io_provider_class.return_value = mock_io_instance

        connector = GPSFabricConnector(custom_config)

        assert connector.fabric_endpoint == "http://custom:8080"

    @pytest.mark.asyncio
    @patch("actions.gps.connector.fabric.IOProvider")
    async def test_connect_share_location(
        self, mock_io_provider_class, default_config, gps_input_share
    ):
        """Test connect with share location action."""
        mock_io_instance = Mock()
        mock_io_provider_class.return_value = mock_io_instance

        connector = GPSFabricConnector(default_config)

        with patch.object(connector, "send_coordinates") as mock_send:
            await connector.connect(gps_input_share)
            mock_send.assert_called_once()

    @pytest.mark.asyncio
    @patch("actions.gps.connector.fabric.IOProvider")
    async def test_connect_idle(
        self, mock_io_provider_class, default_config, gps_input_idle
    ):
        """Test connect with idle action does not send coordinates."""
        mock_io_instance = Mock()
        mock_io_provider_class.return_value = mock_io_instance

        connector = GPSFabricConnector(default_config)

        with patch.object(connector, "send_coordinates") as mock_send:
            await connector.connect(gps_input_idle)
            mock_send.assert_not_called()

    @patch("actions.gps.connector.fabric.requests")
    @patch("actions.gps.connector.fabric.IOProvider")
    def test_send_coordinates_success(
        self, mock_io_provider_class, mock_requests, default_config
    ):
        """Test send_coordinates with successful response."""
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

    @patch("actions.gps.connector.fabric.requests")
    @patch("actions.gps.connector.fabric.IOProvider")
    def test_send_coordinates_no_coordinates(
        self, mock_io_provider_class, mock_requests, default_config
    ):
        """Test send_coordinates when no coordinates available."""
        mock_io_instance = Mock()
        mock_io_instance.get_dynamic_variable.return_value = None
        mock_io_provider_class.return_value = mock_io_instance

        connector = GPSFabricConnector(default_config)
        result = connector.send_coordinates()

        assert result is None
        mock_requests.post.assert_not_called()

    @patch("actions.gps.connector.fabric.requests")
    @patch("actions.gps.connector.fabric.IOProvider")
    def test_send_coordinates_request_failure(
        self, mock_io_provider_class, mock_requests, default_config
    ):
        """Test send_coordinates handles request exception."""
        import requests as req

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
