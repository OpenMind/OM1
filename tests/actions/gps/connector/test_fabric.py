"""Tests for GPS Fabric Connector."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from actions.gps.connector.fabric import GPSFabricConfig, GPSFabricConnector
from actions.gps.interface import GPSAction, GPSInput


class TestGPSFabricConfig:
    """Tests for GPSFabricConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = GPSFabricConfig()
        assert config.fabric_endpoint == "http://localhost:8545"
        assert config.request_timeout == 10

    def test_custom_values(self):
        """Test custom configuration values."""
        config = GPSFabricConfig(
            fabric_endpoint="http://custom:9000",
            request_timeout=30,
        )
        assert config.fabric_endpoint == "http://custom:9000"
        assert config.request_timeout == 30


class TestGPSFabricConnector:
    """Tests for GPSFabricConnector class."""

    @pytest.fixture
    def mock_io_provider(self):
        """Create a mock IO provider."""
        return MagicMock()

    @pytest.fixture
    def connector(self, mock_io_provider):
        """Create a connector instance for testing."""
        with patch("actions.gps.connector.fabric.IOProvider") as mock_io_class:
            mock_io_class.return_value = mock_io_provider
            config = GPSFabricConfig(
                fabric_endpoint="http://test:8545",
                request_timeout=5,
            )
            connector = GPSFabricConnector(config)
        return connector

    def test_initialization(self, connector):
        """Test connector initialization."""
        assert connector.fabric_endpoint == "http://test:8545"
        assert connector.request_timeout == 5

    @pytest.mark.asyncio
    async def test_connect_share_location(self, connector):
        """Test connect method with SHARE_LOCATION action."""
        connector.send_coordinates = MagicMock(return_value=True)
        output = GPSInput(action=GPSAction.SHARE_LOCATION)

        await connector.connect(output)

        connector.send_coordinates.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_other_action(self, connector):
        """Test connect method with non-SHARE_LOCATION action."""
        connector.send_coordinates = MagicMock()
        # Create a mock GPSInput with a different action
        output = MagicMock(spec=GPSInput)
        output.action = MagicMock()

        await connector.connect(output)

        connector.send_coordinates.assert_not_called()

    def test_send_coordinates_success(self, connector, mock_io_provider):
        """Test successful coordinate sending."""
        mock_io_provider.get_dynamic_variable.side_effect = lambda key: {
            "latitude": 37.7749,
            "longitude": -122.4194,
            "yaw_deg": 90.0,
        }.get(key)

        with patch("actions.gps.connector.fabric.requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.ok = True
            mock_response.json.return_value = {"result": True}
            mock_post.return_value = mock_response

            result = connector.send_coordinates()

            assert result is True
            mock_post.assert_called_once()

    def test_send_coordinates_missing_latitude(self, connector, mock_io_provider):
        """Test coordinate sending when latitude is None."""
        mock_io_provider.get_dynamic_variable.side_effect = lambda key: {
            "latitude": None,
            "longitude": -122.4194,
            "yaw_deg": 90.0,
        }.get(key)

        result = connector.send_coordinates()

        assert result is False

    def test_send_coordinates_missing_longitude(self, connector, mock_io_provider):
        """Test coordinate sending when longitude is None."""
        mock_io_provider.get_dynamic_variable.side_effect = lambda key: {
            "latitude": 37.7749,
            "longitude": None,
            "yaw_deg": 90.0,
        }.get(key)

        result = connector.send_coordinates()

        assert result is False

    def test_send_coordinates_missing_yaw(self, connector, mock_io_provider):
        """Test coordinate sending when yaw is None."""
        mock_io_provider.get_dynamic_variable.side_effect = lambda key: {
            "latitude": 37.7749,
            "longitude": -122.4194,
            "yaw_deg": None,
        }.get(key)

        result = connector.send_coordinates()

        assert result is False

    def test_send_coordinates_http_error(self, connector, mock_io_provider):
        """Test coordinate sending when HTTP error occurs."""
        mock_io_provider.get_dynamic_variable.side_effect = lambda key: {
            "latitude": 37.7749,
            "longitude": -122.4194,
            "yaw_deg": 90.0,
        }.get(key)

        with patch("actions.gps.connector.fabric.requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.ok = False
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"
            mock_post.return_value = mock_response

            result = connector.send_coordinates()

            assert result is False

    def test_send_coordinates_json_decode_error(self, connector, mock_io_provider):
        """Test coordinate sending when JSON decode fails."""
        mock_io_provider.get_dynamic_variable.side_effect = lambda key: {
            "latitude": 37.7749,
            "longitude": -122.4194,
            "yaw_deg": 90.0,
        }.get(key)

        with patch("actions.gps.connector.fabric.requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.ok = True
            mock_response.json.side_effect = requests.exceptions.JSONDecodeError(
                "Error", "doc", 0
            )
            mock_post.return_value = mock_response

            result = connector.send_coordinates()

            assert result is False

    def test_send_coordinates_jsonrpc_error(self, connector, mock_io_provider):
        """Test coordinate sending when JSON-RPC error is returned."""
        mock_io_provider.get_dynamic_variable.side_effect = lambda key: {
            "latitude": 37.7749,
            "longitude": -122.4194,
            "yaw_deg": 90.0,
        }.get(key)

        with patch("actions.gps.connector.fabric.requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.ok = True
            mock_response.json.return_value = {
                "error": {"code": -32600, "message": "Invalid Request"}
            }
            mock_post.return_value = mock_response

            result = connector.send_coordinates()

            assert result is False

    def test_send_coordinates_no_result(self, connector, mock_io_provider):
        """Test coordinate sending when result is empty."""
        mock_io_provider.get_dynamic_variable.side_effect = lambda key: {
            "latitude": 37.7749,
            "longitude": -122.4194,
            "yaw_deg": 90.0,
        }.get(key)

        with patch("actions.gps.connector.fabric.requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.ok = True
            mock_response.json.return_value = {"result": None}
            mock_post.return_value = mock_response

            result = connector.send_coordinates()

            assert result is False

    def test_send_coordinates_timeout(self, connector, mock_io_provider):
        """Test coordinate sending when request times out."""
        mock_io_provider.get_dynamic_variable.side_effect = lambda key: {
            "latitude": 37.7749,
            "longitude": -122.4194,
            "yaw_deg": 90.0,
        }.get(key)

        with patch("actions.gps.connector.fabric.requests.post") as mock_post:
            mock_post.side_effect = requests.Timeout()

            result = connector.send_coordinates()

            assert result is False

    def test_send_coordinates_connection_error(self, connector, mock_io_provider):
        """Test coordinate sending when connection error occurs."""
        mock_io_provider.get_dynamic_variable.side_effect = lambda key: {
            "latitude": 37.7749,
            "longitude": -122.4194,
            "yaw_deg": 90.0,
        }.get(key)

        with patch("actions.gps.connector.fabric.requests.post") as mock_post:
            mock_post.side_effect = requests.ConnectionError("Connection refused")

            result = connector.send_coordinates()

            assert result is False

    def test_send_coordinates_request_exception(self, connector, mock_io_provider):
        """Test coordinate sending when general request exception occurs."""
        mock_io_provider.get_dynamic_variable.side_effect = lambda key: {
            "latitude": 37.7749,
            "longitude": -122.4194,
            "yaw_deg": 90.0,
        }.get(key)

        with patch("actions.gps.connector.fabric.requests.post") as mock_post:
            mock_post.side_effect = requests.RequestException("Unknown error")

            result = connector.send_coordinates()

            assert result is False
