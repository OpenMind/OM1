# tests/actions/gps/test_gps_fabric_connector.py
"""Unit tests for the GPS Fabric action connector."""

import logging
import sys
from unittest.mock import MagicMock, patch

import pytest
import requests

from actions.gps.interface import GPSAction, GPSInput

# Mock IOProvider before importing the connector
sys.modules["providers.io_provider"] = MagicMock()


class TestGPSFabricConfig:
    """Tests for GPSFabricConfig."""

    def test_config_default_values(self):
        """Test default configuration values."""
        from actions.gps.connector.fabric import GPSFabricConfig

        config = GPSFabricConfig()
        assert config.fabric_endpoint == "http://localhost:8545"

    def test_config_custom_values(self):
        """Test configuration with custom values."""
        from actions.gps.connector.fabric import GPSFabricConfig

        config = GPSFabricConfig(fabric_endpoint="http://remote-node:8545")
        assert config.fabric_endpoint == "http://remote-node:8545"


class TestGPSFabricConnector:
    """Tests for GPSFabricConnector."""

    @pytest.fixture
    def mock_io_provider(self):
        """Create a mock IOProvider."""
        mock_io = MagicMock()
        mock_io.get_dynamic_variable.side_effect = lambda key: {
            "latitude": 40.7128,
            "longitude": -74.0060,
            "yaw_deg": 90.0,
        }.get(key)
        return mock_io

    def test_connector_initialization(self, mock_io_provider):
        """Test connector initialization."""
        with patch(
            "actions.gps.connector.fabric.IOProvider", return_value=mock_io_provider
        ):
            from actions.gps.connector.fabric import GPSFabricConfig, GPSFabricConnector

            config = GPSFabricConfig(fabric_endpoint="http://test-endpoint")
            connector = GPSFabricConnector(config)

            assert connector.io_provider == mock_io_provider
            assert connector.fabric_endpoint == "http://test-endpoint"

    @pytest.mark.asyncio
    async def test_connect_share_location_success(self, mock_io_provider, caplog):
        """Test successful location sharing to Fabric network."""
        with patch(
            "actions.gps.connector.fabric.IOProvider", return_value=mock_io_provider
        ):
            from actions.gps.connector.fabric import GPSFabricConfig, GPSFabricConnector

            mock_response = MagicMock()
            mock_response.json.return_value = {"result": True}

            with patch("requests.post", return_value=mock_response) as mock_post:
                config = GPSFabricConfig()
                connector = GPSFabricConnector(config)

                gps_input = GPSInput(action=GPSAction.SHARE_LOCATION)
                with caplog.at_level(logging.INFO):
                    await connector.connect(gps_input)

                # Verify API call
                mock_post.assert_called_once()
                args, kwargs = mock_post.call_args
                assert kwargs["json"]["method"] == "omp2p_shareStatus"
                assert kwargs["json"]["params"][0]["latitude"] == 40.7128

                # Check success logs
                assert "Coordinates shared successfully" in caplog.text

    @pytest.mark.asyncio
    async def test_connect_share_location_api_error(self, mock_io_provider, caplog):
        """Test handles API error response."""
        with patch(
            "actions.gps.connector.fabric.IOProvider", return_value=mock_io_provider
        ):
            from actions.gps.connector.fabric import GPSFabricConfig, GPSFabricConnector

            mock_response = MagicMock()
            mock_response.json.return_value = {"error": "Unauthorized"}

            with patch("requests.post", return_value=mock_response):
                config = GPSFabricConfig()
                connector = GPSFabricConnector(config)

                gps_input = GPSInput(action=GPSAction.SHARE_LOCATION)
                await connector.connect(gps_input)

                # Check error logs
                assert "Failed to share coordinates" in caplog.text

    @pytest.mark.asyncio
    async def test_connect_share_location_no_coordinates(
        self, mock_io_provider, caplog
    ):
        """Test handle scenario where coordinates are missing."""
        mock_io_provider.get_dynamic_variable.side_effect = None
        mock_io_provider.get_dynamic_variable.return_value = None

        with patch(
            "actions.gps.connector.fabric.IOProvider", return_value=mock_io_provider
        ):
            from actions.gps.connector.fabric import GPSFabricConfig, GPSFabricConnector

            with patch("requests.post") as mock_post:
                config = GPSFabricConfig()
                connector = GPSFabricConnector(config)

                gps_input = GPSInput(action=GPSAction.SHARE_LOCATION)
                with caplog.at_level(logging.ERROR):
                    await connector.connect(gps_input)

                # Should not make API call if coords missing
                mock_post.assert_not_called()
                assert "Coordinates not available" in caplog.text

    @pytest.mark.asyncio
    async def test_connect_request_exception(self, mock_io_provider, caplog):
        """Test handles network/request exceptions."""
        with patch(
            "actions.gps.connector.fabric.IOProvider", return_value=mock_io_provider
        ):
            from actions.gps.connector.fabric import GPSFabricConfig, GPSFabricConnector

            with patch(
                "requests.post",
                side_effect=requests.RequestException("Connection refused"),
            ):
                config = GPSFabricConfig()
                connector = GPSFabricConnector(config)

                gps_input = GPSInput(action=GPSAction.SHARE_LOCATION)
                await connector.connect(gps_input)

                assert "Error sending coordinates: Connection refused" in caplog.text

    def test_connector_inherits_from_action_connector(self):
        """Test that GPSFabricConnector inherits from ActionConnector."""
        from actions.base import ActionConnector
        from actions.gps.connector.fabric import GPSFabricConnector

        assert issubclass(GPSFabricConnector, ActionConnector)
