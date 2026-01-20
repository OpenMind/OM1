# tests/actions/remember_location/test_unitree_go2_location_connector.py
"""Unit tests for the Unitree Go2 Remember Location connector."""

import asyncio
import logging
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from actions.remember_location.interface import RememberLocationInput

# Mock providers before importing the connector
sys.modules["providers.elevenlabs_tts_provider"] = MagicMock()


class TestUnitreeGo2RememberLocationConfig:
    """Tests for UnitreeGo2RememberLocationConfig."""

    def test_config_default_values(self):
        """Test default configuration values."""
        from actions.remember_location.connector.unitree_go2_location import (
            UnitreeGo2RememberLocationConfig,
        )

        config = UnitreeGo2RememberLocationConfig()
        assert config.base_url == "http://localhost:5000/maps/locations/add/slam"
        assert config.timeout == 5
        assert config.map_name == "map"


class TestUnitreeGo2RememberLocationConnector:
    """Tests for UnitreeGo2RememberLocationConnector."""

    @pytest.fixture
    def mock_tts_provider(self):
        """Create a mock ElevenLabsTTSProvider."""
        return MagicMock()

    def test_connector_initialization(self, mock_tts_provider):
        """Test connector initialization."""
        with patch(
            "actions.remember_location.connector.unitree_go2_location.ElevenLabsTTSProvider",
            return_value=mock_tts_provider,
        ):
            from actions.remember_location.connector.unitree_go2_location import (
                UnitreeGo2RememberLocationConfig,
                UnitreeGo2RememberLocationConnector,
            )

            config = UnitreeGo2RememberLocationConfig()
            connector = UnitreeGo2RememberLocationConnector(config)

            assert connector.elevenlabs_provider == mock_tts_provider
            assert connector.base_url == config.base_url
            assert connector.timeout == config.timeout
            assert connector.map_name == config.map_name

    @pytest.mark.asyncio
    async def test_connect_success(self, mock_tts_provider):
        """Test connect method success scenario."""
        with patch(
            "actions.remember_location.connector.unitree_go2_location.ElevenLabsTTSProvider",
            return_value=mock_tts_provider,
        ):
            from actions.remember_location.connector.unitree_go2_location import (
                UnitreeGo2RememberLocationConfig,
                UnitreeGo2RememberLocationConnector,
            )

            config = UnitreeGo2RememberLocationConfig()
            connector = UnitreeGo2RememberLocationConnector(config)

            # Mock aiohttp
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text.return_value = "Success"

            mock_session = MagicMock()
            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_session.__aenter__.return_value = mock_session

            with patch("aiohttp.ClientSession", return_value=mock_session):
                remember_input = RememberLocationInput(action="charging_station")
                await connector.connect(remember_input)

                # Check API call
                mock_session.post.assert_called_once()
                args, kwargs = mock_session.post.call_args
                assert kwargs["json"]["label"] == "charging_station"
                assert kwargs["json"]["map_name"] == "map"

                # Check TTS feedback
                mock_tts_provider.add_pending_message.assert_called_once()
                assert (
                    "charging_station"
                    in mock_tts_provider.add_pending_message.call_args[0][0]
                )

    @pytest.mark.asyncio
    async def test_connect_api_error(self, mock_tts_provider, caplog):
        """Test connect method API error scenario."""
        with patch(
            "actions.remember_location.connector.unitree_go2_location.ElevenLabsTTSProvider",
            return_value=mock_tts_provider,
        ):
            from actions.remember_location.connector.unitree_go2_location import (
                UnitreeGo2RememberLocationConfig,
                UnitreeGo2RememberLocationConnector,
            )

            config = UnitreeGo2RememberLocationConfig()
            connector = UnitreeGo2RememberLocationConnector(config)

            # Mock aiohttp error response
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_response.text.return_value = "Server Error"

            mock_session = MagicMock()
            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_session.__aenter__.return_value = mock_session

            with patch("aiohttp.ClientSession", return_value=mock_session):
                remember_input = RememberLocationInput(action="kitchen")
                with caplog.at_level(logging.ERROR):
                    await connector.connect(remember_input)

                assert "API returned 500" in caplog.text
                mock_tts_provider.add_pending_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_connect_timeout(self, mock_tts_provider, caplog):
        """Test connect method timeout scenario."""
        with patch(
            "actions.remember_location.connector.unitree_go2_location.ElevenLabsTTSProvider",
            return_value=mock_tts_provider,
        ):
            from actions.remember_location.connector.unitree_go2_location import (
                UnitreeGo2RememberLocationConfig,
                UnitreeGo2RememberLocationConnector,
            )

            config = UnitreeGo2RememberLocationConfig()
            connector = UnitreeGo2RememberLocationConnector(config)

            # Mock aiohttp timeout
            mock_session = MagicMock()
            mock_session.post.side_effect = asyncio.TimeoutError()
            mock_session.__aenter__.return_value = mock_session

            with patch("aiohttp.ClientSession", return_value=mock_session):
                remember_input = RememberLocationInput(action="living_room")
                with caplog.at_level(logging.ERROR):
                    await connector.connect(remember_input)

                assert "request timed out" in caplog.text
