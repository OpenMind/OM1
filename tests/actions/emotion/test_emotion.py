# tests/actions/emotion/test_connector.py
"""Unit tests for the Emotion action connector."""

import logging
import sys
from unittest.mock import MagicMock, patch

import pytest

from actions.emotion.interface import EmotionAction, EmotionInput

# Mock the unitree SDK module before importing the connector
mock_audio_client_module = MagicMock()
sys.modules["unitree.unitree_sdk2py.g1.audio.g1_audio_client"] = (
    mock_audio_client_module
)


class TestEmotionUnitreeConfig:
    """Tests for EmotionUnitreeConfig."""

    def test_config_default_values(self):
        """Test default configuration values."""
        from actions.emotion.connector.unitree_sdk import EmotionUnitreeConfig

        config = EmotionUnitreeConfig()
        assert config.unitree_ethernet == ""

    def test_config_custom_ethernet(self):
        """Test configuration with custom ethernet adapter."""
        from actions.emotion.connector.unitree_sdk import EmotionUnitreeConfig

        config = EmotionUnitreeConfig(unitree_ethernet="eth0")
        assert config.unitree_ethernet == "eth0"

    def test_config_allows_extra_fields(self):
        """Test that config allows extra fields (pydantic ConfigDict)."""
        from actions.emotion.connector.unitree_sdk import EmotionUnitreeConfig

        config = EmotionUnitreeConfig(unitree_ethernet="eth0", extra_field="value")
        assert config.unitree_ethernet == "eth0"


class TestEmotionUnitreeConnector:
    """Tests for EmotionUnitreeConnector."""

    @pytest.fixture
    def mock_audio_client(self):
        """Create a mock AudioClient instance."""
        mock_client = MagicMock()
        mock_client.SetTimeout = MagicMock()
        mock_client.Init = MagicMock()
        mock_client.LedControl = MagicMock()
        mock_client.LedControlNoReply = MagicMock()
        return mock_client

    def test_connector_init_without_ethernet(self):
        """Test connector initialization without ethernet adapter."""
        from actions.emotion.connector.unitree_sdk import (
            EmotionUnitreeConfig,
            EmotionUnitreeConnector,
        )

        config = EmotionUnitreeConfig(unitree_ethernet="")
        connector = EmotionUnitreeConnector(config)

        assert connector.ao_client is None
        assert connector.unitree_ethernet == ""

    def test_connector_init_with_ethernet(self, mock_audio_client):
        """Test connector initialization with ethernet adapter."""
        from actions.emotion.connector.unitree_sdk import (
            EmotionUnitreeConfig,
            EmotionUnitreeConnector,
        )

        # Patch AudioClient at the module level where it's imported
        with patch(
            "actions.emotion.connector.unitree_sdk.AudioClient",
            return_value=mock_audio_client,
        ):
            config = EmotionUnitreeConfig(unitree_ethernet="eth0")
            connector = EmotionUnitreeConnector(config)

            assert connector.ao_client is mock_audio_client
            assert connector.unitree_ethernet == "eth0"
            mock_audio_client.SetTimeout.assert_called_once_with(10.0)
            mock_audio_client.Init.assert_called_once()
            mock_audio_client.LedControl.assert_called_once_with(0, 255, 0)

    @pytest.mark.asyncio
    async def test_connect_without_client_logs_error(self, caplog):
        """Test connect logs error when no client is available."""
        from actions.emotion.connector.unitree_sdk import (
            EmotionUnitreeConfig,
            EmotionUnitreeConnector,
        )

        config = EmotionUnitreeConfig(unitree_ethernet="")
        connector = EmotionUnitreeConnector(config)

        emotion_input = EmotionInput(action=EmotionAction.HAPPY)

        with caplog.at_level(logging.ERROR):
            await connector.connect(emotion_input)

        assert "No Unitree Emotion Client" in caplog.text

    @pytest.mark.asyncio
    async def test_connect_happy_emotion(self, mock_audio_client):
        """Test connect with HAPPY emotion sets green LED."""
        from actions.emotion.connector.unitree_sdk import (
            EmotionUnitreeConfig,
            EmotionUnitreeConnector,
        )

        with patch(
            "actions.emotion.connector.unitree_sdk.AudioClient",
            return_value=mock_audio_client,
        ):
            config = EmotionUnitreeConfig(unitree_ethernet="eth0")
            connector = EmotionUnitreeConnector(config)

            # Reset mock to clear init calls
            mock_audio_client.LedControlNoReply.reset_mock()

            emotion_input = EmotionInput(action=EmotionAction.HAPPY)
            await connector.connect(emotion_input)

            # Green LED for happy (R=0, G=255, B=0)
            mock_audio_client.LedControlNoReply.assert_called_with(0, 255, 0)

    @pytest.mark.asyncio
    async def test_connect_sad_emotion(self, mock_audio_client):
        """Test connect with SAD emotion sets yellow LED."""
        from actions.emotion.connector.unitree_sdk import (
            EmotionUnitreeConfig,
            EmotionUnitreeConnector,
        )

        with patch(
            "actions.emotion.connector.unitree_sdk.AudioClient",
            return_value=mock_audio_client,
        ):
            config = EmotionUnitreeConfig(unitree_ethernet="eth0")
            connector = EmotionUnitreeConnector(config)

            mock_audio_client.LedControlNoReply.reset_mock()

            emotion_input = EmotionInput(action=EmotionAction.SAD)
            await connector.connect(emotion_input)

            # Yellow LED for sad (R=255, G=255, B=0)
            mock_audio_client.LedControlNoReply.assert_called_with(255, 255, 0)

    @pytest.mark.asyncio
    async def test_connect_mad_emotion(self, mock_audio_client):
        """Test connect with MAD emotion sets red LED."""
        from actions.emotion.connector.unitree_sdk import (
            EmotionUnitreeConfig,
            EmotionUnitreeConnector,
        )

        with patch(
            "actions.emotion.connector.unitree_sdk.AudioClient",
            return_value=mock_audio_client,
        ):
            config = EmotionUnitreeConfig(unitree_ethernet="eth0")
            connector = EmotionUnitreeConnector(config)

            mock_audio_client.LedControlNoReply.reset_mock()

            emotion_input = EmotionInput(action=EmotionAction.MAD)
            await connector.connect(emotion_input)

            # Red LED for mad (R=255, G=0, B=0)
            mock_audio_client.LedControlNoReply.assert_called_with(255, 0, 0)

    @pytest.mark.asyncio
    async def test_connect_curious_emotion(self, mock_audio_client):
        """Test connect with CURIOUS emotion sets blue LED."""
        from actions.emotion.connector.unitree_sdk import (
            EmotionUnitreeConfig,
            EmotionUnitreeConnector,
        )

        with patch(
            "actions.emotion.connector.unitree_sdk.AudioClient",
            return_value=mock_audio_client,
        ):
            config = EmotionUnitreeConfig(unitree_ethernet="eth0")
            connector = EmotionUnitreeConnector(config)

            mock_audio_client.LedControlNoReply.reset_mock()

            emotion_input = EmotionInput(action=EmotionAction.CURIOUS)
            await connector.connect(emotion_input)

            # Blue LED for curious (R=0, G=0, B=255)
            mock_audio_client.LedControlNoReply.assert_called_with(0, 0, 255)

    @pytest.mark.asyncio
    async def test_connect_unknown_emotion(self, mock_audio_client, caplog):
        """Test connect with unknown emotion logs info message."""
        from actions.emotion.connector.unitree_sdk import (
            EmotionUnitreeConfig,
            EmotionUnitreeConnector,
        )

        with patch(
            "actions.emotion.connector.unitree_sdk.AudioClient",
            return_value=mock_audio_client,
        ):
            config = EmotionUnitreeConfig(unitree_ethernet="eth0")
            connector = EmotionUnitreeConnector(config)

            # Create a mock input with an unknown action
            mock_input = MagicMock()
            mock_input.action = "unknown_emotion"

            with caplog.at_level(logging.INFO):
                await connector.connect(mock_input)

            assert "Unknown emotion" in caplog.text

    def test_tick_method(self):
        """Test tick method sleeps for 5 seconds."""
        from actions.emotion.connector.unitree_sdk import (
            EmotionUnitreeConfig,
            EmotionUnitreeConnector,
        )

        with patch("time.sleep") as mock_sleep:
            config = EmotionUnitreeConfig(unitree_ethernet="")
            connector = EmotionUnitreeConnector(config)

            connector.tick()

            mock_sleep.assert_called_once_with(5)

    def test_connector_inherits_from_action_connector(self):
        """Test that EmotionUnitreeConnector inherits from ActionConnector."""
        from actions.base import ActionConnector
        from actions.emotion.connector.unitree_sdk import EmotionUnitreeConnector

        assert issubclass(EmotionUnitreeConnector, ActionConnector)
