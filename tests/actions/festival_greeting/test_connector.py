"""Unit tests for FestivalGreetingElevenLabsTTS connector."""

from unittest.mock import MagicMock, patch

import pytest

from actions.festival_greeting.connector.elevenlabs_tts import (
    FestivalGreetingElevenLabsTTSConfig,
    FestivalGreetingElevenLabsTTSConnector,
)
from actions.festival_greeting.interface import FestivalGreetingInput, FestivalType


@pytest.fixture
def mock_config():
    """Create a mock config for testing."""
    return FestivalGreetingElevenLabsTTSConfig(
        elevenlabs_api_key="test_key",
        voice_id="test_voice",
        model_id="eleven_flash_v2_5",
        output_format="mp3_44100_128",
        silence_rate=0,
    )


@pytest.fixture
def minimal_config():
    """Create a minimal config with defaults."""
    return FestivalGreetingElevenLabsTTSConfig()


class TestFestivalGreetingElevenLabsTTSConfig:
    """Test cases for FestivalGreetingElevenLabsTTSConfig."""

    def test_config_defaults(self):
        """Test config default values."""
        config = FestivalGreetingElevenLabsTTSConfig()
        assert config.elevenlabs_api_key is None
        assert config.voice_id == "JBFqnCBsd6RMkjVDRZzb"
        assert config.model_id == "eleven_flash_v2_5"
        assert config.output_format == "mp3_44100_128"
        assert config.silence_rate == 0

    def test_config_custom_values(self):
        """Test config with custom values."""
        config = FestivalGreetingElevenLabsTTSConfig(
            elevenlabs_api_key="custom_key",
            voice_id="custom_voice",
            model_id="custom_model",
            output_format="custom_format",
            silence_rate=5,
        )
        assert config.elevenlabs_api_key == "custom_key"
        assert config.voice_id == "custom_voice"
        assert config.model_id == "custom_model"
        assert config.output_format == "custom_format"
        assert config.silence_rate == 5


class TestFestivalGreetingElevenLabsTTSConnector:
    """Test cases for FestivalGreetingElevenLabsTTSConnector."""

    @patch("actions.festival_greeting.connector.elevenlabs_tts.open_zenoh_session")
    @patch("actions.festival_greeting.connector.elevenlabs_tts.IOProvider")
    @patch("actions.festival_greeting.connector.elevenlabs_tts.ElevenLabsTTSProvider")
    def test_connector_initialization(self, mock_tts, mock_io, mock_zenoh, mock_config):
        """Test connector initialization."""
        mock_zenoh.return_value = MagicMock()
        mock_io.return_value = MagicMock()
        mock_tts.return_value = MagicMock()

        connector = FestivalGreetingElevenLabsTTSConnector(mock_config)
        assert connector.config == mock_config

    @patch("actions.festival_greeting.connector.elevenlabs_tts.open_zenoh_session")
    @patch("actions.festival_greeting.connector.elevenlabs_tts.IOProvider")
    @patch("actions.festival_greeting.connector.elevenlabs_tts.ElevenLabsTTSProvider")
    def test_connector_with_minimal_config(
        self, mock_tts, mock_io, mock_zenoh, minimal_config
    ):
        """Test connector with minimal config (defaults)."""
        mock_zenoh.return_value = MagicMock()
        mock_io.return_value = MagicMock()
        mock_tts.return_value = MagicMock()

        connector = FestivalGreetingElevenLabsTTSConnector(minimal_config)
        assert connector.config == minimal_config

    def test_connector_config_validation(self):
        """Test that config validation works."""
        # Should not raise with valid config
        config = FestivalGreetingElevenLabsTTSConfig(
            voice_id="test", model_id="test", output_format="test"
        )
        assert config.voice_id == "test"

    @patch("actions.festival_greeting.connector.elevenlabs_tts.open_zenoh_session")
    @patch("actions.festival_greeting.connector.elevenlabs_tts.IOProvider")
    @patch("actions.festival_greeting.connector.elevenlabs_tts.ElevenLabsTTSProvider")
    def test_connector_handles_none_api_key(
        self, mock_tts, mock_io, mock_zenoh, minimal_config
    ):
        """Test connector handles None API key gracefully."""
        mock_zenoh.return_value = MagicMock()
        mock_io.return_value = MagicMock()
        mock_tts.return_value = MagicMock()

        # Should not raise even with None API key
        connector = FestivalGreetingElevenLabsTTSConnector(minimal_config)
        assert connector.config.elevenlabs_api_key is None


class TestEdgeCases:
    """Edge case tests for connector."""

    def test_config_empty_strings(self):
        """Test config with empty string values."""
        config = FestivalGreetingElevenLabsTTSConfig(
            voice_id="", model_id="", output_format=""
        )
        assert config.voice_id == ""
        assert config.model_id == ""
        assert config.output_format == ""

    def test_config_negative_silence_rate(self):
        """Test config with negative silence_rate."""
        config = FestivalGreetingElevenLabsTTSConfig(silence_rate=-1)
        assert config.silence_rate == -1

    def test_config_large_silence_rate(self):
        """Test config with large silence_rate."""
        config = FestivalGreetingElevenLabsTTSConfig(silence_rate=1000)
        assert config.silence_rate == 1000
