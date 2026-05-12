from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inputs.base import Message
from inputs.plugins.vlm_gemini import VLMGemini, VLMGeminiConfig


def test_initialization():
    """Test basic initialization."""
    with (
        patch("inputs.plugins.vlm_gemini.IOProvider"),
        patch("inputs.plugins.vlm_gemini.VLMGeminiProvider"),
    ):
        config = VLMGeminiConfig(api_key="test-api-key")
        sensor = VLMGemini(config=config)

        assert hasattr(sensor, "messages")


@pytest.mark.asyncio
async def test_poll():
    """Test _poll method."""
    with (
        patch("inputs.plugins.vlm_gemini.IOProvider"),
        patch("inputs.plugins.vlm_gemini.VLMGeminiProvider"),
        patch("inputs.plugins.vlm_gemini.asyncio.sleep", new=AsyncMock()),
    ):
        config = VLMGeminiConfig(api_key="test-api-key")
        sensor = VLMGemini(config=config)

        result = await sensor._poll()
        assert result is None


def test_formatted_latest_buffer():
    """Test formatted_latest_buffer."""
    with (
        patch("inputs.plugins.vlm_gemini.IOProvider"),
        patch("inputs.plugins.vlm_gemini.VLMGeminiProvider"),
    ):
        config = VLMGeminiConfig(api_key="test-api-key")
        sensor = VLMGemini(config=config)

        result = sensor.formatted_latest_buffer()
        assert result is None

        test_message = Message(timestamp=123.456, message="I see a person standing in front of a building")
        sensor.messages.append(test_message)

        result = sensor.formatted_latest_buffer()
        assert isinstance(result, str)
        assert "Vision" in result
        assert "I see a person" in result
        assert len(sensor.messages) == 0


def test_initialization_missing_api_key():
    """Test initialization without API key raises ValueError."""
    with (
        patch("inputs.plugins.vlm_gemini.IOProvider"),
        patch("inputs.plugins.vlm_gemini.VLMGeminiProvider"),
    ):
        config = VLMGeminiConfig(api_key=None)
        with pytest.raises(ValueError, match="config file missing api_key"):
            VLMGemini(config=config)

        config = VLMGeminiConfig(api_key="")
        with pytest.raises(ValueError, match="config file missing api_key"):
            VLMGemini(config=config)


def test_initialization_with_use_sim():
    """Test initialization with use_sim=True uses VLMGeminiZenohProvider."""
    with (
        patch("inputs.plugins.vlm_gemini.IOProvider"),
        patch("inputs.plugins.vlm_gemini.VLMGeminiZenohProvider") as mock_zenoh_provider,
    ):
        mock_vlm = MagicMock()
        mock_zenoh_provider.return_value = mock_vlm

        config = VLMGeminiConfig(api_key="test-api-key", use_sim=True, topic="test/topic")
        VLMGemini(config=config)

        mock_zenoh_provider.assert_called_once()
        call_kwargs = mock_zenoh_provider.call_args[1]
        assert call_kwargs["base_url"] == "https://api.openmind.com/api/core/gemini"
        assert call_kwargs["api_key"] == "test-api-key"
        assert call_kwargs["topic"] == "test/topic"
        assert call_kwargs["model"] == "gemini-2.5-flash"
        assert call_kwargs["max_tokens"] == 1024
        mock_vlm.start.assert_called_once()
        mock_vlm.register_message_callback.assert_called_once()


def test_initialization_with_custom_parameters():
    """Test initialization with custom model, max_tokens, and prompt."""
    with (
        patch("inputs.plugins.vlm_gemini.IOProvider"),
        patch("inputs.plugins.vlm_gemini.VLMGeminiProvider") as mock_provider,
    ):
        mock_vlm = MagicMock()
        mock_provider.return_value = mock_vlm

        config = VLMGeminiConfig(
            api_key="test-api-key",
            model="gemini-2.5-pro",
            max_tokens=2048,
            prompt="Describe what you see in detail",
        )
        sensor = VLMGemini(config=config)

        mock_provider.assert_called_once()
        call_kwargs = mock_provider.call_args[1]
        assert call_kwargs["model"] == "gemini-2.5-pro"
        assert call_kwargs["max_tokens"] == 2048
        assert call_kwargs["prompt"] == "Describe what you see in detail"
        assert sensor.descriptor_for_LLM == "Vision"


def test_handle_vlm_message_with_content():
    """Test _handle_vlm_message with valid content."""
    with (
        patch("inputs.plugins.vlm_gemini.IOProvider"),
        patch("inputs.plugins.vlm_gemini.VLMGeminiProvider"),
    ):
        config = VLMGeminiConfig(api_key="test-api-key")
        sensor = VLMGemini(config=config)

        sensor._handle_vlm_message("Test vision message")

        assert not sensor.message_buffer.empty()
        message = sensor.message_buffer.get_nowait()
        assert message == "Test vision message"


def test_handle_vlm_message_with_empty_content():
    """Test _handle_vlm_message with empty content."""
    with (
        patch("inputs.plugins.vlm_gemini.IOProvider"),
        patch("inputs.plugins.vlm_gemini.VLMGeminiProvider"),
    ):
        config = VLMGeminiConfig(api_key="test-api-key")
        sensor = VLMGemini(config=config)

        sensor._handle_vlm_message("")
        sensor._handle_vlm_message(None)  # type: ignore

        assert sensor.message_buffer.empty()


@pytest.mark.asyncio
async def test_poll_with_message_in_buffer():
    """Test _poll when message is available in buffer."""
    with (
        patch("inputs.plugins.vlm_gemini.IOProvider"),
        patch("inputs.plugins.vlm_gemini.VLMGeminiProvider"),
        patch("inputs.plugins.vlm_gemini.asyncio.sleep", new=AsyncMock()),
    ):
        config = VLMGeminiConfig(api_key="test-api-key")
        sensor = VLMGemini(config=config)

        # Add message to buffer
        sensor.message_buffer.put("Scene description")

        result = await sensor._poll()
        assert result == "Scene description"


@pytest.mark.asyncio
async def test_raw_to_text_with_valid_input():
    """Test _raw_to_text with valid input."""
    with (
        patch("inputs.plugins.vlm_gemini.IOProvider"),
        patch("inputs.plugins.vlm_gemini.VLMGeminiProvider"),
    ):
        config = VLMGeminiConfig(api_key="test-api-key")
        sensor = VLMGemini(config=config)

        with patch("inputs.plugins.vlm_gemini.time.time", return_value=5000.0):
            result = await sensor._raw_to_text("A person walking")

        assert result is not None
        assert result.timestamp == 5000.0
        assert result.message == "A person walking"


@pytest.mark.asyncio
async def test_raw_to_text_with_none_input():
    """Test _raw_to_text with None input."""
    with (
        patch("inputs.plugins.vlm_gemini.IOProvider"),
        patch("inputs.plugins.vlm_gemini.VLMGeminiProvider"),
    ):
        config = VLMGeminiConfig(api_key="test-api-key")
        sensor = VLMGemini(config=config)

        result = await sensor._raw_to_text(None)
        assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_appends_to_messages():
    """Test raw_to_text appends messages to buffer."""
    with (
        patch("inputs.plugins.vlm_gemini.IOProvider"),
        patch("inputs.plugins.vlm_gemini.VLMGeminiProvider"),
    ):
        config = VLMGeminiConfig(api_key="test-api-key")
        sensor = VLMGemini(config=config)

        assert len(sensor.messages) == 0

        with patch("inputs.plugins.vlm_gemini.time.time", return_value=1000.0):
            await sensor.raw_to_text("First message")

        assert len(sensor.messages) == 1
        assert sensor.messages[0].message == "First message"

        with patch("inputs.plugins.vlm_gemini.time.time", return_value=2000.0):
            await sensor.raw_to_text("Second message")

        assert len(sensor.messages) == 2
        assert sensor.messages[1].message == "Second message"


@pytest.mark.asyncio
async def test_raw_to_text_with_none_input_no_append():
    """Test raw_to_text with None input does not append."""
    with (
        patch("inputs.plugins.vlm_gemini.IOProvider"),
        patch("inputs.plugins.vlm_gemini.VLMGeminiProvider"),
    ):
        config = VLMGeminiConfig(api_key="test-api-key")
        sensor = VLMGemini(config=config)

        await sensor.raw_to_text(None)
        assert len(sensor.messages) == 0


def test_formatted_latest_buffer_with_io_provider():
    """Test formatted_latest_buffer interacts with IO provider."""
    with (
        patch("inputs.plugins.vlm_gemini.IOProvider") as mock_io_provider_class,
        patch("inputs.plugins.vlm_gemini.VLMGeminiProvider"),
    ):
        mock_io_provider = MagicMock()
        mock_io_provider_class.return_value = mock_io_provider

        config = VLMGeminiConfig(api_key="test-api-key")
        sensor = VLMGemini(config=config)

        test_message = Message(timestamp=123.456, message="Test vision output")
        sensor.messages.append(test_message)

        result = sensor.formatted_latest_buffer()

        assert result is not None
        mock_io_provider.add_input.assert_called_once_with("VLMGemini", "Test vision output", 123.456)
        assert len(sensor.messages) == 0


def test_stop():
    """Test stop method deregisters callback and stops VLM."""
    with (
        patch("inputs.plugins.vlm_gemini.IOProvider"),
        patch("inputs.plugins.vlm_gemini.VLMGeminiProvider") as mock_provider,
    ):
        mock_vlm = MagicMock()
        mock_provider.return_value = mock_vlm

        config = VLMGeminiConfig(api_key="test-api-key")
        sensor = VLMGemini(config=config)

        sensor.stop()

        mock_vlm.deregister_message_callback.assert_called_once()
        mock_vlm.stop.assert_called_once()


def test_initialization_with_custom_stream_url():
    """Test initialization with custom stream_base_url."""
    with (
        patch("inputs.plugins.vlm_gemini.IOProvider"),
        patch("inputs.plugins.vlm_gemini.VLMGeminiProvider") as mock_provider,
    ):
        mock_vlm = MagicMock()
        mock_provider.return_value = mock_vlm

        custom_stream_url = "wss://custom.example.com/stream"
        config = VLMGeminiConfig(
            api_key="test-api-key",
            stream_base_url=custom_stream_url,
        )
        VLMGemini(config=config)

        mock_provider.assert_called_once()
        call_kwargs = mock_provider.call_args[1]
        assert call_kwargs["stream_url"] == custom_stream_url


def test_initialization_default_stream_url():
    """Test initialization generates default stream URL with API key."""
    with (
        patch("inputs.plugins.vlm_gemini.IOProvider"),
        patch("inputs.plugins.vlm_gemini.VLMGeminiProvider") as mock_provider,
    ):
        mock_vlm = MagicMock()
        mock_provider.return_value = mock_vlm

        config = VLMGeminiConfig(api_key="my-key-123")
        VLMGemini(config=config)

        mock_provider.assert_called_once()
        call_kwargs = mock_provider.call_args[1]
        expected_url = "wss://api.openmind.com/api/core/teleops/stream/video?api_key=my-key-123"
        assert call_kwargs["stream_url"] == expected_url
