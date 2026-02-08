from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inputs.base import Message, SensorConfig
from inputs.plugins.x402_input import X402Input


@patch("inputs.plugins.x402_input.Flask")
@patch("inputs.plugins.x402_input.IOProvider")
def test_initialization(mock_io_provider, mock_flask):
    mock_app = MagicMock()
    mock_flask.return_value = mock_app
    mock_app.route = MagicMock(return_value=lambda f: f)

    with patch("inputs.plugins.x402_input.threading.Thread") as mock_thread:
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance

        config = SensorConfig()
        sensor = X402Input(config=config)

        assert sensor.messages == []
        assert sensor.descriptor_for_LLM == "X402 Input"
        mock_thread_instance.start.assert_called_once()


@patch("inputs.plugins.x402_input.Flask")
@patch("inputs.plugins.x402_input.IOProvider")
def test_initialization_custom_config(mock_io_provider, mock_flask):
    mock_app = MagicMock()
    mock_flask.return_value = mock_app
    mock_app.route = MagicMock(return_value=lambda f: f)

    with patch("inputs.plugins.x402_input.threading.Thread") as mock_thread:
        mock_thread.return_value = MagicMock()

        config = SensorConfig(**{"input_name": "Custom Input", "fee": "0.05"})
        sensor = X402Input(config=config)

        assert sensor.descriptor_for_LLM == "Custom Input"


@pytest.mark.asyncio
@patch("inputs.plugins.x402_input.Flask")
@patch("inputs.plugins.x402_input.IOProvider")
async def test_poll_with_message(mock_io_provider, mock_flask):
    mock_app = MagicMock()
    mock_flask.return_value = mock_app
    mock_app.route = MagicMock(return_value=lambda f: f)

    with patch("inputs.plugins.x402_input.threading.Thread") as mock_thread:
        mock_thread.return_value = MagicMock()

        config = SensorConfig()
        sensor = X402Input(config=config)

        test_message = Message(timestamp=1234.0, message="Hello")
        sensor.message_buffer.put(test_message)

        with patch("inputs.plugins.x402_input.asyncio.sleep", new=AsyncMock()):
            result = await sensor._poll()

        assert result is not None
        assert result.message == "Hello"
        assert result.timestamp == 1234.0


@pytest.mark.asyncio
@patch("inputs.plugins.x402_input.Flask")
@patch("inputs.plugins.x402_input.IOProvider")
async def test_poll_empty_buffer(mock_io_provider, mock_flask):
    mock_app = MagicMock()
    mock_flask.return_value = mock_app
    mock_app.route = MagicMock(return_value=lambda f: f)

    with patch("inputs.plugins.x402_input.threading.Thread") as mock_thread:
        mock_thread.return_value = MagicMock()

        config = SensorConfig()
        sensor = X402Input(config=config)

        with patch("inputs.plugins.x402_input.asyncio.sleep", new=AsyncMock()):
            result = await sensor._poll()

        assert result is None


@pytest.mark.asyncio
@patch("inputs.plugins.x402_input.Flask")
@patch("inputs.plugins.x402_input.IOProvider")
async def test_raw_to_text_with_message(mock_io_provider, mock_flask):
    mock_app = MagicMock()
    mock_flask.return_value = mock_app
    mock_app.route = MagicMock(return_value=lambda f: f)

    with patch("inputs.plugins.x402_input.threading.Thread") as mock_thread:
        mock_thread.return_value = MagicMock()

        config = SensorConfig()
        sensor = X402Input(config=config)

        msg = Message(timestamp=1234.0, message="Test message")
        result = await sensor._raw_to_text(msg)

        assert result is not None
        assert result.message == "Test message"


@pytest.mark.asyncio
@patch("inputs.plugins.x402_input.Flask")
@patch("inputs.plugins.x402_input.IOProvider")
async def test_raw_to_text_with_none(mock_io_provider, mock_flask):
    mock_app = MagicMock()
    mock_flask.return_value = mock_app
    mock_app.route = MagicMock(return_value=lambda f: f)

    with patch("inputs.plugins.x402_input.threading.Thread") as mock_thread:
        mock_thread.return_value = MagicMock()

        config = SensorConfig()
        sensor = X402Input(config=config)

        result = await sensor._raw_to_text(None)
        assert result is None


@pytest.mark.asyncio
@patch("inputs.plugins.x402_input.Flask")
@patch("inputs.plugins.x402_input.IOProvider")
async def test_raw_to_text_appends_to_messages(mock_io_provider, mock_flask):
    mock_app = MagicMock()
    mock_flask.return_value = mock_app
    mock_app.route = MagicMock(return_value=lambda f: f)

    with patch("inputs.plugins.x402_input.threading.Thread") as mock_thread:
        mock_thread.return_value = MagicMock()

        config = SensorConfig()
        sensor = X402Input(config=config)

        msg = Message(timestamp=1234.0, message="Buffered message")
        await sensor.raw_to_text(msg)

        assert len(sensor.messages) == 1
        assert sensor.messages[0].message == "Buffered message"


@pytest.mark.asyncio
@patch("inputs.plugins.x402_input.Flask")
@patch("inputs.plugins.x402_input.IOProvider")
async def test_raw_to_text_skips_none(mock_io_provider, mock_flask):
    mock_app = MagicMock()
    mock_flask.return_value = mock_app
    mock_app.route = MagicMock(return_value=lambda f: f)

    with patch("inputs.plugins.x402_input.threading.Thread") as mock_thread:
        mock_thread.return_value = MagicMock()

        config = SensorConfig()
        sensor = X402Input(config=config)

        await sensor.raw_to_text(None)
        assert len(sensor.messages) == 0


@patch("inputs.plugins.x402_input.Flask")
@patch("inputs.plugins.x402_input.IOProvider")
def test_formatted_latest_buffer_with_messages(mock_io_provider, mock_flask):
    mock_app = MagicMock()
    mock_flask.return_value = mock_app
    mock_app.route = MagicMock(return_value=lambda f: f)

    with patch("inputs.plugins.x402_input.threading.Thread") as mock_thread:
        mock_thread.return_value = MagicMock()

        config = SensorConfig()
        sensor = X402Input(config=config)
        sensor.io_provider = MagicMock()

        sensor.messages = [
            Message(timestamp=1000.0, message="Paid message content"),
        ]

        result = sensor.formatted_latest_buffer()

        assert result is not None
        assert "X402 Input" in result
        assert "Paid message content" in result
        sensor.io_provider.add_input.assert_called_once()
        assert len(sensor.messages) == 0


@patch("inputs.plugins.x402_input.Flask")
@patch("inputs.plugins.x402_input.IOProvider")
def test_formatted_latest_buffer_empty(mock_io_provider, mock_flask):
    mock_app = MagicMock()
    mock_flask.return_value = mock_app
    mock_app.route = MagicMock(return_value=lambda f: f)

    with patch("inputs.plugins.x402_input.threading.Thread") as mock_thread:
        mock_thread.return_value = MagicMock()

        config = SensorConfig()
        sensor = X402Input(config=config)

        result = sensor.formatted_latest_buffer()
        assert result is None
