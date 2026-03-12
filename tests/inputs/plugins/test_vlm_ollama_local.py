import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import numpy as np
import pytest

from inputs.base import Message
from inputs.plugins.vlm_ollama_local import (
    VLM_Ollama_Local,
    VLM_Ollama_LocalConfig,
    check_webcam,
)


def test_check_webcam_found():
    """Test check_webcam returns True when camera is available."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True

    with patch(
        "inputs.plugins.vlm_ollama_local.cv2.VideoCapture", return_value=mock_cap
    ):
        result = check_webcam(0)
        assert result is True


def test_check_webcam_not_found():
    """Test check_webcam returns False when camera is unavailable."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False

    with patch(
        "inputs.plugins.vlm_ollama_local.cv2.VideoCapture", return_value=mock_cap
    ):
        result = check_webcam(0)
        assert result is False


def test_initialization_no_camera():
    """Test initialization without camera sets safe defaults."""
    with (
        patch("inputs.plugins.vlm_ollama_local.IOProvider"),
        patch("inputs.plugins.vlm_ollama_local.check_webcam", return_value=False),
    ):
        config = VLM_Ollama_LocalConfig()
        sensor = VLM_Ollama_Local(config=config)

        assert hasattr(sensor, "messages")
        assert sensor.descriptor_for_LLM == "Vision"
        assert sensor.have_cam is False
        assert sensor.cap is None


def test_initialization_with_camera():
    """Test initialization opens VideoCapture when camera is available."""
    mock_cap = MagicMock()
    mock_cap.get.return_value = 640

    with (
        patch("inputs.plugins.vlm_ollama_local.IOProvider"),
        patch("inputs.plugins.vlm_ollama_local.check_webcam", return_value=True),
        patch(
            "inputs.plugins.vlm_ollama_local.cv2.VideoCapture",
            return_value=mock_cap,
        ),
    ):
        config = VLM_Ollama_LocalConfig()
        sensor = VLM_Ollama_Local(config=config)

        assert sensor.have_cam is True
        assert sensor.cap is not None


def test_chat_url_built_correctly():
    """Test Ollama chat URL is correctly built from base_url config."""
    with (
        patch("inputs.plugins.vlm_ollama_local.IOProvider"),
        patch("inputs.plugins.vlm_ollama_local.check_webcam", return_value=False),
    ):
        config = VLM_Ollama_LocalConfig(base_url="http://localhost:11434")
        sensor = VLM_Ollama_Local(config=config)

        assert sensor._chat_url == "http://localhost:11434/api/chat"


def test_chat_url_strips_trailing_slash():
    """Test that trailing slash in base_url is stripped correctly."""
    with (
        patch("inputs.plugins.vlm_ollama_local.IOProvider"),
        patch("inputs.plugins.vlm_ollama_local.check_webcam", return_value=False),
    ):
        config = VLM_Ollama_LocalConfig(base_url="http://localhost:11434/")
        sensor = VLM_Ollama_Local(config=config)

        assert sensor._chat_url == "http://localhost:11434/api/chat"


@pytest.mark.asyncio
async def test_poll_returns_frame_when_camera_available():
    """Test _poll returns a numpy frame when camera read succeeds."""
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap = MagicMock()
    mock_cap.read.return_value = (True, fake_frame)

    with (
        patch("inputs.plugins.vlm_ollama_local.IOProvider"),
        patch("inputs.plugins.vlm_ollama_local.check_webcam", return_value=True),
        patch(
            "inputs.plugins.vlm_ollama_local.cv2.VideoCapture",
            return_value=mock_cap,
        ),
        patch("inputs.plugins.vlm_ollama_local.asyncio.sleep", new=AsyncMock()),
    ):
        config = VLM_Ollama_LocalConfig()
        sensor = VLM_Ollama_Local(config=config)

        result = await sensor._poll()
        assert result is not None
        assert isinstance(result, np.ndarray)


@pytest.mark.asyncio
async def test_poll_returns_none_when_no_camera():
    """Test _poll returns None when no camera is available."""
    with (
        patch("inputs.plugins.vlm_ollama_local.IOProvider"),
        patch("inputs.plugins.vlm_ollama_local.check_webcam", return_value=False),
        patch("inputs.plugins.vlm_ollama_local.asyncio.sleep", new=AsyncMock()),
    ):
        config = VLM_Ollama_LocalConfig()
        sensor = VLM_Ollama_Local(config=config)

        result = await sensor._poll()
        assert result is None


@pytest.mark.asyncio
async def test_poll_returns_none_on_failed_read():
    """Test _poll returns None when cap.read() returns ret=False."""
    mock_cap = MagicMock()
    mock_cap.read.return_value = (False, None)

    with (
        patch("inputs.plugins.vlm_ollama_local.IOProvider"),
        patch("inputs.plugins.vlm_ollama_local.check_webcam", return_value=True),
        patch(
            "inputs.plugins.vlm_ollama_local.cv2.VideoCapture",
            return_value=mock_cap,
        ),
        patch("inputs.plugins.vlm_ollama_local.asyncio.sleep", new=AsyncMock()),
    ):
        config = VLM_Ollama_LocalConfig()
        sensor = VLM_Ollama_Local(config=config)

        result = await sensor._poll()
        assert result is None


@pytest.mark.asyncio
async def test_poll_returns_none_on_none_frame():
    """Test _poll returns None when cap.read() returns True but frame is None."""
    mock_cap = MagicMock()
    mock_cap.read.return_value = (True, None)

    with (
        patch("inputs.plugins.vlm_ollama_local.IOProvider"),
        patch("inputs.plugins.vlm_ollama_local.check_webcam", return_value=True),
        patch(
            "inputs.plugins.vlm_ollama_local.cv2.VideoCapture",
            return_value=mock_cap,
        ),
        patch("inputs.plugins.vlm_ollama_local.asyncio.sleep", new=AsyncMock()),
    ):
        config = VLM_Ollama_LocalConfig()
        sensor = VLM_Ollama_Local(config=config)

        result = await sensor._poll()
        assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_returns_none_on_none_input():
    """Test _raw_to_text returns None when input is None."""
    with (
        patch("inputs.plugins.vlm_ollama_local.IOProvider"),
        patch("inputs.plugins.vlm_ollama_local.check_webcam", return_value=False),
    ):
        config = VLM_Ollama_LocalConfig()
        sensor = VLM_Ollama_Local(config=config)

        result = await sensor._raw_to_text(None)
        assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_returns_none_on_encode_failure():
    """Test _raw_to_text returns None when cv2.imencode fails."""
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with (
        patch("inputs.plugins.vlm_ollama_local.IOProvider"),
        patch("inputs.plugins.vlm_ollama_local.check_webcam", return_value=False),
        patch(
            "inputs.plugins.vlm_ollama_local.cv2.imencode",
            return_value=(False, None),
        ),
    ):
        config = VLM_Ollama_LocalConfig()
        sensor = VLM_Ollama_Local(config=config)

        result = await sensor._raw_to_text(fake_frame)
        assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_returns_message_on_success():
    """Test _raw_to_text returns a Message when Ollama responds successfully."""
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(
        return_value={"message": {"content": "I see a room with a chair."}}
    )
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.post.return_value = mock_response
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("inputs.plugins.vlm_ollama_local.IOProvider"),
        patch("inputs.plugins.vlm_ollama_local.check_webcam", return_value=False),
        patch(
            "inputs.plugins.vlm_ollama_local.aiohttp.ClientSession",
            return_value=mock_session,
        ),
    ):
        config = VLM_Ollama_LocalConfig()
        sensor = VLM_Ollama_Local(config=config)

        result = await sensor._raw_to_text(fake_frame)

        assert result is not None
        assert isinstance(result, Message)
        assert "chair" in result.message


@pytest.mark.asyncio
async def test_raw_to_text_returns_none_on_empty_response():
    """Test _raw_to_text returns None when Ollama returns empty content."""
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"message": {"content": ""}})
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.post.return_value = mock_response
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("inputs.plugins.vlm_ollama_local.IOProvider"),
        patch("inputs.plugins.vlm_ollama_local.check_webcam", return_value=False),
        patch(
            "inputs.plugins.vlm_ollama_local.aiohttp.ClientSession",
            return_value=mock_session,
        ),
    ):
        config = VLM_Ollama_LocalConfig()
        sensor = VLM_Ollama_Local(config=config)

        result = await sensor._raw_to_text(fake_frame)
        assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_returns_none_on_api_error():
    """Test _raw_to_text returns None when Ollama returns non-200 status."""
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    mock_response = MagicMock()
    mock_response.status = 500
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.post.return_value = mock_response
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("inputs.plugins.vlm_ollama_local.IOProvider"),
        patch("inputs.plugins.vlm_ollama_local.check_webcam", return_value=False),
        patch(
            "inputs.plugins.vlm_ollama_local.aiohttp.ClientSession",
            return_value=mock_session,
        ),
    ):
        config = VLM_Ollama_LocalConfig()
        sensor = VLM_Ollama_Local(config=config)

        result = await sensor._raw_to_text(fake_frame)
        assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_returns_none_on_connection_error():
    """Test _raw_to_text returns None when Ollama is unreachable."""
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    mock_session = MagicMock()
    mock_session.post.side_effect = aiohttp.ClientConnectorError(
        MagicMock(), MagicMock()
    )
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("inputs.plugins.vlm_ollama_local.IOProvider"),
        patch("inputs.plugins.vlm_ollama_local.check_webcam", return_value=False),
        patch(
            "inputs.plugins.vlm_ollama_local.aiohttp.ClientSession",
            return_value=mock_session,
        ),
    ):
        config = VLM_Ollama_LocalConfig()
        sensor = VLM_Ollama_Local(config=config)

        result = await sensor._raw_to_text(fake_frame)
        assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_returns_none_on_timeout():
    """Test _raw_to_text returns None when Ollama request times out."""
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    mock_session = MagicMock()
    mock_session.post.side_effect = asyncio.TimeoutError()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("inputs.plugins.vlm_ollama_local.IOProvider"),
        patch("inputs.plugins.vlm_ollama_local.check_webcam", return_value=False),
        patch(
            "inputs.plugins.vlm_ollama_local.aiohttp.ClientSession",
            return_value=mock_session,
        ),
    ):
        config = VLM_Ollama_LocalConfig()
        sensor = VLM_Ollama_Local(config=config)

        result = await sensor._raw_to_text(fake_frame)
        assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_returns_none_on_unexpected_exception():
    """Test _raw_to_text returns None on unexpected exceptions."""
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    mock_session = MagicMock()
    mock_session.post.side_effect = RuntimeError("unexpected error")
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("inputs.plugins.vlm_ollama_local.IOProvider"),
        patch("inputs.plugins.vlm_ollama_local.check_webcam", return_value=False),
        patch(
            "inputs.plugins.vlm_ollama_local.aiohttp.ClientSession",
            return_value=mock_session,
        ),
    ):
        config = VLM_Ollama_LocalConfig()
        sensor = VLM_Ollama_Local(config=config)

        result = await sensor._raw_to_text(fake_frame)
        assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_appends_to_buffer():
    """Test raw_to_text appends message to buffer on success."""
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    fake_message = Message(timestamp=123.456, message="I see a table.")

    with (
        patch("inputs.plugins.vlm_ollama_local.IOProvider"),
        patch("inputs.plugins.vlm_ollama_local.check_webcam", return_value=False),
        patch.object(
            VLM_Ollama_Local,
            "_raw_to_text",
            new=AsyncMock(return_value=fake_message),
        ),
    ):
        config = VLM_Ollama_LocalConfig()
        sensor = VLM_Ollama_Local(config=config)

        await sensor.raw_to_text(fake_frame)
        assert len(sensor.messages) == 1
        assert sensor.messages[0].message == "I see a table."


@pytest.mark.asyncio
async def test_raw_to_text_does_not_append_on_none():
    """Test raw_to_text does not append when _raw_to_text returns None."""
    with (
        patch("inputs.plugins.vlm_ollama_local.IOProvider"),
        patch("inputs.plugins.vlm_ollama_local.check_webcam", return_value=False),
        patch.object(
            VLM_Ollama_Local,
            "_raw_to_text",
            new=AsyncMock(return_value=None),
        ),
    ):
        config = VLM_Ollama_LocalConfig()
        sensor = VLM_Ollama_Local(config=config)

        await sensor.raw_to_text(None)
        assert len(sensor.messages) == 0


def test_formatted_latest_buffer_empty():
    """Test formatted_latest_buffer returns None when buffer is empty."""
    with (
        patch("inputs.plugins.vlm_ollama_local.IOProvider"),
        patch("inputs.plugins.vlm_ollama_local.check_webcam", return_value=False),
    ):
        config = VLM_Ollama_LocalConfig()
        sensor = VLM_Ollama_Local(config=config)

        result = sensor.formatted_latest_buffer()
        assert result is None


def test_formatted_latest_buffer_returns_formatted_string():
    """Test formatted_latest_buffer returns correct format and clears buffer."""
    with (
        patch("inputs.plugins.vlm_ollama_local.IOProvider"),
        patch("inputs.plugins.vlm_ollama_local.check_webcam", return_value=False),
    ):
        config = VLM_Ollama_LocalConfig()
        sensor = VLM_Ollama_Local(config=config)

        test_message = Message(
            timestamp=123.456, message="I see a person sitting on a chair."
        )
        sensor.messages.append(test_message)

        result = sensor.formatted_latest_buffer()

        assert result is not None
        assert isinstance(result, str)
        assert "INPUT:" in result
        assert "Vision" in result
        assert "I see a person sitting on a chair." in result
        assert "// START" in result
        assert "// END" in result
        assert len(sensor.messages) == 0
