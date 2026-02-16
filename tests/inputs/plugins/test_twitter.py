from queue import Queue
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inputs.plugins.twitter import TwitterInput, TwitterSensorConfig


def test_initialization():
    """Test basic initialization."""
    config = TwitterSensorConfig()
    sensor = TwitterInput(config=config)

    assert sensor.buffer == []
    assert isinstance(sensor.message_buffer, Queue)
    assert sensor.query == "What's new in AI and technology?"


def test_initialization_with_custom_query():
    """Test initialization with custom query."""
    config = TwitterSensorConfig(query="Custom search query")
    sensor = TwitterInput(config=config)

    assert sensor.query == "Custom search query"


@pytest.mark.asyncio
async def test_init_session():
    """Test session initialization."""
    config = TwitterSensorConfig()
    sensor = TwitterInput(config=config)

    assert sensor.session is None
    await sensor._init_session()
    assert sensor.session is not None


@pytest.mark.asyncio
async def test_query_context_success():
    """Test successful context query."""
    config = TwitterSensorConfig()
    sensor = TwitterInput(config=config)

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(
        return_value={
            "results": [
                {"content": {"text": "Document 1"}},
                {"content": {"text": "Document 2"}},
            ]
        }
    )

    with patch("aiohttp.ClientSession.post", return_value=mock_response) as mock_post:
        mock_post.return_value.__aenter__.return_value = mock_response
        await sensor._query_context("test query")

    assert sensor.context is not None
    assert "Document 1" in sensor.context
    assert "Document 2" in sensor.context


@pytest.mark.asyncio
async def test_query_context_failure():
    """Test failed context query."""
    config = TwitterSensorConfig()
    sensor = TwitterInput(config=config)

    mock_response = MagicMock()
    mock_response.status = 500
    mock_response.text = AsyncMock(return_value="Server error")

    with patch("aiohttp.ClientSession.post", return_value=mock_response) as mock_post:
        mock_post.return_value.__aenter__.return_value = mock_response
        await sensor._query_context("test query")

    # Context should not be set on failure
    assert sensor.context is None


@pytest.mark.asyncio
async def test_poll():
    """Test _poll method."""
    config = TwitterSensorConfig()
    sensor = TwitterInput(config=config)
    sensor.message_buffer.put("Test message")

    with patch("inputs.plugins.twitter.asyncio.sleep", new=AsyncMock()):
        result = await sensor._poll()

    assert result == "Test message"


@pytest.mark.asyncio
async def test_poll_empty_buffer():
    """Test _poll with empty buffer."""
    config = TwitterSensorConfig()
    sensor = TwitterInput(config=config)

    with patch("inputs.plugins.twitter.asyncio.sleep", new=AsyncMock()):
        result = await sensor._poll()

    assert result is None


@pytest.mark.asyncio
async def test_context_manager():
    """Test async context manager."""
    config = TwitterSensorConfig()

    async with TwitterInput(config=config) as sensor:
        assert sensor.session is not None


@pytest.mark.asyncio
async def test_raw_to_text_with_input():
    """Test raw_to_text adds input to message_buffer and buffer."""
    config = TwitterSensorConfig()
    sensor = TwitterInput(config=config)

    result = await sensor.raw_to_text("Test tweet content")
    assert result == "Test tweet content"
    assert "Test tweet content" in sensor.buffer


@pytest.mark.asyncio
async def test_raw_to_text_empty_returns_empty():
    """Test raw_to_text with no input and empty buffer returns empty string."""
    config = TwitterSensorConfig()
    sensor = TwitterInput(config=config)

    result = await sensor.raw_to_text()
    assert result == ""


@pytest.mark.asyncio
async def test_start():
    """Test start queries context and adds query to buffer."""
    config = TwitterSensorConfig(query="AI news")
    sensor = TwitterInput(config=config)

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"results": []})

    with patch("aiohttp.ClientSession.post", return_value=mock_response) as mock_post:
        mock_post.return_value.__aenter__.return_value = mock_response
        await sensor.start()

    assert sensor.message_buffer.qsize() >= 1


@pytest.mark.asyncio
async def test_initialize_with_query():
    """Test initialize_with_query adds query to buffer and queries context."""
    config = TwitterSensorConfig()
    sensor = TwitterInput(config=config)

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"results": []})

    with patch("aiohttp.ClientSession.post", return_value=mock_response) as mock_post:
        mock_post.return_value.__aenter__.return_value = mock_response
        await sensor.initialize_with_query("custom query")

    assert not sensor.message_buffer.empty()
    msg = sensor.message_buffer.get_nowait()
    assert msg == "custom query"


def test_formatted_latest_buffer_with_context():
    """Test formatted_latest_buffer returns context when available."""
    config = TwitterSensorConfig()
    sensor = TwitterInput(config=config)

    sensor.context = "Context about AI trends"

    result = sensor.formatted_latest_buffer()
    assert result is not None
    assert "Context about AI trends" in result
    assert "TwitterInput CONTEXT" in result


def test_formatted_latest_buffer_with_buffer_fallback():
    """Test formatted_latest_buffer falls back to buffer when no context."""
    config = TwitterSensorConfig()
    sensor = TwitterInput(config=config)

    sensor.buffer.append("Latest tweet")

    result = sensor.formatted_latest_buffer()
    assert result is not None
    assert "Latest tweet" in result


def test_formatted_latest_buffer_empty():
    """Test formatted_latest_buffer returns None when no content."""
    config = TwitterSensorConfig()
    sensor = TwitterInput(config=config)

    result = sensor.formatted_latest_buffer()
    assert result is None
