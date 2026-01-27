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
