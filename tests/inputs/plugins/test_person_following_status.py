"""Tests for PersonFollowingStatus input plugin."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from inputs.plugins.person_following_status import (
    PersonFollowingStatus,
    PersonFollowingStatusConfig,
)


@pytest.fixture
def config():
    """Create a test configuration."""
    return PersonFollowingStatusConfig(
        person_follow_base_url="http://localhost:8080",
        poll_interval=0.1,
        enroll_retry_interval=1.0,
    )


@pytest.fixture
def person_following_status(config):
    """Create a PersonFollowingStatus instance."""
    return PersonFollowingStatus(config)


@pytest.mark.asyncio
async def test_poll_successful_tracking(person_following_status):
    """Test successful polling when person is being tracked."""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(
        return_value={
            "is_tracked": True,
            "status": "TRACKING_ACTIVE",
            "x": 0.5,
            "z": 1.5,
            "target_track_id": "123",
        }
    )
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    mock_session = AsyncMock()
    mock_session.get = AsyncMock(return_value=mock_response)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    with patch("aiohttp.ClientSession", return_value=mock_session):
        result = await person_following_status._poll()
        assert result is not None
        assert "TRACKING" in result or "TRACKING STARTED" in result


@pytest.mark.asyncio
async def test_poll_not_tracked(person_following_status):
    """Test polling when person is not being tracked."""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(
        return_value={
            "is_tracked": False,
            "status": "INACTIVE",
            "x": 0.0,
            "z": 0.0,
            "target_track_id": None,
        }
    )

    with patch("aiohttp.ClientSession") as mock_session_class:
        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        result = await person_following_status._poll()
        # Should return None or a waiting message
        assert result is None or "WAITING" in result or "SEARCHING" in result


@pytest.mark.asyncio
async def test_poll_non_200_status(person_following_status):
    """Test polling when API returns non-200 status."""
    mock_response = AsyncMock()
    mock_response.status = 500

    with patch("aiohttp.ClientSession") as mock_session_class:
        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        result = await person_following_status._poll()
        assert result is None


@pytest.mark.asyncio
async def test_poll_json_decode_error(person_following_status):
    """Test polling when JSON decode fails."""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(side_effect=Exception("Invalid JSON"))

    with patch("aiohttp.ClientSession") as mock_session_class:
        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        result = await person_following_status._poll()
        assert result is None


@pytest.mark.asyncio
async def test_poll_client_error(person_following_status):
    """Test polling when client error occurs."""
    with patch("aiohttp.ClientSession") as mock_session_class:
        mock_session = AsyncMock()
        mock_session.get = AsyncMock(side_effect=aiohttp.ClientError("Connection error"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        result = await person_following_status._poll()
        assert result is None


@pytest.mark.asyncio
async def test_try_enroll_success(person_following_status):
    """Test successful enrollment attempt."""
    mock_response = AsyncMock()
    mock_response.status = 200

    with patch("aiohttp.ClientSession") as mock_session_class:
        mock_session = AsyncMock()
        mock_session.post = AsyncMock(return_value=mock_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        await person_following_status._try_enroll(mock_session)
        mock_session.post.assert_called_once()


@pytest.mark.asyncio
async def test_try_enroll_failure(person_following_status):
    """Test enrollment attempt when it fails."""
    mock_response = AsyncMock()
    mock_response.status = 500

    with patch("aiohttp.ClientSession") as mock_session_class:
        mock_session = AsyncMock()
        mock_session.post = AsyncMock(return_value=mock_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        await person_following_status._try_enroll(mock_session)
        mock_session.post.assert_called_once()


@pytest.mark.asyncio
async def test_try_enroll_exception(person_following_status):
    """Test enrollment attempt when exception occurs."""
    mock_session = AsyncMock()
    mock_session.post = AsyncMock(side_effect=Exception("Network error"))

    await person_following_status._try_enroll(mock_session)
    # Should not raise, just log


def test_format_status_tracking_started(person_following_status):
    """Test formatting status when tracking starts."""
    person_following_status._previous_is_tracked = False
    data = {
        "is_tracked": True,
        "status": "TRACKING_ACTIVE",
        "x": 0.5,
        "z": 1.5,
        "target_track_id": "123",
    }

    result = person_following_status._format_status(data)
    assert result is not None
    assert "TRACKING STARTED" in result
    assert person_following_status._previous_is_tracked is True


def test_format_status_tracking_lost(person_following_status):
    """Test formatting status when tracking is lost."""
    person_following_status._previous_is_tracked = True
    data = {
        "is_tracked": False,
        "status": "SEARCHING",
        "x": 0.0,
        "z": 0.0,
        "target_track_id": "123",
    }

    result = person_following_status._format_status(data)
    # Should return None initially when tracking is lost
    assert result is None or "SEARCHING" in result or "WAITING" in result


@pytest.mark.asyncio
async def test_raw_to_text_with_input(person_following_status):
    """Test raw_to_text with valid input."""
    message = await person_following_status._raw_to_text("TRACKING: Following person")
    assert message is not None
    assert message.message == "TRACKING: Following person"


@pytest.mark.asyncio
async def test_raw_to_text_with_none(person_following_status):
    """Test raw_to_text with None input."""
    message = await person_following_status._raw_to_text(None)
    assert message is None


def test_formatted_latest_buffer_empty(person_following_status):
    """Test formatted_latest_buffer when no messages."""
    result = person_following_status.formatted_latest_buffer()
    assert result is None


def test_formatted_latest_buffer_with_messages(person_following_status):
    """Test formatted_latest_buffer with messages."""
    from inputs.base import Message
    import time

    message = Message(timestamp=time.time(), message="TRACKING: Following person")
    person_following_status.messages.append(message)

    result = person_following_status.formatted_latest_buffer()
    assert result is not None
    assert "TRACKING: Following person" in result
    assert len(person_following_status.messages) == 0  # Should be cleared
