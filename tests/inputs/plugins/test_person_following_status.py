import time
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from inputs.base import Message
from inputs.plugins.person_following_status import (
    PersonFollowingStatus,
    PersonFollowingStatusConfig,
)


def make_sensor():
    with patch("inputs.plugins.person_following_status.IOProvider"):
        config = PersonFollowingStatusConfig()
        return PersonFollowingStatus(config=config)


def make_mock_session(response_data, response_status=200, enroll_status=200):
    """Helper untuk membuat mock session dengan async context manager yang benar."""
    mock_response = MagicMock()
    mock_response.status = response_status
    mock_response.json = AsyncMock(return_value=response_data)

    mock_session = MagicMock()
    mock_session.closed = False
    mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
    mock_session.get.return_value.__aexit__ = AsyncMock(return_value=False)

    mock_enroll_response = MagicMock()
    mock_enroll_response.status = enroll_status
    mock_session.post.return_value.__aenter__ = AsyncMock(
        return_value=mock_enroll_response
    )
    mock_session.post.return_value.__aexit__ = AsyncMock(return_value=False)

    return mock_session


def test_initialization():
    """Test basic initialization."""
    sensor = make_sensor()
    assert hasattr(sensor, "messages")


@pytest.mark.asyncio
async def test_poll():
    """Test _poll method."""
    sensor = make_sensor()
    with patch("inputs.plugins.person_following_status.asyncio.sleep", new=AsyncMock()):
        result = await sensor._poll()
        assert result is None


def test_formatted_latest_buffer():
    """Test formatted_latest_buffer."""
    sensor = make_sensor()

    assert sensor.formatted_latest_buffer() is None

    sensor.messages.append(
        Message(
            timestamp=123.456,
            message="TRACKING STARTED: Person detected and now following. Distance: 2.5m ahead, 0.3m to the side.",
        )
    )
    result = sensor.formatted_latest_buffer()
    assert isinstance(result, str)
    assert "INPUT:" in result
    assert "Person Following Status" in result
    assert "TRACKING STARTED" in result
    assert "// START" in result
    assert "// END" in result
    assert len(sensor.messages) == 0


def test_session_is_none_on_init():
    """Test session belum dibuat saat inisialisasi."""
    sensor = make_sensor()
    assert sensor._session is None


@pytest.mark.asyncio
async def test_session_reuse():
    """Test session yang sama dipakai ulang, tidak dibuat baru setiap poll."""
    sensor = make_sensor()

    sensor._session = aiohttp.ClientSession()
    existing_session = sensor._session

    if sensor._session is None or sensor._session.closed:
        sensor._session = aiohttp.ClientSession()

    assert sensor._session is existing_session

    await existing_session.close()


@pytest.mark.asyncio
async def test_session_recreated_if_closed():
    """Test session baru dibuat jika session sebelumnya sudah closed."""
    sensor = make_sensor()

    sensor._session = aiohttp.ClientSession()
    await sensor._session.close()
    assert sensor._session.closed

    if sensor._session is None or sensor._session.closed:
        sensor._session = aiohttp.ClientSession()

    assert not sensor._session.closed
    await sensor._session.close()


def test_tracking_throttle():
    """Test TRACKING message hanya dikirim setiap 10 detik, bukan setiap poll."""
    sensor = make_sensor()
    sensor._previous_is_tracked = True

    data = {
        "is_tracked": True,
        "x": 0.1,
        "z": 2.0,
        "status": "TRACKING_ACTIVE",
        "target_track_id": 1,
    }

    sensor._last_tracking_report = time.time()
    assert sensor._format_status(data) is None

    sensor._last_tracking_report = time.time() - 11.0
    result = sensor._format_status(data)
    assert result is not None
    assert "TRACKING" in result


def test_tracking_started_bypasses_throttle():
    """Test event TRACKING STARTED selalu dikirim meski throttle belum melewati 10 detik."""
    sensor = make_sensor()
    sensor._previous_is_tracked = False
    sensor._last_tracking_report = time.time()

    data = {
        "is_tracked": True,
        "x": 0.1,
        "z": 2.0,
        "status": "TRACKING_ACTIVE",
        "target_track_id": 1,
    }

    result = sensor._format_status(data)
    assert result is not None
    assert "TRACKING STARTED" in result


@pytest.mark.asyncio
async def test_poll_with_successful_response():
    """Test _poll saat HTTP response berhasil dan data tracking tersedia."""
    sensor = make_sensor()

    mock_session = make_mock_session(
        {
            "is_tracked": True,
            "status": "TRACKING_ACTIVE",
            "target_track_id": 1,
            "x": 0.1,
            "z": 2.0,
        }
    )

    with patch("inputs.plugins.person_following_status.asyncio.sleep", new=AsyncMock()):
        sensor._session = mock_session
        await sensor._poll()
        assert sensor._has_ever_tracked is True


@pytest.mark.asyncio
async def test_poll_inactive_triggers_enroll():
    """Test _poll memanggil enroll saat status INACTIVE."""
    sensor = make_sensor()
    sensor._last_enroll_attempt = 0.0

    mock_session = make_mock_session(
        {
            "is_tracked": False,
            "status": "INACTIVE",
            "target_track_id": None,
            "x": 0.0,
            "z": 0.0,
        }
    )

    with patch("inputs.plugins.person_following_status.asyncio.sleep", new=AsyncMock()):
        sensor._session = mock_session
        await sensor._poll()
        mock_session.post.assert_called_once()


@pytest.mark.asyncio
async def test_poll_exception_handling():
    """Test _poll menangani exception dengan benar."""
    sensor = make_sensor()

    mock_session = MagicMock()
    mock_session.closed = False
    mock_session.get.side_effect = Exception("unexpected error")

    with patch("inputs.plugins.person_following_status.asyncio.sleep", new=AsyncMock()):
        sensor._session = mock_session
        result = await sensor._poll()
        assert result is None


@pytest.mark.asyncio
async def test_try_enroll_success():
    """Test _try_enroll saat server mengembalikan status 200."""
    sensor = make_sensor()

    mock_response = MagicMock()
    mock_response.status = 200

    mock_session = MagicMock()
    mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
    mock_session.post.return_value.__aexit__ = AsyncMock(return_value=False)

    await sensor._try_enroll(mock_session)
    mock_session.post.assert_called_once()


@pytest.mark.asyncio
async def test_try_enroll_failure_status():
    """Test _try_enroll saat server mengembalikan status non-200."""
    sensor = make_sensor()

    mock_response = MagicMock()
    mock_response.status = 500

    mock_session = MagicMock()
    mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
    mock_session.post.return_value.__aexit__ = AsyncMock(return_value=False)

    await sensor._try_enroll(mock_session)
    mock_session.post.assert_called_once()


@pytest.mark.asyncio
async def test_try_enroll_exception():
    """Test _try_enroll menangani exception dengan benar."""
    sensor = make_sensor()

    mock_session = MagicMock()
    mock_session.post.side_effect = Exception("connection error")

    await sensor._try_enroll(mock_session)


def test_format_status_tracking_lost_searching():
    """Test _format_status mengembalikan SEARCHING saat tracking hilang dan status SEARCHING."""
    sensor = make_sensor()
    sensor._previous_is_tracked = True

    data = {
        "is_tracked": False,
        "x": 0.0,
        "z": 0.0,
        "status": "SEARCHING",
        "target_track_id": 1,
    }

    sensor._format_status(data)

    sensor._lost_tracking_time = time.time() - 3.0
    result = sensor._format_status(data)
    assert result is not None
    assert "SEARCHING" in result


def test_format_status_tracking_lost_waiting():
    """Test _format_status mengembalikan WAITING saat status INACTIVE."""
    sensor = make_sensor()
    sensor._previous_is_tracked = True

    data = {
        "is_tracked": False,
        "x": 0.0,
        "z": 0.0,
        "status": "INACTIVE",
        "target_track_id": None,
    }

    sensor._format_status(data)
    sensor._lost_tracking_time = time.time() - 3.0
    result = sensor._format_status(data)
    assert result is not None
    assert "WAITING" in result


@pytest.mark.asyncio
async def test_raw_to_text_with_input():
    """Test raw_to_text menambahkan message ke buffer."""
    sensor = make_sensor()

    await sensor.raw_to_text("TRACKING STARTED: test message")
    assert len(sensor.messages) == 1
    assert "TRACKING STARTED" in sensor.messages[0].message


@pytest.mark.asyncio
async def test_raw_to_text_with_none():
    """Test raw_to_text tidak menambahkan apapun saat input None."""
    sensor = make_sensor()

    await sensor.raw_to_text(None)
    assert len(sensor.messages) == 0


@pytest.mark.asyncio
async def test_poll_non_200_response():
    """Test _poll mengembalikan None saat response status bukan 200."""
    sensor = make_sensor()

    mock_session = make_mock_session({}, response_status=503)

    with patch("inputs.plugins.person_following_status.asyncio.sleep", new=AsyncMock()):
        sensor._session = mock_session
        result = await sensor._poll()
        assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_returns_none_for_none():
    """Test _raw_to_text mengembalikan None saat input None."""
    sensor = make_sensor()
    result = await sensor._raw_to_text(None)
    assert result is None
