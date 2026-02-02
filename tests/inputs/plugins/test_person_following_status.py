from unittest.mock import AsyncMock, patch

import pytest

from inputs.base import Message
from inputs.plugins.person_following_status import (
    PersonFollowingStatus,
    PersonFollowingStatusConfig,
)


def test_initialization():
    """Test basic initialization."""
    with patch("inputs.plugins.person_following_status.IOProvider"):
        config = PersonFollowingStatusConfig()
        sensor = PersonFollowingStatus(config=config)

        assert hasattr(sensor, "messages")


@pytest.mark.asyncio
async def test_poll():
    """Test _poll method."""
    with patch("inputs.plugins.person_following_status.IOProvider"):
        config = PersonFollowingStatusConfig()
        sensor = PersonFollowingStatus(config=config)

        with patch(
            "inputs.plugins.person_following_status.asyncio.sleep", new=AsyncMock()
        ):
            result = await sensor._poll()
            assert result is None


def test_formatted_latest_buffer():
    """Test formatted_latest_buffer."""
    with patch("inputs.plugins.person_following_status.IOProvider"):
        config = PersonFollowingStatusConfig()
        sensor = PersonFollowingStatus(config=config)

        result = sensor.formatted_latest_buffer()
        assert result is None

        test_message = Message(
            timestamp=123.456,
            message="TRACKING STARTED: Person detected and now following. Distance: 2.5m ahead, 0.3m to the side.",
        )
        sensor.messages.append(test_message)

        result = sensor.formatted_latest_buffer()
        assert isinstance(result, str)
        assert "INPUT:" in result
        assert "Person Following Status" in result
        assert "TRACKING STARTED" in result
        assert "// START" in result
        assert "// END" in result
        assert len(sensor.messages) == 0
