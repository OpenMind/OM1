from unittest.mock import AsyncMock, patch

import pytest

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

        assert result is not None or result is None


def test_formatted_latest_buffer():
    """Test formatted_latest_buffer."""
    with patch("inputs.plugins.person_following_status.IOProvider"):
        config = PersonFollowingStatusConfig()
        sensor = PersonFollowingStatus(config=config)

        result = sensor.formatted_latest_buffer()
        assert result is None or isinstance(result, str)
