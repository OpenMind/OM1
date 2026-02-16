from unittest.mock import AsyncMock, create_autospec

import pytest

from src.inputs.base import SensorConfig
from src.inputs.base.loop import FuserInput


@pytest.mark.asyncio
async def test_listen_loop_yields_values_from_poll():
    """Test that _listen_loop yields values returned by _poll."""
    config = create_autospec(SensorConfig)
    instance = FuserInput(config)

    mock_poll = AsyncMock(side_effect=[10, 20, 30])
    instance._poll = mock_poll

    collected = []
    async for value in instance._listen_loop():
        collected.append(value)
        if len(collected) == 3:
            break

    assert collected == [10, 20, 30]
    assert mock_poll.await_count == 3


@pytest.mark.asyncio
async def test_poll_raises_not_implemented():
    """Test that _poll raises NotImplementedError by default."""
    config = create_autospec(SensorConfig)
    instance = FuserInput(config)

    with pytest.raises(NotImplementedError):
        await instance._poll()
