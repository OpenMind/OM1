from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inputs.base import Message
from inputs.plugins.unitree_go2_odom_zenoh import (
    UnitreeGo2OdomZenoh,
    UnitreeGo2OdomZenohConfig,
)
from providers.odom_provider_base import RobotState


@pytest.fixture
def patches():
    with (
        patch("inputs.plugins.unitree_go2_odom_zenoh.IOProvider"),
        patch("inputs.plugins.unitree_go2_odom_zenoh.UnitreeGo2OdomZenohProvider") as mock_provider_class,
    ):
        instance = MagicMock()
        instance.position = None
        mock_provider_class.return_value = instance
        yield {"provider_class": mock_provider_class, "provider": instance}


def test_initialization(patches):
    config = UnitreeGo2OdomZenohConfig()
    sensor = UnitreeGo2OdomZenoh(config=config)
    assert sensor.messages == []
    assert "location" in sensor.descriptor_for_LLM.lower()
    patches["provider_class"].assert_called_once_with(api_key=None, topic="utlidar/robot_pose", use_sim=False)


def test_initialization_custom_config(patches):
    config = UnitreeGo2OdomZenohConfig(api_key="k", topic="odom", use_sim=True)
    UnitreeGo2OdomZenoh(config=config)
    patches["provider_class"].assert_called_once_with(api_key="k", topic="odom", use_sim=True)


@pytest.mark.asyncio
async def test_poll_returns_position(patches):
    patches["provider"].position = {"moving": False, "body_attitude": RobotState.STANDING}
    config = UnitreeGo2OdomZenohConfig()
    sensor = UnitreeGo2OdomZenoh(config=config)
    with patch("inputs.plugins.unitree_go2_odom_zenoh.asyncio.sleep", new=AsyncMock()):
        result = await sensor._poll()
    assert result == {"moving": False, "body_attitude": RobotState.STANDING}


@pytest.mark.asyncio
async def test_poll_returns_none_when_empty(patches):
    from queue import Empty as QueueEmpty

    type(patches["provider"]).position = property(lambda _: (_ for _ in ()).throw(QueueEmpty()))
    config = UnitreeGo2OdomZenohConfig()
    sensor = UnitreeGo2OdomZenoh(config=config)
    with patch("inputs.plugins.unitree_go2_odom_zenoh.asyncio.sleep", new=AsyncMock()):
        result = await sensor._poll()
    assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_sitting(patches):
    config = UnitreeGo2OdomZenohConfig()
    sensor = UnitreeGo2OdomZenoh(config=config)
    msg = await sensor._raw_to_text({"moving": False, "body_attitude": RobotState.SITTING})
    assert "sitting" in msg.message.lower()


@pytest.mark.asyncio
async def test_raw_to_text_moving(patches):
    config = UnitreeGo2OdomZenohConfig()
    sensor = UnitreeGo2OdomZenoh(config=config)
    msg = await sensor._raw_to_text({"moving": True, "body_attitude": RobotState.STANDING})
    assert "moving" in msg.message.lower()


@pytest.mark.asyncio
async def test_raw_to_text_standing_still(patches):
    config = UnitreeGo2OdomZenohConfig()
    sensor = UnitreeGo2OdomZenoh(config=config)
    msg = await sensor._raw_to_text({"moving": False, "body_attitude": RobotState.STANDING})
    assert "standing still" in msg.message.lower()


@pytest.mark.asyncio
async def test_raw_to_text_none_input_returns_none(patches):
    config = UnitreeGo2OdomZenohConfig()
    sensor = UnitreeGo2OdomZenoh(config=config)
    msg = await sensor._raw_to_text(None)
    assert msg is None


@pytest.mark.asyncio
async def test_raw_to_text_replaces_existing_message(patches):
    config = UnitreeGo2OdomZenohConfig()
    sensor = UnitreeGo2OdomZenoh(config=config)

    await sensor.raw_to_text({"moving": False, "body_attitude": RobotState.STANDING})
    assert len(sensor.messages) == 1
    first = sensor.messages[0]

    await sensor.raw_to_text({"moving": True, "body_attitude": RobotState.STANDING})
    assert len(sensor.messages) == 1
    assert sensor.messages[0] is not first


@pytest.mark.asyncio
async def test_raw_to_text_skips_none(patches):
    config = UnitreeGo2OdomZenohConfig()
    sensor = UnitreeGo2OdomZenoh(config=config)
    await sensor.raw_to_text(None)
    assert sensor.messages == []


def test_formatted_latest_buffer_empty(patches):
    config = UnitreeGo2OdomZenohConfig()
    sensor = UnitreeGo2OdomZenoh(config=config)
    assert sensor.formatted_latest_buffer() is None


def test_formatted_latest_buffer_with_message(patches):
    config = UnitreeGo2OdomZenohConfig()
    sensor = UnitreeGo2OdomZenoh(config=config)
    sensor.messages.append(Message(timestamp=1.0, message="standing still"))
    result = sensor.formatted_latest_buffer()
    assert result is not None
    assert "standing still" in result
    assert sensor.messages == []
