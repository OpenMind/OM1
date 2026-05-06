from unittest.mock import MagicMock, patch

import pytest

from inputs.plugins.unitree_go2_battery_zenoh import (
    UnitreeGo2BatteryZenoh,
    UnitreeGo2BatteryZenohConfig,
)


@pytest.fixture
def patches():
    with (
        patch("inputs.plugins.unitree_go2_battery_zenoh.IOProvider"),
        patch(
            "inputs.plugins.unitree_go2_battery_zenoh.TeleopsStatusProvider"
        ) as mock_status,
        patch(
            "inputs.plugins.unitree_go2_battery_zenoh.open_zenoh_session"
        ) as mock_open_session,
    ):
        mock_session = MagicMock()
        mock_open_session.return_value = mock_session
        yield {
            "status": mock_status,
            "session": mock_session,
            "open_session": mock_open_session,
        }


def test_initialization(patches):
    config = UnitreeGo2BatteryZenohConfig()
    sensor = UnitreeGo2BatteryZenoh(config=config)

    assert sensor.battery_percentage == 0.0
    assert sensor.battery_voltage == 0.0
    assert sensor.battery_amperes == 0.0
    assert sensor.descriptor_for_LLM == "Energy Levels"
    patches["session"].declare_subscriber.assert_called_once()
    args, _ = patches["session"].declare_subscriber.call_args
    assert args[0] == "lowstate"


def test_initialization_with_custom_topic(patches):
    config = UnitreeGo2BatteryZenohConfig(topic="lf/lowstate", api_key="abc", use_sim=True)
    UnitreeGo2BatteryZenoh(config=config)

    args, _ = patches["session"].declare_subscriber.call_args
    assert args[0] == "lf/lowstate"
    patches["status"].assert_called_once_with(api_key="abc")


def test_initialization_session_failure_does_not_raise(patches):
    patches["open_session"].side_effect = RuntimeError("zenoh down")
    config = UnitreeGo2BatteryZenohConfig()
    sensor = UnitreeGo2BatteryZenoh(config=config)
    assert sensor._session is None


def _make_sample(percentage=42.5, voltage=12.7, amperes=-1.3, t1=30, t2=32):
    msg = MagicMock()
    msg.bms_state.soc = percentage
    msg.power_v = voltage
    msg.power_a = amperes
    msg.temperature_ntc1 = t1
    msg.temperature_ntc2 = t2
    sample = MagicMock()
    sample.payload.to_bytes.return_value = b"binary"
    return sample, msg


def test_low_state_message_handler_updates_state(patches):
    config = UnitreeGo2BatteryZenohConfig()
    sensor = UnitreeGo2BatteryZenoh(config=config)
    sample, msg = _make_sample(percentage=42.55, voltage=12.7, amperes=-1.3, t1=30, t2=32)
    with patch(
        "inputs.plugins.unitree_go2_battery_zenoh.LowState.deserialize",
        return_value=msg,
    ):
        sensor.LowStateMessageHandler(sample)
    assert sensor.battery_percentage == 42.55
    assert sensor.battery_voltage == 12.7
    assert sensor.battery_amperes == -1.3
    assert sensor.battery_t == 31


def test_low_state_message_handler_decode_failure(patches):
    config = UnitreeGo2BatteryZenohConfig()
    sensor = UnitreeGo2BatteryZenoh(config=config)
    sample = MagicMock()
    sample.payload.to_bytes.return_value = b"garbage"
    with patch(
        "inputs.plugins.unitree_go2_battery_zenoh.LowState.deserialize",
        side_effect=ValueError("bad"),
    ):
        sensor.LowStateMessageHandler(sample)
    # Values stay at defaults
    assert sensor.battery_percentage == 0.0


def test_low_state_message_handler_incomplete_message(patches):
    config = UnitreeGo2BatteryZenohConfig()
    sensor = UnitreeGo2BatteryZenoh(config=config)
    msg = MagicMock(spec=[])  # no attributes -> AttributeError on access
    sample = MagicMock()
    sample.payload.to_bytes.return_value = b"x"
    with patch(
        "inputs.plugins.unitree_go2_battery_zenoh.LowState.deserialize",
        return_value=msg,
    ):
        sensor.LowStateMessageHandler(sample)
    assert sensor.battery_percentage == 0.0


@pytest.mark.asyncio
async def test_poll_returns_snapshot(patches):
    config = UnitreeGo2BatteryZenohConfig()
    sensor = UnitreeGo2BatteryZenoh(config=config)
    sensor.battery_percentage = 50.0
    sensor.battery_voltage = 12.0
    sensor.battery_amperes = -2.0
    with patch(
        "inputs.plugins.unitree_go2_battery_zenoh.asyncio.sleep",
        new=MagicMock(return_value=None),
    ) as _sleep:
        # asyncio.sleep needs to be awaitable; use AsyncMock
        from unittest.mock import AsyncMock

        _sleep.side_effect = None
        _sleep.return_value = None
        # Replace with proper AsyncMock
        with patch(
            "inputs.plugins.unitree_go2_battery_zenoh.asyncio.sleep", new=AsyncMock()
        ):
            result = await sensor._poll()
    assert result == [50.0, 12.0, -2.0]


@pytest.mark.asyncio
async def test_raw_to_text_critical(patches):
    config = UnitreeGo2BatteryZenohConfig()
    sensor = UnitreeGo2BatteryZenoh(config=config)
    msg = await sensor._raw_to_text([5.0, 12.0, -1.0])
    assert msg is not None
    assert "CRITICAL" in msg.message


@pytest.mark.asyncio
async def test_raw_to_text_warning(patches):
    config = UnitreeGo2BatteryZenohConfig()
    sensor = UnitreeGo2BatteryZenoh(config=config)
    msg = await sensor._raw_to_text([12.0, 12.0, -1.0])
    assert msg is not None
    assert "WARNING" in msg.message


@pytest.mark.asyncio
async def test_raw_to_text_above_threshold_returns_none(patches):
    config = UnitreeGo2BatteryZenohConfig()
    sensor = UnitreeGo2BatteryZenoh(config=config)
    msg = await sensor._raw_to_text([80.0, 12.0, -1.0])
    assert msg is None


@pytest.mark.asyncio
async def test_raw_to_text_appends_message_when_low(patches):
    config = UnitreeGo2BatteryZenohConfig()
    sensor = UnitreeGo2BatteryZenoh(config=config)
    await sensor.raw_to_text([5.0, 12.0, -1.0])
    assert len(sensor.messages) == 1


@pytest.mark.asyncio
async def test_raw_to_text_skips_when_safe(patches):
    config = UnitreeGo2BatteryZenohConfig()
    sensor = UnitreeGo2BatteryZenoh(config=config)
    await sensor.raw_to_text([90.0, 12.0, -1.0])
    assert sensor.messages == []


def test_formatted_latest_buffer_empty(patches):
    config = UnitreeGo2BatteryZenohConfig()
    sensor = UnitreeGo2BatteryZenoh(config=config)
    assert sensor.formatted_latest_buffer() is None


def test_formatted_latest_buffer_with_message(patches):
    from inputs.base import Message

    config = UnitreeGo2BatteryZenohConfig()
    sensor = UnitreeGo2BatteryZenoh(config=config)
    sensor.messages.append(Message(timestamp=1.0, message="WARNING: low battery"))
    result = sensor.formatted_latest_buffer()
    assert result is not None
    assert "WARNING: low battery" in result
    assert "Energy Levels" in result
