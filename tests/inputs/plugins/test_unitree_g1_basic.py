from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inputs.plugins.unitree_g1_basic import UnitreeG1Basic, UnitreeG1BasicConfig


@pytest.fixture
def sensor():
    with (
        patch("inputs.plugins.unitree_g1_basic.ChannelSubscriber"),
        patch("inputs.plugins.unitree_g1_basic.IOProvider"),
        patch("inputs.plugins.unitree_g1_basic.TeleopsStatusProvider"),
    ):
        config = UnitreeG1BasicConfig()
        yield UnitreeG1Basic(config=config)


def test_initialization(sensor):
    """Test basic initialization."""
    assert sensor.messages == []
    assert sensor.battery_percentage == 0.0
    assert sensor.battery_voltage == 0.0
    assert sensor.battery_amperes == 0.0


def test_initialization_with_api_key():
    """Test initialization with API key."""
    with (
        patch("inputs.plugins.unitree_g1_basic.ChannelSubscriber"),
        patch("inputs.plugins.unitree_g1_basic.IOProvider"),
        patch("inputs.plugins.unitree_g1_basic.TeleopsStatusProvider"),
    ):
        config = UnitreeG1BasicConfig(api_key="test_key")
        sensor = UnitreeG1Basic(config=config)
        assert sensor.config.api_key == "test_key"


def test_initialization_with_ethernet():
    """Test initialization with unitree_ethernet sets up subscribers."""
    with (
        patch("inputs.plugins.unitree_g1_basic.IOProvider"),
        patch("inputs.plugins.unitree_g1_basic.TeleopsStatusProvider"),
        patch(
            "inputs.plugins.unitree_g1_basic.ChannelSubscriber"
        ) as mock_subscriber_cls,
    ):
        mock_subscriber = MagicMock()
        mock_subscriber_cls.return_value = mock_subscriber

        config = UnitreeG1BasicConfig(unitree_ethernet="eth0")
        sensor = UnitreeG1Basic(config=config)

        assert mock_subscriber_cls.call_count == 2
        assert mock_subscriber.Init.call_count == 2
        assert sensor.lowstate_subscriber is not None
        assert sensor.bmsstate_subscriber is not None


def test_bms_state_handler(sensor):
    """Test BMSStateHandler updates battery fields from message."""
    msg = MagicMock()
    msg.bmsvoltage = [48000]
    msg.current = 5
    msg.soc = 85
    msg.temperature = [30]

    sensor.BMSStateHandler(msg)

    assert sensor.battery_voltage == 48000.0
    assert sensor.battery_amperes == 5.0
    assert sensor.battery_percentage == 85.0
    assert sensor.battery_temperature == 30.0


def test_low_state_handler(sensor):
    """Test LowStateHandler stores message."""
    msg = MagicMock()
    sensor.LowStateHandler(msg)
    assert sensor.low_state is msg


@pytest.mark.asyncio
async def test_poll(sensor):
    """Test _poll method."""
    sensor.battery_percentage = 75.0
    sensor.battery_voltage = 48.5
    sensor.battery_amperes = 3.2

    with patch("inputs.plugins.unitree_g1_basic.asyncio.sleep", new=AsyncMock()):
        result = await sensor._poll()

    assert result == [75.0, 48.5, 3.2]


@pytest.mark.asyncio
async def test_poll_calls_update_status(sensor):
    """Test _poll calls update_status."""
    sensor.update_status = AsyncMock()

    with patch("inputs.plugins.unitree_g1_basic.asyncio.sleep", new=AsyncMock()):
        await sensor._poll()

    sensor.update_status.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_status(sensor):
    """Test update_status calls share_status with correct data."""
    sensor.battery_percentage = 60.0
    sensor.battery_temperature = 25.0
    sensor.battery_voltage = 47.0

    with patch("inputs.plugins.unitree_g1_basic.time.time", return_value=1000.0):
        await sensor.update_status()

    sensor.status_provider.share_status.assert_called_once()
    call_arg = sensor.status_provider.share_status.call_args[0][0]
    assert call_arg.machine_name == "UnitreeG1"
    assert call_arg.battery_status.battery_level == 60.0
    assert call_arg.battery_status.voltage == 47.0


@pytest.mark.asyncio
async def test_raw_to_text_with_low_battery(sensor):
    """Test _raw_to_text with critically low battery (<20%)."""
    with patch("inputs.plugins.unitree_g1_basic.time.time", return_value=1234.0):
        result = await sensor._raw_to_text([10.0, 48.0, 3.0])

    assert result is not None
    assert result.timestamp == 1234.0
    assert "SIT DOWN NOW" in result.message


@pytest.mark.asyncio
async def test_raw_to_text_with_medium_low_battery(sensor):
    """Test _raw_to_text with medium-low battery (20-30%)."""
    with patch("inputs.plugins.unitree_g1_basic.time.time", return_value=5678.0):
        result = await sensor._raw_to_text([25.0, 48.0, 3.0])

    assert result is not None
    assert result.timestamp == 5678.0
    assert "Consider sitting down" in result.message


@pytest.mark.asyncio
async def test_raw_to_text_with_normal_battery(sensor):
    """Test _raw_to_text with sufficient battery returns None."""
    result = await sensor._raw_to_text([80.0, 48.0, 3.0])
    assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_appends_to_buffer(sensor):
    """Test raw_to_text appends message to buffer when battery is low."""
    with patch("inputs.plugins.unitree_g1_basic.time.time", return_value=1000.0):
        await sensor.raw_to_text([10.0, 48.0, 3.0])

    assert len(sensor.messages) == 1
    assert "SIT DOWN NOW" in sensor.messages[0].message


@pytest.mark.asyncio
async def test_raw_to_text_no_append_when_normal(sensor):
    """Test raw_to_text does not append to buffer when battery is sufficient."""
    await sensor.raw_to_text([80.0, 48.0, 3.0])
    assert len(sensor.messages) == 0


def test_formatted_latest_buffer_empty(sensor):
    """Test formatted_latest_buffer returns None when buffer is empty."""
    assert sensor.formatted_latest_buffer() is None


def test_formatted_latest_buffer_with_message(sensor):
    """Test formatted_latest_buffer returns formatted string and clears buffer."""
    from inputs.base import Message

    sensor.messages = [Message(timestamp=1000.0, message="WARNING: test")]

    result = sensor.formatted_latest_buffer()

    assert result is not None
    assert "WARNING: test" in result
    assert "Energy Level" in result
    assert sensor.messages == []
    sensor.io_provider.add_input.assert_called_once_with(
        "Energy Level", "WARNING: test", 1000.0
    )
