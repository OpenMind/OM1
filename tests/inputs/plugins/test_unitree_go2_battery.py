from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inputs.base import Message
from inputs.plugins.unitree_go2_battery import (
    UnitreeGo2Battery,
    UnitreeGo2BatteryConfig,
)


def test_initialization():
    """Test basic initialization."""
    with (
        patch("inputs.plugins.unitree_go2_battery.ChannelSubscriber"),
        patch("inputs.plugins.unitree_go2_battery.IOProvider"),
        patch("inputs.plugins.unitree_go2_battery.TeleopsStatusProvider"),
    ):
        config = UnitreeGo2BatteryConfig()
        sensor = UnitreeGo2Battery(config=config)

        assert sensor.messages == []
        assert sensor.battery_percentage == 0.0
        assert sensor.battery_voltage == 0.0
        assert sensor.battery_amperes == 0.0


def test_initialization_with_api_key():
    """Test initialization with API key."""
    with (
        patch("inputs.plugins.unitree_go2_battery.ChannelSubscriber"),
        patch("inputs.plugins.unitree_go2_battery.IOProvider"),
        patch("inputs.plugins.unitree_go2_battery.TeleopsStatusProvider"),
    ):
        config = UnitreeGo2BatteryConfig(api_key="test_key")
        sensor = UnitreeGo2Battery(config=config)

        assert sensor.config.api_key == "test_key"


@pytest.mark.asyncio
async def test_poll():
    """Test _poll method."""
    with (
        patch("inputs.plugins.unitree_go2_battery.ChannelSubscriber"),
        patch("inputs.plugins.unitree_go2_battery.IOProvider"),
        patch("inputs.plugins.unitree_go2_battery.TeleopsStatusProvider"),
    ):
        config = UnitreeGo2BatteryConfig()
        sensor = UnitreeGo2Battery(config=config)
        sensor.battery_percentage = 80.0
        sensor.battery_voltage = 24.5
        sensor.battery_amperes = 2.5

        with patch("inputs.plugins.unitree_go2_battery.asyncio.sleep", new=AsyncMock()):
            result = await sensor._poll()

        assert result is not None
        assert len(result) == 3
        assert result[0] == 80.0
        assert result[1] == 24.5
        assert result[2] == 2.5


@pytest.mark.asyncio
async def test_raw_to_text_with_low_battery():
    """Test _raw_to_text with low battery (warning level)."""
    with (
        patch("inputs.plugins.unitree_go2_battery.ChannelSubscriber"),
        patch("inputs.plugins.unitree_go2_battery.IOProvider"),
        patch("inputs.plugins.unitree_go2_battery.TeleopsStatusProvider"),
    ):
        config = UnitreeGo2BatteryConfig()
        sensor = UnitreeGo2Battery(config=config)

        with patch("inputs.plugins.unitree_go2_battery.time.time", return_value=1234.0):
            result = await sensor._raw_to_text([10.0, 25.0, 3.0])

        assert result is not None
        assert result.timestamp == 1234.0
        assert "WARNING" in result.message or "energy" in result.message.lower()


@pytest.mark.asyncio
async def test_raw_to_text_with_critical_battery():
    """Test _raw_to_text with critical battery level."""
    with (
        patch("inputs.plugins.unitree_go2_battery.ChannelSubscriber"),
        patch("inputs.plugins.unitree_go2_battery.IOProvider"),
        patch("inputs.plugins.unitree_go2_battery.TeleopsStatusProvider"),
    ):
        config = UnitreeGo2BatteryConfig()
        sensor = UnitreeGo2Battery(config=config)

        with patch("inputs.plugins.unitree_go2_battery.time.time", return_value=1234.0):
            result = await sensor._raw_to_text([5.0, 25.0, 3.0])

        assert result is not None
        assert result.timestamp == 1234.0
        assert "CRITICAL" in result.message


@pytest.mark.asyncio
async def test_raw_to_text_with_normal_battery():
    """Test _raw_to_text with normal battery level (no message)."""
    with (
        patch("inputs.plugins.unitree_go2_battery.ChannelSubscriber"),
        patch("inputs.plugins.unitree_go2_battery.IOProvider"),
        patch("inputs.plugins.unitree_go2_battery.TeleopsStatusProvider"),
    ):
        config = UnitreeGo2BatteryConfig()
        sensor = UnitreeGo2Battery(config=config)

        with patch("inputs.plugins.unitree_go2_battery.time.time", return_value=1234.0):
            result = await sensor._raw_to_text([85.0, 25.0, 3.0])

        assert result is None


def test_formatted_latest_buffer_with_messages():
    """Test formatted_latest_buffer with messages."""
    with (
        patch("inputs.plugins.unitree_go2_battery.ChannelSubscriber"),
        patch("inputs.plugins.unitree_go2_battery.IOProvider"),
        patch("inputs.plugins.unitree_go2_battery.TeleopsStatusProvider"),
    ):
        config = UnitreeGo2BatteryConfig()
        sensor = UnitreeGo2Battery(config=config)
        sensor.io_provider = MagicMock()

        sensor.messages = [
            Message(timestamp=1000.0, message="Battery: 85%"),
        ]

        result = sensor.formatted_latest_buffer()

        assert result is not None
        sensor.io_provider.add_input.assert_called_once()
        assert len(sensor.messages) == 0


def test_formatted_latest_buffer_empty():
    """Test formatted_latest_buffer with empty buffer."""
    with (
        patch("inputs.plugins.unitree_go2_battery.ChannelSubscriber"),
        patch("inputs.plugins.unitree_go2_battery.IOProvider"),
        patch("inputs.plugins.unitree_go2_battery.TeleopsStatusProvider"),
    ):
        config = UnitreeGo2BatteryConfig()
        sensor = UnitreeGo2Battery(config=config)

        result = sensor.formatted_latest_buffer()
        assert result is None


def test_initialization_with_use_sim():
    """Test initialization with Zenoh subscriber when use_sim is True."""
    with (
        patch("inputs.plugins.unitree_go2_battery.IOProvider"),
        patch("inputs.plugins.unitree_go2_battery.TeleopsStatusProvider"),
        patch("inputs.plugins.unitree_go2_battery.open_zenoh_session") as mock_zenoh,
    ):
        mock_session = MagicMock()
        mock_zenoh.return_value = mock_session

        config = UnitreeGo2BatteryConfig(use_sim=True, topic="test/lowstate")
        sensor = UnitreeGo2Battery(config=config)

        mock_zenoh.assert_called_once()
        mock_session.declare_subscriber.assert_called_once()
        assert sensor._lowstate_cyclonedds_subscriber is None
        assert sensor._lowstate_zenoh_subscriber is not None


def test_initialization_zenoh_failure():
    """Test initialization when Zenoh session fails."""
    with (
        patch("inputs.plugins.unitree_go2_battery.IOProvider"),
        patch("inputs.plugins.unitree_go2_battery.TeleopsStatusProvider"),
        patch("inputs.plugins.unitree_go2_battery.open_zenoh_session", side_effect=Exception("Connection failed")),
    ):
        config = UnitreeGo2BatteryConfig(use_sim=True)
        sensor = UnitreeGo2Battery(config=config)

        assert sensor._lowstate_zenoh_subscriber is None


def test_initialization_cyclonedds_failure():
    """Test initialization when CycloneDDS subscriber fails."""
    with (
        patch("inputs.plugins.unitree_go2_battery.IOProvider"),
        patch("inputs.plugins.unitree_go2_battery.TeleopsStatusProvider"),
        patch("inputs.plugins.unitree_go2_battery.ChannelSubscriber", side_effect=Exception("DDS init failed")),
    ):
        config = UnitreeGo2BatteryConfig(use_sim=False)
        sensor = UnitreeGo2Battery(config=config)

        assert sensor._lowstate_cyclonedds_subscriber is None


def test_lowstate_message_handler_cyclonedds():
    """Test LowStateMessageHandler with CycloneDDS message."""
    with (
        patch("inputs.plugins.unitree_go2_battery.ChannelSubscriber"),
        patch("inputs.plugins.unitree_go2_battery.IOProvider"),
        patch("inputs.plugins.unitree_go2_battery.TeleopsStatusProvider"),
    ):
        config = UnitreeGo2BatteryConfig()
        sensor = UnitreeGo2Battery(config=config)

        mock_msg = MagicMock()
        mock_msg.bms_state.soc = 75.5
        mock_msg.power_v = 24.8
        mock_msg.power_a = 3.2
        mock_msg.temperature_ntc1 = 30
        mock_msg.temperature_ntc2 = 32

        sensor.LowStateMessageHandler(mock_msg)

        assert sensor.battery_percentage == 75.5
        assert sensor.battery_voltage == 24.8
        assert sensor.battery_amperes == 3.2
        assert sensor.battery_t == 31


def test_lowstate_message_handler_zenoh():
    """Test LowStateMessageHandler with Zenoh message."""
    with (
        patch("inputs.plugins.unitree_go2_battery.IOProvider"),
        patch("inputs.plugins.unitree_go2_battery.TeleopsStatusProvider"),
        patch("inputs.plugins.unitree_go2_battery.open_zenoh_session") as mock_zenoh,
    ):
        mock_session = MagicMock()
        mock_zenoh.return_value = mock_session

        config = UnitreeGo2BatteryConfig(use_sim=True)
        sensor = UnitreeGo2Battery(config=config)

        mock_sample = MagicMock()
        mock_payload = MagicMock()
        mock_payload.to_bytes.return_value = b"test_payload"
        mock_sample.payload = mock_payload

        mock_lowstate = MagicMock()
        mock_lowstate.bms_state.soc = 82.3
        mock_lowstate.power_v = 25.1
        mock_lowstate.power_a = 2.8
        mock_lowstate.temperature_ntc1 = 28
        mock_lowstate.temperature_ntc2 = 30

        with patch("inputs.plugins.unitree_go2_battery.LowState_.deserialize", return_value=mock_lowstate):
            sensor.LowStateMessageHandler(mock_sample)

        assert sensor.battery_percentage == 82.3
        assert sensor.battery_voltage == 25.1
        assert sensor.battery_amperes == 2.8
        assert sensor.battery_t == 29


def test_lowstate_message_handler_incomplete_message():
    """Test LowStateMessageHandler with incomplete message (AttributeError)."""
    with (
        patch("inputs.plugins.unitree_go2_battery.ChannelSubscriber"),
        patch("inputs.plugins.unitree_go2_battery.IOProvider"),
        patch("inputs.plugins.unitree_go2_battery.TeleopsStatusProvider"),
    ):
        config = UnitreeGo2BatteryConfig()
        sensor = UnitreeGo2Battery(config=config)

        sensor.battery_percentage = 50.0
        sensor.battery_voltage = 24.0
        sensor.battery_amperes = 2.0
        sensor.battery_t = 25

        mock_msg = MagicMock(spec=["other_field"])

        sensor.LowStateMessageHandler(mock_msg)

        assert sensor.battery_percentage == 0.0
        assert sensor.battery_voltage == 0.0
        assert sensor.battery_amperes == 0.0
        assert sensor.battery_t == 0


def test_lowstate_message_handler_zenoh_deserialization_failure():
    """Test LowStateMessageHandler when Zenoh deserialization fails."""
    with (
        patch("inputs.plugins.unitree_go2_battery.IOProvider"),
        patch("inputs.plugins.unitree_go2_battery.TeleopsStatusProvider"),
        patch("inputs.plugins.unitree_go2_battery.open_zenoh_session") as mock_zenoh,
    ):
        mock_session = MagicMock()
        mock_zenoh.return_value = mock_session

        config = UnitreeGo2BatteryConfig(use_sim=True)
        sensor = UnitreeGo2Battery(config=config)

        sensor.battery_percentage = 50.0

        mock_sample = MagicMock()
        mock_payload = MagicMock()
        mock_payload.to_bytes.side_effect = Exception("Deserialization error")
        mock_sample.payload = mock_payload

        with patch(
            "inputs.plugins.unitree_go2_battery.LowState_.deserialize", side_effect=Exception("Deserialization failed")
        ):
            sensor.LowStateMessageHandler(mock_sample)

        assert sensor.battery_percentage == 50.0


@pytest.mark.asyncio
async def test_report_status():
    """Test report_status method."""
    with (
        patch("inputs.plugins.unitree_go2_battery.ChannelSubscriber"),
        patch("inputs.plugins.unitree_go2_battery.IOProvider"),
        patch("inputs.plugins.unitree_go2_battery.TeleopsStatusProvider") as mock_provider_class,
    ):
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = UnitreeGo2BatteryConfig()
        sensor = UnitreeGo2Battery(config=config)
        sensor.battery_percentage = 65.0
        sensor.battery_voltage = 24.5
        sensor.battery_t = 28

        with patch("inputs.plugins.unitree_go2_battery.time.time", return_value=5000.0):
            await sensor.report_status()

        mock_provider.share_status.assert_called_once()
        call_args = mock_provider.share_status.call_args[0][0]
        assert call_args.machine_name == "UnitreeGo2"
        assert call_args.battery_status.battery_level == 65.0
        assert call_args.battery_status.temperature == 28
        assert call_args.battery_status.voltage == 24.5
        assert call_args.battery_status.charging_status is False


@pytest.mark.asyncio
async def test_raw_to_text_appends_to_buffer():
    """Test raw_to_text method appends messages to buffer."""
    with (
        patch("inputs.plugins.unitree_go2_battery.ChannelSubscriber"),
        patch("inputs.plugins.unitree_go2_battery.IOProvider"),
        patch("inputs.plugins.unitree_go2_battery.TeleopsStatusProvider"),
    ):
        config = UnitreeGo2BatteryConfig()
        sensor = UnitreeGo2Battery(config=config)

        assert len(sensor.messages) == 0

        with patch("inputs.plugins.unitree_go2_battery.time.time", return_value=1000.0):
            await sensor.raw_to_text([5.0, 25.0, 3.0])

        assert len(sensor.messages) == 1
        assert "CRITICAL" in sensor.messages[0].message

        with patch("inputs.plugins.unitree_go2_battery.time.time", return_value=2000.0):
            await sensor.raw_to_text([10.0, 25.0, 3.0])

        assert len(sensor.messages) == 2

        with patch("inputs.plugins.unitree_go2_battery.time.time", return_value=3000.0):
            await sensor.raw_to_text([85.0, 25.0, 3.0])

        assert len(sensor.messages) == 2


@pytest.mark.asyncio
async def test_poll_calls_report_status():
    """Test that _poll calls report_status."""
    with (
        patch("inputs.plugins.unitree_go2_battery.ChannelSubscriber"),
        patch("inputs.plugins.unitree_go2_battery.IOProvider"),
        patch("inputs.plugins.unitree_go2_battery.TeleopsStatusProvider"),
    ):
        config = UnitreeGo2BatteryConfig()
        sensor = UnitreeGo2Battery(config=config)
        sensor.battery_percentage = 75.0
        sensor.battery_voltage = 24.0
        sensor.battery_amperes = 2.5

        sensor.report_status = AsyncMock()

        with patch("inputs.plugins.unitree_go2_battery.asyncio.sleep", new=AsyncMock()):
            await sensor._poll()

        sensor.report_status.assert_called_once()


def test_formatted_latest_buffer_with_descriptor():
    """Test formatted_latest_buffer includes descriptor_for_LLM."""
    with (
        patch("inputs.plugins.unitree_go2_battery.ChannelSubscriber"),
        patch("inputs.plugins.unitree_go2_battery.IOProvider"),
        patch("inputs.plugins.unitree_go2_battery.TeleopsStatusProvider"),
    ):
        config = UnitreeGo2BatteryConfig()
        sensor = UnitreeGo2Battery(config=config)
        sensor.io_provider = MagicMock()

        sensor.messages = [
            Message(timestamp=1000.0, message="Low battery warning"),
        ]

        result = sensor.formatted_latest_buffer()

        assert result is not None
        assert "Energy Levels" in result
        assert "Low battery warning" in result
