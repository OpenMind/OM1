import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inputs.base import Message
from inputs.plugins.unitree_go2_imu import (
    UnitreeGo2IMU,
    UnitreeGo2IMUConfig,
)


MODULE = "inputs.plugins.unitree_go2_imu"


def _make_sensor(**config_kwargs) -> UnitreeGo2IMU:
    """Create a sensor instance with mocked external dependencies."""
    config = UnitreeGo2IMUConfig(**config_kwargs)
    return UnitreeGo2IMU(config=config)


class TestInitialization:
    """Test UnitreeGo2IMU initialization."""

    def test_init_subscribes_to_lowstate(self):
        """Test that initialization creates a ChannelSubscriber for rt/lowstate."""
        with (
            patch(f"{MODULE}.ChannelSubscriber") as mock_sub,
            patch(f"{MODULE}.IOProvider"),
        ):
            sensor = _make_sensor()

            mock_sub.assert_called_once()
            call_args = mock_sub.call_args[0]
            assert call_args[0] == "rt/lowstate"
            assert sensor.roll_deg == 0.0
            assert sensor.pitch_deg == 0.0
            assert sensor.messages == []

    def test_init_default_thresholds(self):
        """Test default threshold values."""
        with (
            patch(f"{MODULE}.ChannelSubscriber"),
            patch(f"{MODULE}.IOProvider"),
        ):
            sensor = _make_sensor()

            assert sensor.config.fall_threshold_deg == 45.0
            assert sensor.config.warning_threshold_deg == 30.0

    def test_init_custom_thresholds(self):
        """Test custom threshold values."""
        with (
            patch(f"{MODULE}.ChannelSubscriber"),
            patch(f"{MODULE}.IOProvider"),
        ):
            sensor = _make_sensor(fall_threshold_deg=60.0, warning_threshold_deg=40.0)

            assert sensor.config.fall_threshold_deg == 60.0
            assert sensor.config.warning_threshold_deg == 40.0

    def test_init_subscriber_error(self):
        """Test initialization when ChannelSubscriber fails."""
        with (
            patch(f"{MODULE}.ChannelSubscriber", side_effect=Exception("No DDS")),
            patch(f"{MODULE}.IOProvider"),
            patch(f"{MODULE}.logging") as mock_logging,
        ):
            sensor = _make_sensor()

            assert sensor.lowstate_subscriber is None
            mock_logging.error.assert_called()


class TestLowStateHandler:
    """Test low_state_handler method."""

    def test_handler_extracts_rpy(self):
        """Test that handler correctly extracts roll and pitch from IMU data."""
        with (
            patch(f"{MODULE}.ChannelSubscriber"),
            patch(f"{MODULE}.IOProvider"),
        ):
            sensor = _make_sensor()

            msg = MagicMock()
            msg.imu_state.rpy = [0.5, -0.3, 1.0]

            sensor.low_state_handler(msg)

            assert sensor.roll_deg == round(math.degrees(0.5), 2)
            assert sensor.pitch_deg == round(math.degrees(-0.3), 2)

    def test_handler_incomplete_message(self):
        """Test handler with incomplete IMU data resets to zero."""
        with (
            patch(f"{MODULE}.ChannelSubscriber"),
            patch(f"{MODULE}.IOProvider"),
        ):
            sensor = _make_sensor()
            sensor.roll_deg = 10.0
            sensor.pitch_deg = 20.0

            msg = MagicMock()
            msg.imu_state = None

            sensor.low_state_handler(msg)

            assert sensor.roll_deg == 0.0
            assert sensor.pitch_deg == 0.0


class TestPoll:
    """Test _poll method."""

    @pytest.mark.asyncio
    async def test_poll_returns_none_when_normal(self):
        """Test that _poll returns None when orientation is normal."""
        with (
            patch(f"{MODULE}.ChannelSubscriber"),
            patch(f"{MODULE}.IOProvider"),
        ):
            sensor = _make_sensor()
            sensor.roll_deg = 5.0
            sensor.pitch_deg = -3.0

            with patch(f"{MODULE}.asyncio.sleep", new=AsyncMock()):
                result = await sensor._poll()

            assert result is None

    @pytest.mark.asyncio
    async def test_poll_returns_warning_on_tilt(self):
        """Test that _poll returns WARNING when tilt exceeds warning threshold."""
        with (
            patch(f"{MODULE}.ChannelSubscriber"),
            patch(f"{MODULE}.IOProvider"),
        ):
            sensor = _make_sensor()
            sensor.roll_deg = 35.0
            sensor.pitch_deg = 5.0

            with patch(f"{MODULE}.asyncio.sleep", new=AsyncMock()):
                result = await sensor._poll()

            assert result is not None
            assert "WARNING" in result
            assert "35.0" in result

    @pytest.mark.asyncio
    async def test_poll_returns_critical_on_fall(self):
        """Test that _poll returns CRITICAL when tilt exceeds fall threshold."""
        with (
            patch(f"{MODULE}.ChannelSubscriber"),
            patch(f"{MODULE}.IOProvider"),
        ):
            sensor = _make_sensor()
            sensor.roll_deg = 60.0
            sensor.pitch_deg = 10.0

            with patch(f"{MODULE}.asyncio.sleep", new=AsyncMock()):
                result = await sensor._poll()

            assert result is not None
            assert "CRITICAL" in result
            assert "recover" in result.lower()

    @pytest.mark.asyncio
    async def test_poll_returns_critical_on_pitch_fall(self):
        """Test that _poll detects fall from pitch exceeding threshold."""
        with (
            patch(f"{MODULE}.ChannelSubscriber"),
            patch(f"{MODULE}.IOProvider"),
        ):
            sensor = _make_sensor()
            sensor.roll_deg = 2.0
            sensor.pitch_deg = -50.0

            with patch(f"{MODULE}.asyncio.sleep", new=AsyncMock()):
                result = await sensor._poll()

            assert result is not None
            assert "CRITICAL" in result

    @pytest.mark.asyncio
    async def test_poll_returns_none_no_data(self):
        """Test that _poll returns None when no IMU data received (defaults)."""
        with (
            patch(f"{MODULE}.ChannelSubscriber"),
            patch(f"{MODULE}.IOProvider"),
        ):
            sensor = _make_sensor()

            with patch(f"{MODULE}.asyncio.sleep", new=AsyncMock()):
                result = await sensor._poll()

            assert result is None


class TestRawToText:
    """Test _raw_to_text and raw_to_text methods."""

    @pytest.mark.asyncio
    async def test_raw_to_text_formats_message(self):
        """Test that _raw_to_text creates a Message from alert string."""
        with (
            patch(f"{MODULE}.ChannelSubscriber"),
            patch(f"{MODULE}.IOProvider"),
        ):
            sensor = _make_sensor()
            alert = "CRITICAL: You have fallen over."

            with patch(f"{MODULE}.time.time", return_value=1234.0):
                result = await sensor._raw_to_text(alert)

            assert result is not None
            assert result.timestamp == 1234.0
            assert result.message == alert

    @pytest.mark.asyncio
    async def test_raw_to_text_returns_none_for_none(self):
        """Test that _raw_to_text returns None when input is None."""
        with (
            patch(f"{MODULE}.ChannelSubscriber"),
            patch(f"{MODULE}.IOProvider"),
        ):
            sensor = _make_sensor()

            result = await sensor._raw_to_text(None)

            assert result is None

    @pytest.mark.asyncio
    async def test_raw_to_text_appends_to_buffer(self):
        """Test that raw_to_text appends message to buffer."""
        with (
            patch(f"{MODULE}.ChannelSubscriber"),
            patch(f"{MODULE}.IOProvider"),
        ):
            sensor = _make_sensor()

            with patch(f"{MODULE}.time.time", return_value=5000.0):
                await sensor.raw_to_text("WARNING: tilting")

            assert len(sensor.messages) == 1
            assert sensor.messages[0].message == "WARNING: tilting"

    @pytest.mark.asyncio
    async def test_raw_to_text_skips_none(self):
        """Test that raw_to_text does not append when input is None."""
        with (
            patch(f"{MODULE}.ChannelSubscriber"),
            patch(f"{MODULE}.IOProvider"),
        ):
            sensor = _make_sensor()

            await sensor.raw_to_text(None)

            assert len(sensor.messages) == 0


class TestFormattedLatestBuffer:
    """Test formatted_latest_buffer method."""

    def test_formatted_latest_buffer_with_messages(self):
        """Test formatted output with messages in buffer."""
        with (
            patch(f"{MODULE}.ChannelSubscriber"),
            patch(f"{MODULE}.IOProvider"),
        ):
            sensor = _make_sensor()
            sensor.io_provider = MagicMock()
            sensor.messages = [
                Message(timestamp=1000.0, message="CRITICAL: fallen over"),
            ]

            result = sensor.formatted_latest_buffer()

            assert result is not None
            assert "Body Orientation" in result
            assert "CRITICAL: fallen over" in result
            sensor.io_provider.add_input.assert_called_once()
            assert len(sensor.messages) == 0

    def test_formatted_latest_buffer_empty(self):
        """Test formatted output with empty buffer returns None."""
        with (
            patch(f"{MODULE}.ChannelSubscriber"),
            patch(f"{MODULE}.IOProvider"),
        ):
            sensor = _make_sensor()

            result = sensor.formatted_latest_buffer()

            assert result is None
