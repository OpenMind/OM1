from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inputs.base import Message
from inputs.plugins.booster_odom import BoosterOdom, BoosterOdomConfig
from providers.k1_odom_provider import RobotState


def test_initialization():
    """Test basic initialization with default config."""
    with (
        patch("inputs.plugins.booster_odom.K1OdomProvider"),
        patch("inputs.plugins.booster_odom.IOProvider"),
    ):
        config = BoosterOdomConfig()
        sensor = BoosterOdom(config=config)

        assert sensor.messages == []
        assert "location" in sensor.descriptor_for_LLM.lower() or "pose" in sensor.descriptor_for_LLM.lower()


def test_initialization_with_custom_topic():
    """Test initialization with custom topic."""
    with (
        patch("inputs.plugins.booster_odom.K1OdomProvider") as mock_provider,
        patch("inputs.plugins.booster_odom.IOProvider"),
    ):
        config = BoosterOdomConfig(topic="custom_odom")
        sensor = BoosterOdom(config=config)

        assert sensor.config.topic == "custom_odom"
        mock_provider.assert_called_once_with("custom_odom")


def test_initialization_default_topic():
    """Test initialization uses default topic."""
    with (
        patch("inputs.plugins.booster_odom.K1OdomProvider") as mock_provider,
        patch("inputs.plugins.booster_odom.IOProvider"),
    ):
        config = BoosterOdomConfig()
        sensor = BoosterOdom(config=config)

        assert sensor.config.topic == "odometer_state"
        mock_provider.assert_called_once_with("odometer_state")


class TestBoosterOdomConfig:
    """Test BoosterOdomConfig configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BoosterOdomConfig()
        assert config.topic == "odometer_state"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = BoosterOdomConfig(topic="my_custom_topic")
        assert config.topic == "my_custom_topic"

    def test_config_is_pydantic_model(self):
        """Test that BoosterOdomConfig is a Pydantic model."""
        BoosterOdomConfig()
        assert hasattr(BoosterOdomConfig, "model_fields") or hasattr(BoosterOdomConfig, "__fields__")


class TestPoll:
    """Test _poll method."""

    @pytest.mark.asyncio
    async def test_poll_with_position_data(self):
        """Test _poll with position data available."""
        with (
            patch("inputs.plugins.booster_odom.K1OdomProvider") as mock_provider_class,
            patch("inputs.plugins.booster_odom.IOProvider"),
        ):
            mock_provider = MagicMock()
            mock_provider.position = {
                "odom_x": 1.0,
                "odom_y": 2.0,
                "moving": False,
                "body_attitude": RobotState.STANDING,
            }
            mock_provider_class.return_value = mock_provider

            config = BoosterOdomConfig()
            sensor = BoosterOdom(config=config)

            with patch("inputs.plugins.booster_odom.asyncio.sleep", new=AsyncMock()):
                result = await sensor._poll()

            assert result == {
                "odom_x": 1.0,
                "odom_y": 2.0,
                "moving": False,
                "body_attitude": RobotState.STANDING,
            }

    @pytest.mark.asyncio
    async def test_poll_sleeps_briefly(self):
        """Test _poll includes sleep to prevent excessive CPU usage."""
        with (
            patch("inputs.plugins.booster_odom.K1OdomProvider") as mock_provider_class,
            patch("inputs.plugins.booster_odom.IOProvider"),
        ):
            mock_provider = MagicMock()
            mock_provider.position = {"moving": False}
            mock_provider_class.return_value = mock_provider

            config = BoosterOdomConfig()
            sensor = BoosterOdom(config=config)

            with patch("inputs.plugins.booster_odom.asyncio.sleep", new=AsyncMock()) as mock_sleep:
                await sensor._poll()

                mock_sleep.assert_called_once_with(0.1)


class TestRawToText:
    """Test _raw_to_text method."""

    @pytest.mark.asyncio
    async def test_raw_to_text_sitting(self):
        """Test _raw_to_text when robot is sitting."""
        with (
            patch("inputs.plugins.booster_odom.K1OdomProvider"),
            patch("inputs.plugins.booster_odom.IOProvider"),
        ):
            config = BoosterOdomConfig()
            sensor = BoosterOdom(config=config)

            position_data = {"moving": False, "body_attitude": RobotState.SITTING}

            with patch("inputs.plugins.booster_odom.time.time", return_value=1234.0):
                result = await sensor._raw_to_text(position_data)

            assert result is not None
            assert result.timestamp == 1234.0
            assert "sitting" in result.message.lower()
            assert "do not generate" in result.message.lower()

    @pytest.mark.asyncio
    async def test_raw_to_text_moving(self):
        """Test _raw_to_text when robot is moving."""
        with (
            patch("inputs.plugins.booster_odom.K1OdomProvider"),
            patch("inputs.plugins.booster_odom.IOProvider"),
        ):
            config = BoosterOdomConfig()
            sensor = BoosterOdom(config=config)

            position_data = {"moving": True, "body_attitude": RobotState.STANDING}

            with patch("inputs.plugins.booster_odom.time.time", return_value=1234.0):
                result = await sensor._raw_to_text(position_data)

            assert result is not None
            assert result.timestamp == 1234.0
            assert "moving" in result.message.lower()
            assert "do not generate" in result.message.lower()

    @pytest.mark.asyncio
    async def test_raw_to_text_standing_still(self):
        """Test _raw_to_text when robot is standing still."""
        with (
            patch("inputs.plugins.booster_odom.K1OdomProvider"),
            patch("inputs.plugins.booster_odom.IOProvider"),
        ):
            config = BoosterOdomConfig()
            sensor = BoosterOdom(config=config)

            position_data = {"moving": False, "body_attitude": RobotState.STANDING}

            with patch("inputs.plugins.booster_odom.time.time", return_value=1234.0):
                result = await sensor._raw_to_text(position_data)

            assert result is not None
            assert result.timestamp == 1234.0
            assert "standing still" in result.message.lower()
            assert "can move" in result.message.lower()

    @pytest.mark.asyncio
    async def test_raw_to_text_with_none(self):
        """Test _raw_to_text with None input."""
        with (
            patch("inputs.plugins.booster_odom.K1OdomProvider"),
            patch("inputs.plugins.booster_odom.IOProvider"),
        ):
            config = BoosterOdomConfig()
            sensor = BoosterOdom(config=config)

            result = await sensor._raw_to_text(None)
            assert result is None

    @pytest.mark.asyncio
    async def test_raw_to_text_returns_message_type(self):
        """Test _raw_to_text returns Message object."""
        with (
            patch("inputs.plugins.booster_odom.K1OdomProvider"),
            patch("inputs.plugins.booster_odom.IOProvider"),
        ):
            config = BoosterOdomConfig()
            sensor = BoosterOdom(config=config)

            position_data = {"moving": False, "body_attitude": RobotState.STANDING}

            with patch("inputs.plugins.booster_odom.time.time", return_value=1234.0):
                result = await sensor._raw_to_text(position_data)

            assert isinstance(result, Message)
            assert hasattr(result, "timestamp")
            assert hasattr(result, "message")


class TestRawToTextWrapper:
    """Test raw_to_text wrapper method."""

    @pytest.mark.asyncio
    async def test_raw_to_text_wrapper_with_valid_input(self):
        """Test raw_to_text appends message to buffer."""
        with (
            patch("inputs.plugins.booster_odom.K1OdomProvider"),
            patch("inputs.plugins.booster_odom.IOProvider"),
        ):
            config = BoosterOdomConfig()
            sensor = BoosterOdom(config=config)

            position_data = {"moving": False, "body_attitude": RobotState.STANDING}

            with patch("inputs.plugins.booster_odom.time.time", return_value=1234.0):
                await sensor.raw_to_text(position_data)

            assert len(sensor.messages) == 1
            assert sensor.messages[0].timestamp == 1234.0

    @pytest.mark.asyncio
    async def test_raw_to_text_wrapper_with_none(self):
        """Test raw_to_text returns early with None input."""
        with (
            patch("inputs.plugins.booster_odom.K1OdomProvider"),
            patch("inputs.plugins.booster_odom.IOProvider"),
        ):
            config = BoosterOdomConfig()
            sensor = BoosterOdom(config=config)

            await sensor.raw_to_text(None)

            assert len(sensor.messages) == 0

    @pytest.mark.asyncio
    async def test_raw_to_text_wrapper_accumulates_messages(self):
        """Test raw_to_text accumulates multiple messages."""
        with (
            patch("inputs.plugins.booster_odom.K1OdomProvider"),
            patch("inputs.plugins.booster_odom.IOProvider"),
        ):
            config = BoosterOdomConfig()
            sensor = BoosterOdom(config=config)

            position_data1 = {"moving": False, "body_attitude": RobotState.STANDING}
            position_data2 = {"moving": True, "body_attitude": RobotState.STANDING}

            with patch("inputs.plugins.booster_odom.time.time", return_value=1234.0):
                await sensor.raw_to_text(position_data1)

            with patch("inputs.plugins.booster_odom.time.time", return_value=1235.0):
                await sensor.raw_to_text(position_data2)

            assert len(sensor.messages) == 2


class TestFormattedLatestBuffer:
    """Test formatted_latest_buffer method."""

    def test_formatted_latest_buffer_with_messages(self):
        """Test formatted_latest_buffer returns formatted string."""
        with (
            patch("inputs.plugins.booster_odom.K1OdomProvider"),
            patch("inputs.plugins.booster_odom.IOProvider") as mock_io_provider_class,
        ):
            mock_io_provider = MagicMock()
            mock_io_provider_class.return_value = mock_io_provider

            config = BoosterOdomConfig()
            sensor = BoosterOdom(config=config)

            # Add a message
            test_message = Message(timestamp=1234.0, message="Test message")
            sensor.messages.append(test_message)

            result = sensor.formatted_latest_buffer()

            assert result is not None
            assert "INPUT:" in result
            assert "Test message" in result
            assert "START" in result
            assert "END" in result

            # Verify IO provider was called
            mock_io_provider.add_input.assert_called_once_with(sensor.descriptor_for_LLM, "Test message", 1234.0)

            # Verify buffer is cleared
            assert len(sensor.messages) == 0

    def test_formatted_latest_buffer_empty(self):
        """Test formatted_latest_buffer returns None when no messages."""
        with (
            patch("inputs.plugins.booster_odom.K1OdomProvider"),
            patch("inputs.plugins.booster_odom.IOProvider"),
        ):
            config = BoosterOdomConfig()
            sensor = BoosterOdom(config=config)

            result = sensor.formatted_latest_buffer()

            assert result is None

    def test_formatted_latest_buffer_uses_latest_message(self):
        """Test formatted_latest_buffer uses the latest message."""
        with (
            patch("inputs.plugins.booster_odom.K1OdomProvider"),
            patch("inputs.plugins.booster_odom.IOProvider") as mock_io_provider_class,
        ):
            mock_io_provider = MagicMock()
            mock_io_provider_class.return_value = mock_io_provider

            config = BoosterOdomConfig()
            sensor = BoosterOdom(config=config)

            # Add multiple messages
            sensor.messages.append(Message(timestamp=1234.0, message="Old message"))
            sensor.messages.append(Message(timestamp=1235.0, message="Latest message"))

            result = sensor.formatted_latest_buffer()

            assert result is not None
            assert "Latest message" in result
            assert "Old message" not in result

            # Verify IO provider was called with latest message
            mock_io_provider.add_input.assert_called_once_with(sensor.descriptor_for_LLM, "Latest message", 1235.0)

    def test_formatted_latest_buffer_clears_messages(self):
        """Test formatted_latest_buffer clears the message buffer."""
        with (
            patch("inputs.plugins.booster_odom.K1OdomProvider"),
            patch("inputs.plugins.booster_odom.IOProvider") as mock_io_provider_class,
        ):
            mock_io_provider = MagicMock()
            mock_io_provider_class.return_value = mock_io_provider

            config = BoosterOdomConfig()
            sensor = BoosterOdom(config=config)

            sensor.messages.append(Message(timestamp=1234.0, message="Test message"))
            assert len(sensor.messages) == 1

            sensor.formatted_latest_buffer()

            assert len(sensor.messages) == 0

    def test_formatted_latest_buffer_includes_descriptor(self):
        """Test formatted_latest_buffer includes descriptor for LLM."""
        with (
            patch("inputs.plugins.booster_odom.K1OdomProvider"),
            patch("inputs.plugins.booster_odom.IOProvider") as mock_io_provider_class,
        ):
            mock_io_provider = MagicMock()
            mock_io_provider_class.return_value = mock_io_provider

            config = BoosterOdomConfig()
            sensor = BoosterOdom(config=config)

            sensor.messages.append(Message(timestamp=1234.0, message="Test"))

            result = sensor.formatted_latest_buffer()

            assert result is not None
            assert sensor.descriptor_for_LLM in result


class TestStop:
    """Test stop method."""

    def test_stop_calls_odom_stop(self):
        """Test stop calls the odom provider's stop method."""
        with (
            patch("inputs.plugins.booster_odom.K1OdomProvider") as mock_provider_class,
            patch("inputs.plugins.booster_odom.IOProvider"),
        ):
            mock_provider = MagicMock()
            mock_provider_class.return_value = mock_provider

            config = BoosterOdomConfig()
            sensor = BoosterOdom(config=config)

            sensor.stop()

            mock_provider.stop.assert_called_once()

    def test_stop_handles_none_odom(self):
        """Test stop handles None odom gracefully."""
        with (
            patch("inputs.plugins.booster_odom.K1OdomProvider"),
            patch("inputs.plugins.booster_odom.IOProvider"),
        ):
            config = BoosterOdomConfig()
            sensor = BoosterOdom(config=config)

            sensor.odom = None  # type: ignore

            # Should not raise exception
            sensor.stop()


class TestIntegration:
    """Integration tests for BoosterOdom."""

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test complete workflow from poll to formatted output."""
        with (
            patch("inputs.plugins.booster_odom.K1OdomProvider") as mock_provider_class,
            patch("inputs.plugins.booster_odom.IOProvider") as mock_io_provider_class,
        ):
            mock_provider = MagicMock()
            mock_provider.position = {"moving": False, "body_attitude": RobotState.STANDING}
            mock_provider_class.return_value = mock_provider

            mock_io_provider = MagicMock()
            mock_io_provider_class.return_value = mock_io_provider

            config = BoosterOdomConfig()
            sensor = BoosterOdom(config=config)

            # Poll for data
            with patch("inputs.plugins.booster_odom.asyncio.sleep", new=AsyncMock()):
                raw_data = await sensor._poll()

            # Convert to text
            with patch("inputs.plugins.booster_odom.time.time", return_value=1234.0):
                await sensor.raw_to_text(raw_data)

            # Format output
            result = sensor.formatted_latest_buffer()

            assert result is not None
            assert "standing still" in result.lower()
            assert "can move" in result.lower()
            mock_io_provider.add_input.assert_called_once()

    @pytest.mark.asyncio
    async def test_multiple_state_changes(self):
        """Test handling multiple state changes."""
        with (
            patch("inputs.plugins.booster_odom.K1OdomProvider") as mock_provider_class,
            patch("inputs.plugins.booster_odom.IOProvider") as mock_io_provider_class,
        ):
            mock_provider = MagicMock()
            mock_provider_class.return_value = mock_provider

            mock_io_provider = MagicMock()
            mock_io_provider_class.return_value = mock_io_provider

            config = BoosterOdomConfig()
            sensor = BoosterOdom(config=config)

            # Standing still
            mock_provider.position = {"moving": False, "body_attitude": RobotState.STANDING}
            with (
                patch("inputs.plugins.booster_odom.asyncio.sleep", new=AsyncMock()),
                patch("inputs.plugins.booster_odom.time.time", return_value=1234.0),
            ):
                raw_data = await sensor._poll()
                await sensor.raw_to_text(raw_data)

            # Moving
            mock_provider.position = {"moving": True, "body_attitude": RobotState.STANDING}
            with (
                patch("inputs.plugins.booster_odom.asyncio.sleep", new=AsyncMock()),
                patch("inputs.plugins.booster_odom.time.time", return_value=1235.0),
            ):
                raw_data = await sensor._poll()
                await sensor.raw_to_text(raw_data)

            # Sitting
            mock_provider.position = {"moving": False, "body_attitude": RobotState.SITTING}
            with (
                patch("inputs.plugins.booster_odom.asyncio.sleep", new=AsyncMock()),
                patch("inputs.plugins.booster_odom.time.time", return_value=1236.0),
            ):
                raw_data = await sensor._poll()
                await sensor.raw_to_text(raw_data)

            assert len(sensor.messages) == 3
