from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inputs.base import Message, SensorConfig
from inputs.plugins.amcl_localization_input import AMCLLocalizationInput


def test_initialization():
    """Test basic initialization."""
    with (
        patch("inputs.plugins.amcl_localization_input.UnitreeGo2AMCLProvider"),
        patch("inputs.plugins.amcl_localization_input.IOProvider"),
    ):
        sensor = AMCLLocalizationInput(config=SensorConfig())

        assert sensor.messages == []
        assert "localization" in sensor.descriptor_for_LLM.lower()


@pytest.mark.asyncio
async def test_poll_when_localized():
    """Test _poll when robot is localized."""
    with (
        patch(
            "inputs.plugins.amcl_localization_input.UnitreeGo2AMCLProvider"
        ) as mock_provider_class,
        patch("inputs.plugins.amcl_localization_input.IOProvider"),
    ):
        mock_provider = MagicMock()
        mock_provider.is_localized = True

        mock_position = MagicMock()
        mock_position.x = 1.0
        mock_position.y = 2.0
        mock_position.__format__ = lambda spec: str(mock_position.x)

        mock_pose = MagicMock()
        mock_pose.position = mock_position
        mock_provider.pose = mock_pose

        mock_provider_class.return_value = mock_provider

        sensor = AMCLLocalizationInput(config=SensorConfig())

        with patch(
            "inputs.plugins.amcl_localization_input.asyncio.sleep", new=AsyncMock()
        ):
            result = await sensor._poll()

        assert result is not None
        assert "LOCALIZED" in result or "position" in result.lower()


@pytest.mark.asyncio
async def test_poll_when_not_localized():
    """Test _poll when robot is not localized."""
    with (
        patch(
            "inputs.plugins.amcl_localization_input.UnitreeGo2AMCLProvider"
        ) as mock_provider_class,
        patch("inputs.plugins.amcl_localization_input.IOProvider"),
    ):
        mock_provider = MagicMock()
        mock_provider.is_localized = False
        mock_provider.pose = None

        mock_provider_class.return_value = mock_provider

        sensor = AMCLLocalizationInput(config=SensorConfig())

        with patch(
            "inputs.plugins.amcl_localization_input.asyncio.sleep", new=AsyncMock()
        ):
            result = await sensor._poll()

        assert result is not None
        assert "NOT LOCALIZED" in result


@pytest.mark.asyncio
async def test_poll_with_exception():
    """Test _poll handles exceptions gracefully."""
    with (
        patch(
            "inputs.plugins.amcl_localization_input.UnitreeGo2AMCLProvider"
        ) as mock_provider_class,
        patch("inputs.plugins.amcl_localization_input.IOProvider"),
    ):
        mock_provider = MagicMock()
        mock_provider.is_localized = MagicMock(side_effect=Exception("Test error"))

        mock_provider_class.return_value = mock_provider

        sensor = AMCLLocalizationInput(config=SensorConfig())

        with patch(
            "inputs.plugins.amcl_localization_input.asyncio.sleep", new=AsyncMock()
        ):
            result = await sensor._poll()

        assert result is not None
        assert isinstance(result, str)


@pytest.mark.asyncio
async def test_raw_to_text_with_valid_input():
    """Test _raw_to_text with valid input."""
    with (
        patch("inputs.plugins.amcl_localization_input.UnitreeGo2AMCLProvider"),
        patch("inputs.plugins.amcl_localization_input.IOProvider"),
    ):
        sensor = AMCLLocalizationInput(config=SensorConfig())

        with patch(
            "inputs.plugins.amcl_localization_input.time.time", return_value=1234.0
        ):
            result = await sensor._raw_to_text("LOCALIZED: Robot position confirmed")

        assert result is not None
        assert isinstance(result, Message)
        assert result.timestamp == 1234.0
        sensor = AMCLLocalizationInput(config=SensorConfig())


@pytest.mark.asyncio
async def test_raw_to_text_with_none():
    """Test _raw_to_text with None input."""
    with (
        patch("inputs.plugins.amcl_localization_input.UnitreeGo2AMCLProvider"),
        patch("inputs.plugins.amcl_localization_input.IOProvider"),
    ):
        sensor = AMCLLocalizationInput(config=SensorConfig())
        result = await sensor._raw_to_text(None)

        assert result is None


def test_formatted_latest_buffer_with_messages():
    """Test formatted_latest_buffer with messages in buffer."""
    with (
        patch("inputs.plugins.amcl_localization_input.UnitreeGo2AMCLProvider"),
        patch("inputs.plugins.amcl_localization_input.IOProvider"),
    ):
        sensor = AMCLLocalizationInput(config=SensorConfig())
        sensor.io_provider = MagicMock()

        sensor.messages = [
            Message(timestamp=1000.0, message="Status 1"),
            Message(timestamp=1001.0, message="Status 2"),
        ]

        result = sensor.formatted_latest_buffer()

        assert result is not None
        assert "Status 2" in result


def test_formatted_latest_buffer_empty():
    """Test formatted_latest_buffer with empty buffer."""
    with (
        patch("inputs.plugins.amcl_localization_input.UnitreeGo2AMCLProvider"),
        patch("inputs.plugins.amcl_localization_input.IOProvider"),
    ):
        sensor = AMCLLocalizationInput(config=SensorConfig())
        sensor.io_provider = MagicMock()
        sensor.messages = []
        result = sensor.formatted_latest_buffer()
        assert result is None
