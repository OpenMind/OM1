"""Tests for SimplePaths input plugin."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inputs.base import Message, SensorConfig
from inputs.plugins.simple_paths import SimplePaths


def test_initialization():
    """Test basic initialization."""
    with (
        patch("inputs.plugins.simple_paths.SimplePathsProvider") as mock_provider,
        patch("inputs.plugins.simple_paths.IOProvider"),
    ):
        mock_provider_instance = MagicMock()
        mock_provider.return_value = mock_provider_instance

        config = SensorConfig()
        sensor = SimplePaths(config=config)

        assert sensor.messages == []
        assert sensor.io_provider is not None
        assert sensor.paths_provider is not None
        assert (
            "Information about objects and walls"
            in sensor.descriptor_for_LLM
        )
        mock_provider_instance.start.assert_called_once()


@pytest.mark.asyncio
async def test_poll_with_data():
    """Test _poll when SimplePaths data is available."""
    with (
        patch("inputs.plugins.simple_paths.SimplePathsProvider") as mock_provider,
        patch("inputs.plugins.simple_paths.IOProvider"),
        patch("inputs.plugins.simple_paths.asyncio.sleep", new=AsyncMock()),
    ):
        mock_provider_instance = MagicMock()
        mock_provider_instance.lidar_string = "Path data available"
        mock_provider.return_value = mock_provider_instance

        config = SensorConfig()
        sensor = SimplePaths(config=config)

        result = await sensor._poll()

        assert result == "Path data available"


@pytest.mark.asyncio
async def test_poll_with_no_data():
    """Test _poll when no SimplePaths data is available."""
    with (
        patch("inputs.plugins.simple_paths.SimplePathsProvider") as mock_provider,
        patch("inputs.plugins.simple_paths.IOProvider"),
        patch("inputs.plugins.simple_paths.asyncio.sleep", new=AsyncMock()),
    ):
        from queue import Empty

        mock_provider_instance = MagicMock()
        mock_provider_instance.lidar_string = None
        mock_provider.return_value = mock_provider_instance

        config = SensorConfig()
        sensor = SimplePaths(config=config)

        # Simulate Empty exception
        with patch.object(sensor.paths_provider, "lidar_string", side_effect=Empty()):
            result = await sensor._poll()

            assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_with_valid_input():
    """Test _raw_to_text with valid input."""
    with (
        patch("inputs.plugins.simple_paths.SimplePathsProvider") as mock_provider,
        patch("inputs.plugins.simple_paths.IOProvider"),
        patch("inputs.plugins.simple_paths.time.time", return_value=1234.0),
    ):
        mock_provider_instance = MagicMock()
        mock_provider.return_value = mock_provider_instance

        config = SensorConfig()
        sensor = SimplePaths(config=config)

        result = await sensor._raw_to_text("Path information")

        assert result is not None
        assert result.timestamp == 1234.0
        assert result.message == "Path information"


@pytest.mark.asyncio
async def test_raw_to_text_with_none():
    """Test _raw_to_text with None input."""
    with (
        patch("inputs.plugins.simple_paths.SimplePathsProvider") as mock_provider,
        patch("inputs.plugins.simple_paths.IOProvider"),
    ):
        mock_provider_instance = MagicMock()
        mock_provider.return_value = mock_provider_instance

        config = SensorConfig()
        sensor = SimplePaths(config=config)

        result = await sensor._raw_to_text(None)

        assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_updates_buffer():
    """Test that raw_to_text updates the message buffer."""
    with (
        patch("inputs.plugins.simple_paths.SimplePathsProvider") as mock_provider,
        patch("inputs.plugins.simple_paths.IOProvider"),
    ):
        mock_provider_instance = MagicMock()
        mock_provider.return_value = mock_provider_instance

        config = SensorConfig()
        sensor = SimplePaths(config=config)

        assert len(sensor.messages) == 0

        await sensor.raw_to_text("Path data")

        assert len(sensor.messages) == 1
        assert sensor.messages[0].message == "Path data"


@pytest.mark.asyncio
async def test_raw_to_text_with_none_does_not_update_buffer():
    """Test that raw_to_text doesn't update buffer when input is None."""
    with (
        patch("inputs.plugins.simple_paths.SimplePathsProvider") as mock_provider,
        patch("inputs.plugins.simple_paths.IOProvider"),
    ):
        mock_provider_instance = MagicMock()
        mock_provider.return_value = mock_provider_instance

        config = SensorConfig()
        sensor = SimplePaths(config=config)

        assert len(sensor.messages) == 0

        await sensor.raw_to_text(None)

        assert len(sensor.messages) == 0


def test_formatted_latest_buffer_empty():
    """Test formatted_latest_buffer returns None when buffer is empty."""
    with (
        patch("inputs.plugins.simple_paths.SimplePathsProvider") as mock_provider,
        patch("inputs.plugins.simple_paths.IOProvider"),
    ):
        mock_provider_instance = MagicMock()
        mock_provider.return_value = mock_provider_instance

        config = SensorConfig()
        sensor = SimplePaths(config=config)

        result = sensor.formatted_latest_buffer()

        assert result is None


def test_formatted_latest_buffer_with_message():
    """Test formatted_latest_buffer formats message correctly."""
    with (
        patch("inputs.plugins.simple_paths.SimplePathsProvider") as mock_provider,
        patch("inputs.plugins.simple_paths.IOProvider") as mock_io_provider,
    ):
        mock_provider_instance = MagicMock()
        mock_provider.return_value = mock_provider_instance
        mock_io_instance = MagicMock()
        mock_io_provider.return_value = mock_io_instance

        config = SensorConfig()
        sensor = SimplePaths(config=config)
        sensor.io_provider = mock_io_instance

        message1 = Message(timestamp=time.time(), message="Message 1")
        message2 = Message(timestamp=time.time(), message="Message 2")
        sensor.messages.extend([message1, message2])

        result = sensor.formatted_latest_buffer()

        assert result is not None
        assert "Message 2" in result  # Should contain latest message
        assert "// START" in result
        assert "// END" in result
        mock_io_instance.add_input.assert_called_once()
        assert len(sensor.messages) == 0
