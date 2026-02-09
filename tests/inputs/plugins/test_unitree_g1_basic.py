from unittest.mock import AsyncMock, patch

import pytest

from inputs.plugins.unitree_g1_basic import UnitreeG1Basic, UnitreeG1BasicConfig


def test_initialization():
    """Test basic initialization."""
    with (
        patch("inputs.plugins.unitree_g1_basic.ChannelSubscriber"),
        patch("inputs.plugins.unitree_g1_basic.IOProvider"),
        patch("inputs.plugins.unitree_g1_basic.TeleopsStatusProvider"),
    ):
        config = UnitreeG1BasicConfig()
        sensor = UnitreeG1Basic(config=config)

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


@pytest.mark.asyncio
async def test_poll():
    """Test _poll method."""
    with (
        patch("inputs.plugins.unitree_g1_basic.ChannelSubscriber"),
        patch("inputs.plugins.unitree_g1_basic.IOProvider"),
        patch("inputs.plugins.unitree_g1_basic.TeleopsStatusProvider"),
    ):
        config = UnitreeG1BasicConfig()
        sensor = UnitreeG1Basic(config=config)
        sensor.battery_percentage = 75.0
        sensor.battery_voltage = 48.5
        sensor.battery_amperes = 3.2

        with patch("inputs.plugins.unitree_g1_basic.asyncio.sleep", new=AsyncMock()):
            result = await sensor._poll()

        assert result is not None
        assert len(result) == 3
        assert result[0] == 75.0
        assert result[1] == 48.5
        assert result[2] == 3.2


@pytest.mark.asyncio
async def test_raw_to_text_with_low_battery():
    """Test _raw_to_text with low battery (warning level)."""
    with (
        patch("inputs.plugins.unitree_g1_basic.ChannelSubscriber"),
        patch("inputs.plugins.unitree_g1_basic.IOProvider"),
        patch("inputs.plugins.unitree_g1_basic.TeleopsStatusProvider"),
    ):
        config = UnitreeG1BasicConfig()
        sensor = UnitreeG1Basic(config=config)

        with patch("inputs.plugins.unitree_g1_basic.time.time", return_value=1234.0):
            result = await sensor._raw_to_text([10.0, 48.0, 3.0])

        assert result is not None
        assert result.timestamp == 1234.0
        assert "WARNING" in result.message or "energy" in result.message.lower()


def test_formatted_latest_buffer():
    """Test formatted_latest_buffer."""
    with (
        patch("inputs.plugins.unitree_g1_basic.ChannelSubscriber"),
        patch("inputs.plugins.unitree_g1_basic.IOProvider"),
        patch("inputs.plugins.unitree_g1_basic.TeleopsStatusProvider"),
    ):
        config = UnitreeG1BasicConfig()
        sensor = UnitreeG1Basic(config=config)

        result = sensor.formatted_latest_buffer()
        assert result is None
