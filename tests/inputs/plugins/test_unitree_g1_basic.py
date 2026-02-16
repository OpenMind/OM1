import sys
from unittest.mock import AsyncMock, MagicMock, patch

# Mock unitree modules before imports
sys.modules["unitree"] = MagicMock()
sys.modules["unitree.unitree_sdk2py"] = MagicMock()
sys.modules["unitree.unitree_sdk2py.core"] = MagicMock()
sys.modules["unitree.unitree_sdk2py.core.channel"] = MagicMock()
sys.modules["unitree.unitree_sdk2py.idl"] = MagicMock()
sys.modules["unitree.unitree_sdk2py.idl.unitree_hg"] = MagicMock()
sys.modules["unitree.unitree_sdk2py.idl.unitree_hg.msg"] = MagicMock()


import pytest  # noqa: E402

from inputs.plugins.unitree_g1_basic import (  # noqa: E402
    UnitreeG1Basic,
    UnitreeG1BasicConfig,
)


def test_initialization():
    """Test basic initialization."""
    with (
        patch("inputs.plugins.unitree_g1_basic.ChannelSubscriber", create=True),
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
        patch("inputs.plugins.unitree_g1_basic.ChannelSubscriber", create=True),
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
        patch("inputs.plugins.unitree_g1_basic.ChannelSubscriber", create=True),
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
        patch("inputs.plugins.unitree_g1_basic.ChannelSubscriber", create=True),
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
        patch("inputs.plugins.unitree_g1_basic.ChannelSubscriber", create=True),
        patch("inputs.plugins.unitree_g1_basic.IOProvider"),
        patch("inputs.plugins.unitree_g1_basic.TeleopsStatusProvider"),
    ):
        config = UnitreeG1BasicConfig()
        sensor = UnitreeG1Basic(config=config)

        result = sensor.formatted_latest_buffer()
        assert result is None


def test_bms_state_handler():
    """Test BMSStateHandler updates battery state (lines 153-159)."""
    with (
        patch("inputs.plugins.unitree_g1_basic.ChannelSubscriber", create=True),
        patch("inputs.plugins.unitree_g1_basic.IOProvider"),
        patch("inputs.plugins.unitree_g1_basic.TeleopsStatusProvider"),
    ):
        config = UnitreeG1BasicConfig()
        g1 = UnitreeG1Basic(config)

        # Create mock BmsState message
        mock_bms = MagicMock()
        mock_bms.bmsvoltage = [48.5]
        mock_bms.current = 2.3
        mock_bms.soc = 85.0
        mock_bms.temperature = [35.5]

        g1.BMSStateHandler(mock_bms)

        assert g1.battery_voltage == 48.5
        assert g1.battery_amperes == 2.3
        assert g1.battery_percentage == 85.0
        assert g1.battery_temperature == 35.5


def test_low_state_handler():
    """Test LowStateHandler updates teleops state."""
    with (
        patch("inputs.plugins.unitree_g1_basic.ChannelSubscriber", create=True),
        patch("inputs.plugins.unitree_g1_basic.IOProvider"),
        patch("inputs.plugins.unitree_g1_basic.TeleopsStatusProvider"),
    ):
        config = UnitreeG1BasicConfig()
        g1 = UnitreeG1Basic(config)

        # Create mock LowState message
        mock_low = MagicMock()
        mock_low.imu_state = MagicMock()
        mock_low.imu_state.rpy = [0.1, 0.2, 0.3]

        g1.LowStateHandler(mock_low)

        assert g1.low_state == mock_low


@pytest.mark.asyncio
async def test_raw_to_text_critical_battery():
    """Test _raw_to_text with critical battery level."""
    with (
        patch("inputs.plugins.unitree_g1_basic.ChannelSubscriber", create=True),
        patch("inputs.plugins.unitree_g1_basic.IOProvider"),
        patch("inputs.plugins.unitree_g1_basic.TeleopsStatusProvider"),
    ):
        config = UnitreeG1BasicConfig()
        g1 = UnitreeG1Basic(config)

        # Set critical battery
        g1.battery_percentage = 2.0
        g1.battery_amperes = 1.0
        g1.battery_voltage = 40.0
        g1.battery_temperature = 30.0

        result = await g1._raw_to_text([0.1, 0.2, 0.3])

        assert result is not None
        assert "CRITICAL" in result.message or "WARNING" in result.message


def test_formatted_latest_buffer_no_messages():
    """Test formatted_latest_buffer with no messages."""
    with (
        patch("inputs.plugins.unitree_g1_basic.ChannelSubscriber", create=True),
        patch("inputs.plugins.unitree_g1_basic.IOProvider"),
        patch("inputs.plugins.unitree_g1_basic.TeleopsStatusProvider"),
    ):
        config = UnitreeG1BasicConfig()
        g1 = UnitreeG1Basic(config)

        result = g1.formatted_latest_buffer()
        assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_warning_battery():
    """Test _raw_to_text with warning battery level (lines 229-233)."""
    with (
        patch("inputs.plugins.unitree_g1_basic.ChannelSubscriber", create=True),
        patch("inputs.plugins.unitree_g1_basic.IOProvider"),
        patch("inputs.plugins.unitree_g1_basic.TeleopsStatusProvider"),
    ):
        config = UnitreeG1BasicConfig()
        g1 = UnitreeG1Basic(config)

        # Set warning battery (between 20% and 30%)
        g1.battery_percentage = 25.0
        g1.battery_amperes = 1.5
        g1.battery_voltage = 45.0
        g1.battery_temperature = 32.0

        result = await g1._raw_to_text([25.0, 45.0, 1.5])

        assert result is not None
        assert "Consider sitting down" in result.message


@pytest.mark.asyncio
async def test_raw_to_text_appends_message():
    """Test raw_to_text appends to messages (lines 244-247)."""
    with (
        patch("inputs.plugins.unitree_g1_basic.ChannelSubscriber", create=True),
        patch("inputs.plugins.unitree_g1_basic.IOProvider"),
        patch("inputs.plugins.unitree_g1_basic.TeleopsStatusProvider"),
    ):
        config = UnitreeG1BasicConfig()
        g1 = UnitreeG1Basic(config)

        # Set low battery to trigger message
        g1.battery_percentage = 15.0
        initial_count = len(g1.messages)

        await g1.raw_to_text([15.0, 45.0, 1.5])

        assert len(g1.messages) == initial_count + 1


def test_formatted_latest_buffer_with_content():
    """Test formatted_latest_buffer returns formatted string (lines 264-278)."""
    with (
        patch("inputs.plugins.unitree_g1_basic.ChannelSubscriber", create=True),
        patch("inputs.plugins.unitree_g1_basic.IOProvider"),
        patch("inputs.plugins.unitree_g1_basic.TeleopsStatusProvider"),
    ):
        config = UnitreeG1BasicConfig()
        g1 = UnitreeG1Basic(config)

        # Add a message
        import time

        from inputs.base import Message

        msg = Message(timestamp=time.time(), message="Test warning message")
        g1.messages.append(msg)

        result = g1.formatted_latest_buffer()

        assert result is not None
        assert "INPUT: Energy Level" in result
        assert "// START" in result
        assert "Test warning message" in result
        assert "// END" in result
        assert len(g1.messages) == 0  # Should be cleared
