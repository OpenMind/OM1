from unittest.mock import Mock, patch

import pytest

from inputs.plugins.charging_status_unitree_go2 import ChargingStatusUnitreeGo2


@pytest.fixture
def mock_charging_provider():
    """Mock UnitreeGo2ChargingProvider."""
    with patch(
        "inputs.plugins.charging_status_unitree_go2.UnitreeGo2ChargingProvider"
    ) as mock:
        mock_instance = Mock()
        mock_instance.charging_status = None
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_io_provider():
    """Mock IOProvider."""
    with patch("inputs.plugins.charging_status_unitree_go2.IOProvider") as mock:
        yield mock


def test_initialization(mock_charging_provider, mock_io_provider):
    """Test ChargingStatusUnitreeGo2 initialization."""
    plugin = ChargingStatusUnitreeGo2()

    assert plugin.messages == []
    assert plugin.descriptor_for_LLM == (
        "Robot charging status - indicates the current battery charging state."
    )
    mock_charging_provider.start.assert_called_once()


@pytest.mark.asyncio
async def test_poll_discharging_status(mock_charging_provider, mock_io_provider):
    """Test polling returns correct message for DISCHARGING status (0)."""
    mock_charging_provider.charging_status = 0
    plugin = ChargingStatusUnitreeGo2()

    result = await plugin._poll()

    assert result == "DISCHARGING: Robot is running on battery power."


@pytest.mark.asyncio
async def test_poll_charging_status(mock_charging_provider, mock_io_provider):
    """Test polling returns correct message for CHARGING status (1)."""
    mock_charging_provider.charging_status = 1
    plugin = ChargingStatusUnitreeGo2()

    result = await plugin._poll()

    assert result == "CHARGING: Robot is currently charging."


@pytest.mark.asyncio
async def test_poll_enroute_charging_status(mock_charging_provider, mock_io_provider):
    """Test polling returns correct message for ENROUTE_CHARGING status (2)."""
    mock_charging_provider.charging_status = 2
    plugin = ChargingStatusUnitreeGo2()

    result = await plugin._poll()

    assert result == "ENROUTE_CHARGING: Robot is moving to charging station."


@pytest.mark.asyncio
async def test_poll_fully_charged_status(mock_charging_provider, mock_io_provider):
    """Test polling returns correct message for FULLY_CHARGED status (3)."""
    mock_charging_provider.charging_status = 3
    plugin = ChargingStatusUnitreeGo2()

    result = await plugin._poll()

    assert result == "FULLY_CHARGED: Robot battery is fully charged."


@pytest.mark.asyncio
async def test_poll_unknown_status(mock_charging_provider, mock_io_provider):
    """Test polling handles unknown status codes."""
    mock_charging_provider.charging_status = 99
    plugin = ChargingStatusUnitreeGo2()

    result = await plugin._poll()

    assert result == "UNKNOWN: Unrecognized charging status (99)."


@pytest.mark.asyncio
async def test_poll_none_status(mock_charging_provider, mock_io_provider):
    """Test polling returns empty string when status is None."""
    mock_charging_provider.charging_status = None
    plugin = ChargingStatusUnitreeGo2()

    result = await plugin._poll()

    assert result == ""


@pytest.mark.asyncio
async def test_poll_exception_handling(mock_charging_provider, mock_io_provider):
    """Test error handling during polling."""
    mock_charging_provider.charging_status = None
    # Make the property access raise an exception
    type(mock_charging_provider).charging_status = property(
        lambda self: (_ for _ in ()).throw(RuntimeError("Test error"))
    )

    plugin = ChargingStatusUnitreeGo2()

    result = await plugin._poll()

    assert result == "CHARGING ERROR: Unable to determine charging state."


@pytest.mark.asyncio
async def test_raw_to_text(mock_charging_provider, mock_io_provider):
    """Test conversion of raw input to Message."""
    plugin = ChargingStatusUnitreeGo2()

    message = await plugin._raw_to_text("CHARGING: Robot is currently charging.")

    assert message is not None
    assert message.message == "CHARGING: Robot is currently charging."
    assert isinstance(message.timestamp, float)


@pytest.mark.asyncio
async def test_raw_to_text_with_none(mock_charging_provider, mock_io_provider):
    """Test handling of None input in raw_to_text."""
    plugin = ChargingStatusUnitreeGo2()

    await plugin.raw_to_text(None)

    assert plugin.messages == []


@pytest.mark.asyncio
async def test_raw_to_text_appends_to_buffer(mock_charging_provider, mock_io_provider):
    """Test that raw_to_text appends messages to buffer."""
    plugin = ChargingStatusUnitreeGo2()

    await plugin.raw_to_text("DISCHARGING: Robot is running on battery power.")

    assert len(plugin.messages) == 1
    assert (
        plugin.messages[0].message == "DISCHARGING: Robot is running on battery power."
    )


def test_formatted_latest_buffer_with_messages(
    mock_charging_provider, mock_io_provider
):
    """Test formatted_latest_buffer returns formatted message and clears buffer."""
    mock_io_instance = Mock()
    mock_io_provider.return_value = mock_io_instance

    plugin = ChargingStatusUnitreeGo2()

    # Simulate adding a message
    import time

    from inputs.base import Message

    test_message = Message(
        timestamp=time.time(), message="CHARGING: Robot is currently charging."
    )
    plugin.messages.append(test_message)

    result = plugin.formatted_latest_buffer()

    assert result is not None
    assert "INPUT: Robot charging status" in result
    assert "// START" in result
    assert "CHARGING: Robot is currently charging." in result
    assert "// END" in result

    # Verify buffer was cleared
    assert plugin.messages == []

    # Verify IO provider was called
    mock_io_instance.add_input.assert_called_once()


def test_formatted_latest_buffer_empty(mock_charging_provider, mock_io_provider):
    """Test formatted_latest_buffer returns None when buffer is empty."""
    plugin = ChargingStatusUnitreeGo2()

    result = plugin.formatted_latest_buffer()

    assert result is None


def test_descriptor_for_llm(mock_charging_provider, mock_io_provider):
    """Test that descriptor_for_LLM is set correctly."""
    plugin = ChargingStatusUnitreeGo2()

    assert "charging status" in plugin.descriptor_for_LLM.lower()
    assert "battery" in plugin.descriptor_for_LLM.lower()


@pytest.mark.asyncio
async def test_full_workflow(mock_charging_provider, mock_io_provider):
    """Test complete workflow: poll -> raw_to_text -> formatted_latest_buffer."""
    mock_charging_provider.charging_status = 1
    plugin = ChargingStatusUnitreeGo2()

    # Poll for status
    raw_status = await plugin._poll()
    assert raw_status == "CHARGING: Robot is currently charging."

    # Convert to text and add to buffer
    await plugin.raw_to_text(raw_status)
    assert len(plugin.messages) == 1

    # Format and retrieve
    formatted = plugin.formatted_latest_buffer()
    assert formatted is not None
    assert "CHARGING: Robot is currently charging." in formatted
    assert plugin.messages == []  # Buffer cleared
