import sys
from unittest.mock import AsyncMock, Mock, patch

import pytest

if "zenoh" not in sys.modules:
    sys.modules["zenoh"] = Mock()
if "zenoh.Session" not in sys.modules:
    sys.modules["zenoh.Session"] = Mock()
if "zenoh.Subscriber" not in sys.modules:
    sys.modules["zenoh.Subscriber"] = Mock()
if "zenoh_msgs" not in sys.modules:
    sys.modules["zenoh_msgs"] = Mock()
    sys.modules["zenoh_msgs.open_zenoh_session"] = Mock()
    sys.modules["zenoh_msgs.sensor_msgs"] = Mock()
    sys.modules["zenoh_msgs.sensor_msgs.BatteryState"] = Mock()
    sys.modules["zenoh_msgs.sensor_msgs.DockStatus"] = Mock()

from inputs.plugins.battery_turtlebot4 import (
    Message,
    TurtleBot4Battery,
    TurtleBot4BatteryConfig,
)


@pytest.fixture
def mock_zenoh_session():
    with patch("inputs.plugins.battery_turtlebot4.open_zenoh_session") as mock:
        mock_session = Mock()
        mock.return_value = mock_session
        yield mock_session


@pytest.fixture
def mock_io_provider():
    with patch("inputs.plugins.battery_turtlebot4.IOProvider") as mock_class:
        mock_instance = Mock()
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_teleops_provider():
    with patch("inputs.plugins.battery_turtlebot4.TeleopsStatusProvider") as mock_class:
        mock_instance = Mock()
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_battery_status():
    with patch("inputs.plugins.battery_turtlebot4.BatteryStatus") as mock_class:
        mock_instance = Mock()
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_teleops_status():
    with patch("inputs.plugins.battery_turtlebot4.TeleopsStatus") as mock_class:
        mock_instance = Mock()
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_sensor_msgs():
    with (
        patch(
            "inputs.plugins.battery_turtlebot4.sensor_msgs.BatteryState.deserialize"
        ) as mock_batt_des,
        patch(
            "inputs.plugins.battery_turtlebot4.sensor_msgs.DockStatus.deserialize"
        ) as mock_dock_des,
    ):
        yield {"battery": mock_batt_des, "dock": mock_dock_des}


@pytest.fixture
def turtlebot4_battery_instance(
    mock_zenoh_session,
    mock_io_provider,
    mock_teleops_provider,
    mock_battery_status,
    mock_teleops_status,
    mock_sensor_msgs,
):
    config = TurtleBot4BatteryConfig(URID="test_urid")
    instance = TurtleBot4Battery(config=config)
    instance.z = mock_zenoh_session
    instance.io_provider = mock_io_provider
    instance.status_provider = mock_teleops_provider
    return instance


def test_initialization_creates_providers_and_subscribers(
    turtlebot4_battery_instance, mock_zenoh_session
):
    assert turtlebot4_battery_instance.io_provider is not None
    assert turtlebot4_battery_instance.status_provider is not None
    assert hasattr(turtlebot4_battery_instance, "messages")
    assert isinstance(turtlebot4_battery_instance.messages, list)
    mock_zenoh_session.declare_subscriber.assert_any_call(
        "test_urid/c3/battery_state", turtlebot4_battery_instance.listener_battery
    )
    mock_zenoh_session.declare_subscriber.assert_any_call(
        "test_urid/c3/dock_status", turtlebot4_battery_instance.listener_dock
    )


def test_listener_battery_updates_state_normal(
    turtlebot4_battery_instance, mock_sensor_msgs
):
    mock_batt_data = Mock()
    mock_batt_data.percentage = 0.75
    mock_batt_data.voltage = 12.6
    mock_batt_data.temperature = 25.5
    mock_batt_data.header.stamp.sec = 1234567890

    mock_sensor_msgs["battery"].return_value = mock_batt_data

    sample = Mock()
    sample.payload.to_bytes.return_value = b"mock_payload"

    with patch("inputs.plugins.battery_turtlebot4.round", side_effect=round):
        turtlebot4_battery_instance.listener_battery(sample)

    assert turtlebot4_battery_instance.battery_percentage == 75
    assert turtlebot4_battery_instance.battery_voltage == 12.6
    assert turtlebot4_battery_instance.battery_temperature == 25.5
    assert turtlebot4_battery_instance.battery_timestamp == 1234567890
    assert turtlebot4_battery_instance.battery_status is None


def test_listener_battery_updates_state_low(
    turtlebot4_battery_instance, mock_sensor_msgs
):
    mock_batt_data = Mock()
    mock_batt_data.percentage = 0.10
    mock_batt_data.voltage = 11.0
    mock_batt_data.temperature = 20.0
    mock_batt_data.header.stamp.sec = 1234567891

    mock_sensor_msgs["battery"].return_value = mock_batt_data

    sample = Mock()
    sample.payload.to_bytes.return_value = b"mock_payload"

    with patch("inputs.plugins.battery_turtlebot4.round", side_effect=round):
        turtlebot4_battery_instance.listener_battery(sample)

    assert turtlebot4_battery_instance.battery_percentage == 10
    assert turtlebot4_battery_instance.battery_voltage == 11.0
    assert turtlebot4_battery_instance.battery_temperature == 20.0
    assert turtlebot4_battery_instance.battery_timestamp == 1234567891
    assert "IMPORTANT" in turtlebot4_battery_instance.battery_status


def test_listener_battery_critical_status(
    turtlebot4_battery_instance, mock_sensor_msgs
):
    mock_batt_data = Mock()
    mock_batt_data.percentage = 0.03
    mock_batt_data.voltage = 10.0
    mock_batt_data.temperature = 15.0
    mock_batt_data.header.stamp.sec = 1234567892

    mock_sensor_msgs["battery"].return_value = mock_batt_data

    sample = Mock()
    sample.payload.to_bytes.return_value = b"mock_payload"

    with patch("inputs.plugins.battery_turtlebot4.round", side_effect=round):
        turtlebot4_battery_instance.listener_battery(sample)

    assert "CRITICAL" in turtlebot4_battery_instance.battery_status


def test_listener_dock_updates_is_docked(turtlebot4_battery_instance, mock_sensor_msgs):
    mock_dock_data = Mock()
    mock_dock_data.is_docked = True

    mock_sensor_msgs["dock"].return_value = mock_dock_data

    sample = Mock()
    sample.payload.to_bytes.return_value = b"mock_payload"

    turtlebot4_battery_instance.listener_dock(sample)

    assert turtlebot4_battery_instance.is_docked is True


@pytest.mark.asyncio
async def test_report_status_calls_share_status(
    turtlebot4_battery_instance,
    mock_teleops_provider,
    mock_teleops_status,
    mock_battery_status,
):
    mock_ts_instance = Mock()
    mock_bs_instance = Mock()
    mock_teleops_status.side_effect = lambda **kwargs: mock_ts_instance
    mock_battery_status.side_effect = lambda **kwargs: mock_bs_instance

    turtlebot4_battery_instance.battery_percentage = 80.0
    turtlebot4_battery_instance.battery_temperature = 30
    turtlebot4_battery_instance.battery_voltage = 13.2
    turtlebot4_battery_instance.battery_timestamp = 1234567890
    turtlebot4_battery_instance.is_docked = True

    with patch("time.time", return_value=1234.0):
        await turtlebot4_battery_instance.report_status()

    mock_teleops_status.assert_called_once()
    mock_battery_status.assert_called_once()

    mock_teleops_provider.share_status.assert_called_once_with(mock_ts_instance)

    ts_call_kwargs = mock_teleops_status.call_args[1]
    bs_call_kwargs = mock_battery_status.call_args[1]

    assert ts_call_kwargs["machine_name"] == "TurtleBot4"
    assert bs_call_kwargs["battery_level"] == 80.0
    assert bs_call_kwargs["temperature"] == 30
    assert bs_call_kwargs["voltage"] == 13.2
    assert bs_call_kwargs["timestamp"] == "1234567890"
    assert bs_call_kwargs["charging_status"]


@pytest.mark.asyncio
async def test_poll_calls_report_status_and_returns_correct_list(
    turtlebot4_battery_instance,
):
    turtlebot4_battery_instance.battery_status = "IMPORTANT: Low battery."

    with patch.object(
        turtlebot4_battery_instance, "report_status", new=AsyncMock()
    ) as mock_report:
        result = await turtlebot4_battery_instance._poll()

    mock_report.assert_awaited_once()
    assert result == ["IMPORTANT: Low battery."]


@pytest.mark.asyncio
async def test_poll_returns_empty_list_if_no_status(
    turtlebot4_battery_instance,
):
    turtlebot4_battery_instance.battery_status = None

    with patch.object(
        turtlebot4_battery_instance, "report_status", new=AsyncMock()
    ) as mock_report:
        result = await turtlebot4_battery_instance._poll()

    mock_report.assert_awaited_once()
    assert result == []


@pytest.mark.asyncio
async def test_raw_to_text_none_if_empty_list(turtlebot4_battery_instance):
    result = await turtlebot4_battery_instance._raw_to_text([])
    assert result is None

    result = await turtlebot4_battery_instance._raw_to_text([""])
    assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_with_message(turtlebot4_battery_instance):
    with patch("time.time", return_value=1234.0):
        result = await turtlebot4_battery_instance._raw_to_text(
            ["test battery message"]
        )

    assert isinstance(result, Message)
    assert result.timestamp == 1234.0
    assert result.message == "test battery message"


@pytest.mark.asyncio
async def test_raw_to_text_adds_to_buffer(turtlebot4_battery_instance):
    with patch("time.time", return_value=1234.0):
        await turtlebot4_battery_instance.raw_to_text(["test message"])

    assert len(turtlebot4_battery_instance.messages) == 1
    assert turtlebot4_battery_instance.messages[0].message == "test message"


@pytest.mark.asyncio
async def test_raw_to_text_none_does_not_add_to_buffer(turtlebot4_battery_instance):
    initial_len = len(turtlebot4_battery_instance.messages)
    await turtlebot4_battery_instance.raw_to_text([])

    assert len(turtlebot4_battery_instance.messages) == initial_len


def test_formatted_latest_buffer_empty(turtlebot4_battery_instance):
    result = turtlebot4_battery_instance.formatted_latest_buffer()
    assert result is None


def test_formatted_latest_buffer_with_message(turtlebot4_battery_instance):
    msg = Message(timestamp=1234.0, message="buffered message")
    turtlebot4_battery_instance.messages = [msg]

    result = turtlebot4_battery_instance.formatted_latest_buffer()

    assert "INPUT:" in result
    assert "buffered message" in result
    assert len(turtlebot4_battery_instance.messages) == 0
    turtlebot4_battery_instance.io_provider.add_input.assert_called_once()
