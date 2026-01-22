import sys
from unittest.mock import AsyncMock, Mock, patch

import pytest

if "unitree" not in sys.modules:
    sys.modules["unitree"] = Mock()
if "unitree.unitree_sdk2py" not in sys.modules:
    sys.modules["unitree.unitree_sdk2py"] = Mock()
if "unitree.unitree_sdk2py.core" not in sys.modules:
    sys.modules["unitree.unitree_sdk2py.core"] = Mock()
if "unitree.unitree_sdk2py.core.channel" not in sys.modules:
    sys.modules["unitree.unitree_sdk2py.core.channel"] = Mock()
if "unitree.unitree_sdk2py.idl" not in sys.modules:
    sys.modules["unitree.unitree_sdk2py.idl"] = Mock()
if "unitree.unitree_sdk2py.idl.unitree_go" not in sys.modules:
    sys.modules["unitree.unitree_sdk2py.idl.unitree_go"] = Mock()
if "unitree.unitree_sdk2py.idl.unitree_go.msg" not in sys.modules:
    sys.modules["unitree.unitree_sdk2py.idl.unitree_go.msg"] = Mock()
if "unitree.unitree_sdk2py.idl.unitree_go.msg.dds_" not in sys.modules:
    sys.modules["unitree.unitree_sdk2py.idl.unitree_go.msg.dds_"] = Mock()

from inputs.plugins.battery_unitree_go2 import (
    Message,
    UnitreeGo2Battery,
    UnitreeGo2BatteryConfig,
)


@pytest.fixture
def mock_channel_subscriber():
    with patch("inputs.plugins.battery_unitree_go2.ChannelSubscriber") as mock_class:
        mock_instance = Mock()
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_lowstate_msg():
    with patch("inputs.plugins.battery_unitree_go2.LowState_") as mock_class:
        mock_instance = Mock()
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_io_provider():
    with patch("inputs.plugins.battery_unitree_go2.IOProvider") as mock_class:
        mock_instance = Mock()
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_teleops_provider():
    with patch(
        "inputs.plugins.battery_unitree_go2.TeleopsStatusProvider"
    ) as mock_class:
        mock_instance = Mock()
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_battery_status():
    with patch("inputs.plugins.battery_unitree_go2.BatteryStatus") as mock_class:
        mock_instance = Mock()
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_teleops_status():
    with patch("inputs.plugins.battery_unitree_go2.TeleopsStatus") as mock_class:
        mock_instance = Mock()
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def unitree_go2_battery_instance(
    mock_channel_subscriber,
    mock_lowstate_msg,
    mock_io_provider,
    mock_teleops_provider,
    mock_battery_status,
    mock_teleops_status,
):
    config = UnitreeGo2BatteryConfig()
    instance = UnitreeGo2Battery(config=config)
    instance.lowstate_subscriber = mock_channel_subscriber
    instance.io_provider = mock_io_provider
    instance.status_provider = mock_teleops_provider
    return instance


def test_initialization_creates_providers_and_subscriber(
    unitree_go2_battery_instance, mock_channel_subscriber
):
    assert unitree_go2_battery_instance.io_provider is not None
    assert unitree_go2_battery_instance.status_provider is not None
    assert hasattr(unitree_go2_battery_instance, "messages")
    assert isinstance(unitree_go2_battery_instance.messages, list)
    mock_channel_subscriber.Init.assert_called_once()


def test_lowstate_message_handler_updates_state(
    unitree_go2_battery_instance, mock_lowstate_msg
):
    mock_bms_state = Mock()
    mock_bms_state.soc = 80.5
    mock_lowstate_msg.bms_state = mock_bms_state
    mock_lowstate_msg.power_v = 16.8
    mock_lowstate_msg.power_a = 2.5
    mock_lowstate_msg.temperature_ntc1 = 30
    mock_lowstate_msg.temperature_ntc2 = 32

    unitree_go2_battery_instance.LowStateMessageHandler(mock_lowstate_msg)

    assert unitree_go2_battery_instance.battery_percentage == 80.5
    assert unitree_go2_battery_instance.battery_voltage == 16.8
    assert unitree_go2_battery_instance.battery_amperes == 2.5
    assert unitree_go2_battery_instance.battery_t == 31  # (30+32)/2


def test_lowstate_message_handler_handles_attribute_error(
    unitree_go2_battery_instance, mock_lowstate_msg
):
    class MockBMSStateWithAttrError:
        @property
        def soc(self):
            raise AttributeError("Mock error accessing soc")

    mock_lowstate_msg.bms_state = MockBMSStateWithAttrError()

    unitree_go2_battery_instance.LowStateMessageHandler(mock_lowstate_msg)

    assert unitree_go2_battery_instance.battery_percentage == 0.0
    assert unitree_go2_battery_instance.battery_voltage == 0.0
    assert unitree_go2_battery_instance.battery_amperes == 0.0
    assert unitree_go2_battery_instance.battery_t == 0


@pytest.mark.asyncio
async def test_report_status_calls_share_status():
    import sys
    from unittest.mock import Mock, patch

    if "unitree" not in sys.modules:
        sys.modules["unitree"] = Mock()
    if "unitree.unitree_sdk2py" not in sys.modules:
        sys.modules["unitree.unitree_sdk2py"] = Mock()
    if "unitree.unitree_sdk2py.core" not in sys.modules:
        sys.modules["unitree.unitree_sdk2py.core"] = Mock()
    if "unitree.unitree_sdk2py.core.channel" not in sys.modules:
        sys.modules["unitree.unitree_sdk2py.core.channel"] = Mock()
    if "unitree.unitree_sdk2py.idl" not in sys.modules:
        sys.modules["unitree.unitree_sdk2py.idl"] = Mock()
    if "unitree.unitree_sdk2py.idl.unitree_go" not in sys.modules:
        sys.modules["unitree.unitree_sdk2py.idl.unitree_go"] = Mock()
    if "unitree.unitree_sdk2py.idl.unitree_go.msg" not in sys.modules:
        sys.modules["unitree.unitree_sdk2py.idl.unitree_go.msg"] = Mock()
    if "unitree.unitree_sdk2py.idl.unitree_go.msg.dds_" not in sys.modules:
        sys.modules["unitree.unitree_sdk2py.idl.unitree_go.msg.dds_"] = Mock()

    from inputs.plugins.battery_unitree_go2 import (
        UnitreeGo2Battery,
        UnitreeGo2BatteryConfig,
    )

    mock_io_provider = Mock()
    mock_teleops_provider = Mock()
    mock_fresh_ts_class = Mock()
    mock_fresh_bs_class = Mock()
    mock_fresh_ts_instance = Mock()
    mock_fresh_bs_instance = Mock()
    mock_fresh_ts_class.return_value = mock_fresh_ts_instance
    mock_fresh_bs_class.return_value = mock_fresh_bs_instance

    with (
        patch(
            "inputs.plugins.battery_unitree_go2.IOProvider",
            return_value=mock_io_provider,
        ),
        patch(
            "inputs.plugins.battery_unitree_go2.TeleopsStatusProvider",
            return_value=mock_teleops_provider,
        ),
        patch("inputs.plugins.battery_unitree_go2.TeleopsStatus", mock_fresh_ts_class),
        patch("inputs.plugins.battery_unitree_go2.BatteryStatus", mock_fresh_bs_class),
    ):

        config = UnitreeGo2BatteryConfig()
        instance = UnitreeGo2Battery(config=config)

        instance.battery_percentage = 80.0
        instance.battery_t = 30
        instance.battery_voltage = 16.8

        with patch("time.time", return_value=1234.0):
            await instance.report_status()

        mock_teleops_provider.share_status.assert_called_once()
        mock_fresh_ts_class.assert_called_once()
        mock_fresh_bs_class.assert_called_once()

        ts_call_kwargs = mock_fresh_ts_class.call_args[1]
        bs_call_kwargs = mock_fresh_bs_class.call_args[1]

        assert ts_call_kwargs["machine_name"] == "UnitreeGo2"
        assert bs_call_kwargs["battery_level"] == 80.0
        assert bs_call_kwargs["temperature"] == 30
        assert bs_call_kwargs["voltage"] == 16.8
        assert bs_call_kwargs["timestamp"] == "1234.0"
        assert bs_call_kwargs["charging_status"] is False


@pytest.mark.asyncio
async def test_poll_calls_report_status_and_returns_correct_list(
    unitree_go2_battery_instance,
):
    unitree_go2_battery_instance.battery_percentage = 80.0
    unitree_go2_battery_instance.battery_voltage = 16.8
    unitree_go2_battery_instance.battery_amperes = 2.5

    with patch.object(
        unitree_go2_battery_instance, "report_status", new=AsyncMock()
    ) as mock_report:
        result = await unitree_go2_battery_instance._poll()

    mock_report.assert_awaited_once()
    assert result == [80.0, 16.8, 2.5]


@pytest.mark.asyncio
async def test_raw_to_text_critical_message(unitree_go2_battery_instance):
    with patch("time.time", return_value=1234.0):
        result = await unitree_go2_battery_instance._raw_to_text([6.0, 16.8, 2.5])

    assert result is not None
    assert "CRITICAL" in result.message
    assert result.timestamp == 1234.0


@pytest.mark.asyncio
async def test_raw_to_text_warning_message(unitree_go2_battery_instance):
    with patch("time.time", return_value=1234.0):
        result = await unitree_go2_battery_instance._raw_to_text([14.0, 16.8, 2.5])

    assert result is not None
    assert "WARNING" in result.message
    assert result.timestamp == 1234.0


@pytest.mark.asyncio
async def test_raw_to_text_no_message_high_battery(unitree_go2_battery_instance):
    result = await unitree_go2_battery_instance._raw_to_text([80.0, 16.8, 2.5])
    assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_adds_to_buffer(unitree_go2_battery_instance):
    with patch("time.time", return_value=1234.0):
        await unitree_go2_battery_instance.raw_to_text(
            [6.0, 16.8, 2.5]
        )  # Should trigger message

    assert len(unitree_go2_battery_instance.messages) == 1
    assert "CRITICAL" in unitree_go2_battery_instance.messages[0].message


@pytest.mark.asyncio
async def test_raw_to_text_none_does_not_add_to_buffer(unitree_go2_battery_instance):
    initial_len = len(unitree_go2_battery_instance.messages)
    await unitree_go2_battery_instance.raw_to_text(
        [80.0, 16.8, 2.5]
    )  # Should not trigger message

    assert len(unitree_go2_battery_instance.messages) == initial_len


def test_formatted_latest_buffer_empty(unitree_go2_battery_instance):
    result = unitree_go2_battery_instance.formatted_latest_buffer()
    assert result is None


def test_formatted_latest_buffer_with_message(unitree_go2_battery_instance):
    msg = Message(timestamp=1234.0, message="buffered message")
    unitree_go2_battery_instance.messages = [msg]

    result = unitree_go2_battery_instance.formatted_latest_buffer()

    assert "INPUT:" in result
    assert "buffered message" in result
    assert len(unitree_go2_battery_instance.messages) == 0
    unitree_go2_battery_instance.io_provider.add_input.assert_called_once()
