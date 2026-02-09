from unittest.mock import MagicMock, patch

import pytest

from providers.unitree_go2_charging_provider import UnitreeGo2ChargingProvider
from zenoh_msgs.idl.status_msgs import ChargingStatus


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    UnitreeGo2ChargingProvider.reset()  # type: ignore
    yield

    try:
        provider = UnitreeGo2ChargingProvider()
        provider.stop()
    except Exception:
        pass

    UnitreeGo2ChargingProvider.reset()  # type: ignore


@pytest.fixture
def mock_zenoh():
    """Mock Zenoh dependencies."""
    with patch("providers.zenoh_listener_provider.open_zenoh_session") as mock_session:
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        yield mock_session, mock_session_instance


def test_initialization(mock_zenoh):
    """Test UnitreeGo2ChargingProvider initialization with default topic."""
    provider = UnitreeGo2ChargingProvider()

    assert provider.sub_topic == "om/go2/charging_status"
    assert provider.latest_status is None
    assert provider.status_history == []
    assert provider.running is False


def test_initialization_custom_topic(mock_zenoh):
    """Test initialization with custom topic."""
    provider = UnitreeGo2ChargingProvider(topic="custom/charging")

    assert provider.sub_topic == "custom/charging"
    assert provider.latest_status is None


def test_singleton_pattern(mock_zenoh):
    """Test that UnitreeGo2ChargingProvider follows singleton pattern."""
    provider1 = UnitreeGo2ChargingProvider(topic="topic1")
    provider2 = UnitreeGo2ChargingProvider(topic="topic2")

    assert provider1 is provider2


def test_charging_message_callback_valid_data(mock_zenoh):
    """Test charging message callback with valid ChargingStatus data."""
    provider = UnitreeGo2ChargingProvider()

    # Create mock ChargingStatus message
    from zenoh_msgs.idl.std_msgs import Header, String, Time

    mock_sample = MagicMock()
    charging_status = ChargingStatus(
        header=Header(stamp=Time(sec=0, nanosec=0), frame_id=""),
        code=1,
        status=String("CHARGING"),
    )

    mock_sample.payload.to_bytes.return_value = charging_status.serialize()

    provider.charging_message_callback(mock_sample)

    assert provider.latest_status == 1
    assert 1 in provider.status_history


def test_charging_message_callback_multiple_messages(mock_zenoh):
    """Test that multiple messages accumulate in status history."""
    provider = UnitreeGo2ChargingProvider()

    # Send multiple status updates
    from zenoh_msgs.idl.std_msgs import Header, String, Time

    for status_code in [0, 1, 2, 3, 1, 0]:
        mock_sample = MagicMock()
        charging_status = ChargingStatus(
            header=Header(stamp=Time(sec=0, nanosec=0), frame_id=""),
            code=status_code,
            status=String(""),
        )

        mock_sample.payload.to_bytes.return_value = charging_status.serialize()
        provider.charging_message_callback(mock_sample)

    assert provider.latest_status == 0  # Last status
    assert provider.status_history == [0, 1, 2, 3, 1, 0]


def test_charging_message_callback_empty_payload(mock_zenoh):
    """Test charging message callback with empty payload."""
    provider = UnitreeGo2ChargingProvider()

    mock_sample = MagicMock()
    mock_sample.payload = None

    provider.charging_message_callback(mock_sample)

    assert provider.latest_status is None
    assert provider.status_history == []


def test_charging_message_callback_invalid_data(mock_zenoh):
    """Test charging message callback with invalid data that fails deserialization."""
    provider = UnitreeGo2ChargingProvider()

    mock_sample = MagicMock()
    mock_sample.payload.to_bytes.return_value = b"invalid data"

    # Should handle exception gracefully
    provider.charging_message_callback(mock_sample)

    assert provider.latest_status is None
    assert provider.status_history == []


def test_start(mock_zenoh):
    """Test starting the provider."""
    mock_session, mock_session_instance = mock_zenoh
    mock_subscriber = MagicMock()
    mock_session_instance.declare_subscriber.return_value = mock_subscriber

    provider = UnitreeGo2ChargingProvider()
    provider.start()

    assert provider.running is True
    mock_session_instance.declare_subscriber.assert_called_once()


def test_start_already_running(mock_zenoh):
    """Test starting provider when already running."""
    provider = UnitreeGo2ChargingProvider()
    provider.start()

    assert provider.running is True

    # Try to start again
    provider.start()

    assert provider.running is True


def test_charging_status_property(mock_zenoh):
    """Test the charging_status property getter."""
    provider = UnitreeGo2ChargingProvider()

    assert provider.charging_status is None

    # Simulate receiving a message
    from zenoh_msgs.idl.std_msgs import Header, String, Time

    mock_sample = MagicMock()
    charging_status = ChargingStatus(
        header=Header(stamp=Time(sec=0, nanosec=0), frame_id=""),
        code=2,
        status=String("ENROUTE_CHARGING"),
    )

    mock_sample.payload.to_bytes.return_value = charging_status.serialize()
    provider.charging_message_callback(mock_sample)

    assert provider.charging_status == 2


def test_get_charging_status(mock_zenoh):
    """Test the get_charging_status() method."""
    from zenoh_msgs.idl.std_msgs import Header, String, Time

    provider = UnitreeGo2ChargingProvider()

    assert provider.get_charging_status() is None

    # Simulate receiving a message
    mock_sample = MagicMock()
    charging_status = ChargingStatus(
        header=Header(stamp=Time(sec=0, nanosec=0), frame_id=""),
        code=3,
        status=String("FULLY_CHARGED"),
    )

    mock_sample.payload.to_bytes.return_value = charging_status.serialize()
    provider.charging_message_callback(mock_sample)

    assert provider.get_charging_status() == 3


def test_get_status_history(mock_zenoh):
    """Test get_status_history() returns accumulated statuses."""
    from zenoh_msgs.idl.std_msgs import Header, String, Time

    provider = UnitreeGo2ChargingProvider()

    assert provider.get_status_history() == []

    # Send multiple updates
    statuses = [0, 1, 1, 2, 3]
    for status_code in statuses:
        mock_sample = MagicMock()
        charging_status = ChargingStatus(
            header=Header(stamp=Time(sec=0, nanosec=0), frame_id=""),
            code=status_code,
            status=String(""),
        )

        mock_sample.payload.to_bytes.return_value = charging_status.serialize()
        provider.charging_message_callback(mock_sample)

    assert provider.get_status_history() == statuses
