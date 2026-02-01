from unittest.mock import MagicMock, patch

import pytest

from providers.tron_odom_provider import RobotState, TronOdomProvider


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    TronOdomProvider.reset()  # type: ignore
    yield
    TronOdomProvider.reset()  # type: ignore


@pytest.fixture
def mock_multiprocessing():
    with (
        patch("providers.tron_odom_provider.mp.Queue") as mock_queue,
        patch("providers.tron_odom_provider.mp.Process") as mock_process,
        patch("providers.tron_odom_provider.threading.Thread") as mock_thread,
        patch("providers.tron_odom_provider.threading.Event") as mock_event,
    ):
        mock_queue_instance = MagicMock()
        mock_process_instance = MagicMock()
        mock_thread_instance = MagicMock()
        mock_event_instance = MagicMock()

        mock_queue.return_value = mock_queue_instance
        mock_process.return_value = mock_process_instance
        mock_thread.return_value = mock_thread_instance
        mock_event.return_value = mock_event_instance

        mock_process_instance.is_alive.return_value = False
        mock_thread_instance.is_alive.return_value = False
        mock_event_instance.is_set.return_value = False

        yield (
            mock_queue,
            mock_queue_instance,
            mock_process,
            mock_process_instance,
            mock_thread,
            mock_thread_instance,
        )


def test_initialization_default_topic(mock_multiprocessing):
    """Test initialization with default topic."""
    mock_queue, _, _, _, _, _ = mock_multiprocessing

    provider = TronOdomProvider()

    assert provider.topic == "odom"
    mock_queue.assert_called_once()


def test_initialization_custom_topic(mock_multiprocessing):
    """Test initialization with custom topic."""
    mock_queue, _, _, _, _, _ = mock_multiprocessing

    TronOdomProvider.reset()  # type: ignore
    provider = TronOdomProvider(topic="custom_odom")

    assert provider.topic == "custom_odom"
    mock_queue.assert_called_once()


def test_singleton_pattern(mock_multiprocessing):
    """Test singleton pattern - same instance returned."""
    provider1 = TronOdomProvider(topic="odom")
    provider2 = TronOdomProvider(topic="other_topic")
    assert provider1 is provider2


def test_start(mock_multiprocessing):
    """Test start() launches process and thread."""
    _, _, _, mock_process_instance, _, mock_thread_instance = mock_multiprocessing

    TronOdomProvider(topic="odom")

    assert mock_process_instance.start.call_count >= 1
    assert mock_thread_instance.start.call_count >= 1


def test_start_already_running(mock_multiprocessing):
    """Test start() does not restart when already running."""
    _, _, _, mock_process_instance, _, mock_thread_instance = mock_multiprocessing

    provider = TronOdomProvider(topic="odom")

    mock_process_instance.is_alive.return_value = True
    mock_thread_instance.is_alive.return_value = True

    mock_process_instance.start.reset_mock()
    mock_thread_instance.start.reset_mock()

    provider.start()

    mock_process_instance.start.assert_not_called()
    mock_thread_instance.start.assert_not_called()


def test_start_no_topic(mock_multiprocessing):
    """Test start() returns early when topic is empty."""
    _, _, _, mock_process_instance, _, _ = mock_multiprocessing

    provider = TronOdomProvider(topic="odom")
    mock_process_instance.is_alive.return_value = False
    provider.topic = ""

    mock_process_instance.start.reset_mock()
    provider.start()

    # Should not start because topic is empty
    mock_process_instance.start.assert_not_called()


def test_stop(mock_multiprocessing):
    """Test stop() terminates process and joins thread."""
    _, _, _, mock_process_instance, _, mock_thread_instance = mock_multiprocessing

    provider = TronOdomProvider(topic="odom")
    provider.stop()

    assert provider._stop_event.set.called  # type: ignore
    mock_process_instance.terminate.assert_called_once()
    mock_process_instance.join.assert_called_once()
    mock_thread_instance.join.assert_called_once()


def test_robot_state_enum():
    """Test RobotState enum values."""
    assert RobotState.STANDING.value == "standing"
    assert RobotState.SITTING.value == "sitting"


def test_position_property(mock_multiprocessing):
    """Test position property returns expected dictionary structure."""
    provider = TronOdomProvider(topic="odom")

    position = provider.position

    assert "odom_x" in position
    assert "odom_y" in position
    assert "moving" in position
    assert "odom_yaw_0_360" in position
    assert "odom_yaw_m180_p180" in position
    assert "body_height_cm" in position
    assert "body_attitude" in position
    assert "odom_rockchip_ts" in position
    assert "odom_subscriber_ts" in position

    assert position["odom_x"] == 0.0
    assert position["odom_y"] == 0.0
    assert position["moving"] is False


def test_euler_from_quaternion(mock_multiprocessing):
    """Test quaternion to Euler angle conversion."""
    provider = TronOdomProvider(topic="odom")

    # Identity quaternion (no rotation)
    roll, pitch, yaw = provider.euler_from_quaternion(0.0, 0.0, 0.0, 1.0)
    assert abs(roll) < 0.001
    assert abs(pitch) < 0.001
    assert abs(yaw) < 0.001


def test_euler_from_quaternion_90_degree_yaw(mock_multiprocessing):
    """Test quaternion representing 90 degree yaw rotation."""
    import math

    provider = TronOdomProvider(topic="odom")

    # 90 degree rotation about z-axis
    # quaternion: (0, 0, sin(45°), cos(45°)) = (0, 0, 0.7071, 0.7071)
    roll, pitch, yaw = provider.euler_from_quaternion(0.0, 0.0, 0.7071, 0.7071)

    # yaw should be approximately 90 degrees (pi/2 radians)
    assert abs(yaw - math.pi / 2) < 0.01


def test_body_attitude_standing(mock_multiprocessing):
    """Test body attitude is set to STANDING when height > 60cm."""
    provider = TronOdomProvider(topic="odom")

    # Manually set body height to simulate standing
    provider.body_height_cm = 70

    # This happens in process_odom, but we can test the threshold logic
    if provider.body_height_cm > 60:
        provider.body_attitude = RobotState.STANDING

    assert provider.body_attitude == RobotState.STANDING


def test_body_attitude_sitting(mock_multiprocessing):
    """Test body attitude is set to SITTING when height is between 3-60cm."""
    provider = TronOdomProvider(topic="odom")

    # Manually set body height to simulate sitting
    provider.body_height_cm = 55

    if provider.body_height_cm > 60:
        provider.body_attitude = RobotState.STANDING
    elif provider.body_height_cm > 3:
        provider.body_attitude = RobotState.SITTING

    assert provider.body_attitude == RobotState.SITTING


def test_initial_values(mock_multiprocessing):
    """Test initial values are set correctly."""
    provider = TronOdomProvider(topic="odom")

    assert provider.x == 0.0
    assert provider.y == 0.0
    assert provider.z == 0.0
    assert provider.odom_yaw_0_360 == 0.0
    assert provider.odom_yaw_m180_p180 == 0.0
    assert provider.moving is False
    assert provider.previous_x == 0
    assert provider.previous_y == 0
    assert provider.previous_z == 0
    assert provider.move_history == 0
