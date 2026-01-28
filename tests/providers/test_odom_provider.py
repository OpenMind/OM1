import math
import multiprocessing as mp
import threading
import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.providers.odom_provider import OdomProvider, RobotState


@pytest.fixture
def odom_provider():
    """
    Fixture to create an OdomProvider instance for testing.
    Bypasses singleton pattern to allow clean test isolation.
    """
    original_class = OdomProvider._singleton_class  # type: ignore
    provider = original_class.__new__(original_class)
    return provider


@pytest.fixture
def mock_pose_data():
    """
    Create a mock PoseWithCovarianceStamped message for testing.
    """
    from src.zenoh_msgs.idl.geometry_msgs import (
        Point,
        Pose,
        PoseWithCovariance,
        Quaternion,
    )
    from src.zenoh_msgs.idl.std_msgs import Header, Time

    header = Header(
        stamp=Time(sec=1234567890, nanosec=123456789),
        frame_id="base_link",
    )
    pose = Pose(
        position=Point(x=1.0, y=2.0, z=0.5),
        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    pose_with_cov = PoseWithCovariance(pose=pose, covariance=[0.0] * 36)
    return type("PoseWithCovarianceStamped", (), {"header": header, "pose": pose_with_cov})()


def test_initialization_with_zenoh(odom_provider):
    """
    Test OdomProvider initialization with Zenoh mode enabled.
    """
    with patch("src.providers.odom_provider.mp.Process") as mock_process:
        mock_process_instance = MagicMock()
        mock_process.return_value = mock_process_instance

        odom_provider.__init__(URID="test_urid", use_zenoh=True, channel="")

        assert odom_provider.use_zenoh is True
        assert odom_provider.URID == "test_urid"
        assert odom_provider.channel == ""
        assert odom_provider.data_queue is not None
        assert odom_provider.x == 0.0
        assert odom_provider.y == 0.0
        assert odom_provider.moving is False


def test_initialization_with_dds(odom_provider):
    """
    Test OdomProvider initialization with CycloneDDS mode (default).
    """
    with patch("src.providers.odom_provider.mp.Process") as mock_process:
        mock_process_instance = MagicMock()
        mock_process.return_value = mock_process_instance

        odom_provider.__init__(URID="", use_zenoh=False, channel="test_channel")

        assert odom_provider.use_zenoh is False
        assert odom_provider.URID == ""
        assert odom_provider.channel == "test_channel"
        assert odom_provider.data_queue is not None


def test_initialization_default_values(odom_provider):
    """
    Test OdomProvider initialization with default parameter values.
    """
    with patch("src.providers.odom_provider.mp.Process") as mock_process:
        mock_process_instance = MagicMock()
        mock_process.return_value = mock_process_instance

        odom_provider.__init__()

        assert odom_provider.use_zenoh is False
        assert odom_provider.URID == ""
        assert odom_provider.channel == ""


def test_start_with_zenoh(odom_provider):
    """
    Test starting OdomProvider in Zenoh mode.
    """
    odom_provider.use_zenoh = True
    odom_provider.URID = "test_urid"
    odom_provider.channel = ""
    odom_provider.data_queue = mp.Queue()
    odom_provider._odom_reader_thread = None
    odom_provider._odom_processor_thread = None
    odom_provider._stop_event = threading.Event()

    with patch("providers.odom_provider.mp.Process") as mock_process, patch(
        "providers.odom_provider.threading.Thread"
    ) as mock_thread:
        mock_process_instance = MagicMock()
        mock_process_instance.is_alive.return_value = False
        mock_process.return_value = mock_process_instance

        mock_thread_instance = MagicMock()
        mock_thread_instance.is_alive.return_value = False
        mock_thread.return_value = mock_thread_instance

        odom_provider.start()

        assert mock_process.called
        assert mock_thread.called
        assert mock_process_instance.start.called
        assert mock_thread_instance.start.called


def test_start_with_dds(odom_provider):
    """
    Test starting OdomProvider in CycloneDDS mode.
    """
    odom_provider.use_zenoh = False
    odom_provider.URID = ""
    odom_provider.channel = "test_channel"
    odom_provider.data_queue = mp.Queue()
    odom_provider._odom_reader_thread = None
    odom_provider._odom_processor_thread = None
    odom_provider._stop_event = threading.Event()

    with patch("providers.odom_provider.mp.Process") as mock_process, patch(
        "providers.odom_provider.threading.Thread"
    ) as mock_thread:
        mock_process_instance = MagicMock()
        mock_process_instance.is_alive.return_value = False
        mock_process.return_value = mock_process_instance

        mock_thread_instance = MagicMock()
        mock_thread_instance.is_alive.return_value = False
        mock_thread.return_value = mock_thread_instance

        odom_provider.start()

        assert mock_process.called
        assert mock_thread.called


def test_start_without_channel_error(odom_provider):
    """
    Test that start() logs an error when channel is not specified in DDS mode.
    """
    odom_provider.use_zenoh = False
    odom_provider.channel = ""
    odom_provider.data_queue = mp.Queue()
    odom_provider._odom_reader_thread = None
    odom_provider._odom_processor_thread = None
    odom_provider._stop_event = threading.Event()

    with patch("src.providers.odom_provider.logging") as mock_logging:
        odom_provider.start()
        mock_logging.error.assert_called_with(
            "Channel must be specified to start the Odom Provider."
        )


def test_start_already_running_warning(odom_provider):
    """
    Test that start() logs a warning when already running.
    """
    odom_provider.use_zenoh = False
    odom_provider.channel = "test_channel"
    odom_provider.data_queue = mp.Queue()
    odom_provider._stop_event = threading.Event()

    mock_process = MagicMock()
    mock_process.is_alive.return_value = True
    odom_provider._odom_reader_thread = mock_process

    mock_thread = MagicMock()
    mock_thread.is_alive.return_value = True
    odom_provider._odom_processor_thread = mock_thread

    with patch("src.providers.odom_provider.logging") as mock_logging:
        odom_provider.start()
        mock_logging.warning.assert_any_call("Odom Provider is already running.")
        mock_logging.warning.assert_any_call("Odom processor thread is already running.")


def test_stop(odom_provider):
    """
    Test stopping OdomProvider and resource cleanup.
    """
    odom_provider._stop_event = threading.Event()
    odom_provider._odom_reader_thread = MagicMock()
    odom_provider._odom_processor_thread = MagicMock()

    with patch("src.providers.odom_provider.logging") as mock_logging:
        odom_provider.stop()

        assert odom_provider._stop_event.is_set()
        assert odom_provider._odom_reader_thread.terminate.called
        assert odom_provider._odom_reader_thread.join.called
        assert odom_provider._odom_processor_thread.join.called
        mock_logging.info.assert_any_call("OdomProvider reader thread stopped.")
        mock_logging.info.assert_any_call("OdomProvider processor thread stopped.")


def test_stop_with_none_threads(odom_provider):
    """
    Test stopping OdomProvider when threads are None.
    """
    odom_provider._stop_event = threading.Event()
    odom_provider._odom_reader_thread = None
    odom_provider._odom_processor_thread = None

    # Should not raise an exception
    odom_provider.stop()
    assert odom_provider._stop_event.is_set()


def test_euler_from_quaternion_identity(odom_provider):
    """
    Test quaternion to euler conversion with identity quaternion (no rotation).
    """
    roll, pitch, yaw = odom_provider.euler_from_quaternion(0.0, 0.0, 0.0, 1.0)

    assert math.isclose(roll, 0.0, abs_tol=1e-6)
    assert math.isclose(pitch, 0.0, abs_tol=1e-6)
    assert math.isclose(yaw, 0.0, abs_tol=1e-6)


def test_euler_from_quaternion_90_degree_yaw(odom_provider):
    """
    Test quaternion to euler conversion for 90 degree yaw rotation.
    """
    # Quaternion for 90 degree rotation around z-axis
    roll, pitch, yaw = odom_provider.euler_from_quaternion(0.0, 0.0, 0.7071068, 0.7071068)

    assert math.isclose(roll, 0.0, abs_tol=1e-4)
    assert math.isclose(pitch, 0.0, abs_tol=1e-4)
    assert math.isclose(yaw, math.pi / 2, abs_tol=1e-4)


def test_euler_from_quaternion_180_degree_yaw(odom_provider):
    """
    Test quaternion to euler conversion for 180 degree yaw rotation.
    """
    # Quaternion for 180 degree rotation around z-axis
    roll, pitch, yaw = odom_provider.euler_from_quaternion(0.0, 0.0, 1.0, 0.0)

    assert math.isclose(roll, 0.0, abs_tol=1e-4)
    assert math.isclose(pitch, 0.0, abs_tol=1e-4)
    assert math.isclose(yaw, math.pi, abs_tol=1e-4)


def test_process_odom_data_basic(odom_provider, mock_pose_data):
    """
    Test processing odometry data and updating internal state.
    """
    odom_provider.data_queue = mp.Queue()
    odom_provider._stop_event = threading.Event()
    odom_provider.channel = ""
    odom_provider.use_zenoh = True

    # Put test data in queue
    odom_provider.data_queue.put(mock_pose_data)

    # Set stop event immediately to exit loop after one iteration
    odom_provider._stop_event.set()

    # Process the data
    odom_provider.process_odom()

    # Verify position was updated
    assert odom_provider.x == 1.0
    assert odom_provider.y == 2.0
    assert odom_provider.odom_rockchip_ts == 1234567890.123456789


def test_process_odom_data_with_movement(odom_provider, mock_pose_data):
    """
    Test processing odometry data that indicates robot movement.
    """
    odom_provider.data_queue = mp.Queue()
    odom_provider._stop_event = threading.Event()
    odom_provider.channel = ""
    odom_provider.use_zenoh = True
    odom_provider.previous_x = 0.0
    odom_provider.previous_y = 0.0
    odom_provider.previous_z = 0.0
    odom_provider.move_history = 0.0

    # Put test data in queue (position at 1.0, 2.0, 0.5)
    odom_provider.data_queue.put(mock_pose_data)

    # Set stop event after processing
    odom_provider._stop_event.set()

    odom_provider.process_odom()

    # Verify movement was detected (delta > 0.01)
    assert odom_provider.moving is True
    assert odom_provider.previous_x == 1.0
    assert odom_provider.previous_y == 2.0
    assert odom_provider.previous_z == 0.5


def test_process_odom_data_no_movement(odom_provider, mock_pose_data):
    """
    Test processing odometry data with no significant movement.
    """
    from src.zenoh_msgs.idl.geometry_msgs import (
        Point,
        Pose,
        PoseWithCovariance,
        Quaternion,
    )
    from src.zenoh_msgs.idl.std_msgs import Header, Time

    odom_provider.data_queue = mp.Queue()
    odom_provider._stop_event = threading.Event()
    odom_provider.channel = ""
    odom_provider.use_zenoh = True
    odom_provider.previous_x = 1.0
    odom_provider.previous_y = 2.0
    odom_provider.previous_z = 0.5
    odom_provider.move_history = 0.0

    # Create pose data with minimal movement (less than 0.01m)
    header = Header(
        stamp=Time(sec=1234567890, nanosec=123456789),
        frame_id="base_link",
    )
    pose = Pose(
        position=Point(x=1.005, y=2.005, z=0.5),  # Very small movement
        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    pose_with_cov = PoseWithCovariance(pose=pose, covariance=[0.0] * 36)
    small_move_data = type(
        "PoseWithCovarianceStamped", (), {"header": header, "pose": pose_with_cov}
    )()

    odom_provider.data_queue.put(small_move_data)
    odom_provider._stop_event.set()

    odom_provider.process_odom()

    # Verify no movement was detected
    assert odom_provider.moving is False


def test_process_odom_data_with_unitree_body_height(odom_provider, mock_pose_data):
    """
    Test processing odometry data for Unitree Go2 with body height calculation.
    """
    from src.zenoh_msgs.idl.geometry_msgs import (
        Point,
        Pose,
        PoseWithCovariance,
        Quaternion,
    )
    from src.zenoh_msgs.idl.std_msgs import Header, Time

    odom_provider.data_queue = mp.Queue()
    odom_provider._stop_event = threading.Event()
    odom_provider.channel = "test_channel"
    odom_provider.use_zenoh = False

    # Create pose data with z=0.3m (30cm, standing)
    header = Header(
        stamp=Time(sec=1234567890, nanosec=123456789),
        frame_id="base_link",
    )
    pose = Pose(
        position=Point(x=1.0, y=2.0, z=0.3),
        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    pose_with_cov = PoseWithCovariance(pose=pose, covariance=[0.0] * 36)
    unitree_data = type(
        "PoseWithCovarianceStamped", (), {"header": header, "pose": pose_with_cov}
    )()

    odom_provider.data_queue.put(unitree_data)
    odom_provider._stop_event.set()

    odom_provider.process_odom()

    # Verify body height and attitude were calculated
    assert odom_provider.body_height_cm == 30
    assert odom_provider.body_attitude == RobotState.STANDING


def test_process_odom_data_queue_error(odom_provider):
    """
    Test that process_odom handles queue errors gracefully.
    """
    odom_provider.data_queue = MagicMock()
    odom_provider.data_queue.get.side_effect = Exception("Queue error")
    odom_provider._stop_event = threading.Event()

    # Set stop event after one iteration
    def set_stop_after_sleep():
        time.sleep(0.1)
        odom_provider._stop_event.set()

    threading.Thread(target=set_stop_after_sleep, daemon=True).start()

    with patch("src.providers.odom_provider.logging") as mock_logging:
        odom_provider.process_odom()
        mock_logging.error.assert_called()


def test_position_property(odom_provider):
    """
    Test the position property returns correct dictionary structure.
    """
    odom_provider.x = 1.5
    odom_provider.y = 2.5
    odom_provider.moving = True
    odom_provider.odom_yaw_0_360 = 90.0
    odom_provider.odom_yaw_m180_p180 = -90.0
    odom_provider.body_height_cm = 30
    odom_provider.body_attitude = RobotState.STANDING
    odom_provider.odom_rockchip_ts = 1234567890.0
    odom_provider.odom_subscriber_ts = 1234567891.0

    position = odom_provider.position

    assert position["odom_x"] == 1.5
    assert position["odom_y"] == 2.5
    assert position["moving"] is True
    assert position["odom_yaw_0_360"] == 90.0
    assert position["odom_yaw_m180_p180"] == -90.0
    assert position["body_height_cm"] == 30
    assert position["body_attitude"] == RobotState.STANDING
    assert position["odom_rockchip_ts"] == 1234567890.0
    assert position["odom_subscriber_ts"] == 1234567891.0


def test_yaw_conversion_0_360(odom_provider, mock_pose_data):
    """
    Test that yaw is correctly converted to 0-360 degree range.
    """
    from src.zenoh_msgs.idl.geometry_msgs import (
        Point,
        Pose,
        PoseWithCovariance,
        Quaternion,
    )
    from src.zenoh_msgs.idl.std_msgs import Header, Time

    odom_provider.data_queue = mp.Queue()
    odom_provider._stop_event = threading.Event()
    odom_provider.channel = ""
    odom_provider.use_zenoh = True

    # Create quaternion for -90 degree yaw (should become 270 in 0-360 range)
    header = Header(
        stamp=Time(sec=1234567890, nanosec=123456789),
        frame_id="base_link",
    )
    pose = Pose(
        position=Point(x=0.0, y=0.0, z=0.0),
        orientation=Quaternion(x=0.0, y=0.0, z=-0.7071068, w=0.7071068),
    )
    pose_with_cov = PoseWithCovariance(pose=pose, covariance=[0.0] * 36)
    yaw_data = type(
        "PoseWithCovarianceStamped", (), {"header": header, "pose": pose_with_cov}
    )()

    odom_provider.data_queue.put(yaw_data)
    odom_provider._stop_event.set()

    odom_provider.process_odom()

    # Verify yaw conversion: -90 degrees should become 270 in 0-360 range
    assert math.isclose(odom_provider.odom_yaw_m180_p180, -90.0, abs_tol=1.0)
    assert math.isclose(odom_provider.odom_yaw_0_360, 270.0, abs_tol=1.0)
