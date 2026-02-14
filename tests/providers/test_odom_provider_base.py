import math
from unittest.mock import MagicMock, patch

import pytest

from providers.odom_provider_base import OdomProviderBase, RobotState


class ConcreteOdomProvider(OdomProviderBase):
    """Concrete implementation of OdomProviderBase for testing."""

    def start(self) -> None:
        """Start the provider - simple implementation for testing."""
        pass


@pytest.fixture
def mock_multiprocessing():
    """Mock multiprocessing and threading components."""
    with (
        patch("providers.odom_provider_base.mp.Queue") as mock_queue,
        patch("providers.odom_provider_base.threading.Event") as mock_event,
    ):
        mock_queue_instance = MagicMock()
        mock_event_instance = MagicMock()

        mock_queue.return_value = mock_queue_instance
        mock_event.return_value = mock_event_instance

        mock_event_instance.is_set.return_value = False

        yield mock_queue, mock_queue_instance, mock_event, mock_event_instance


class TestRobotState:
    """Test cases for RobotState enum."""

    def test_standing_value(self):
        """Test STANDING enum value."""
        assert RobotState.STANDING.value == "standing"

    def test_sitting_value(self):
        """Test SITTING enum value."""
        assert RobotState.SITTING.value == "sitting"


class TestOdomProviderBase:
    """Test cases for OdomProviderBase."""

    def test_initialization(self, mock_multiprocessing):
        """Test basic initialization."""
        provider = ConcreteOdomProvider()

        assert provider.body_height_cm == 0
        assert provider.body_attitude is None
        assert provider.moving is False
        assert provider.previous_x == 0.0
        assert provider.previous_y == 0.0
        assert provider.previous_z == 0.0
        assert provider.move_history == 0.0
        assert provider.x == 0.0
        assert provider.y == 0.0
        assert provider.z == 0.0
        assert provider.odom_yaw_0_360 == 0.0
        assert provider.odom_yaw_m180_p180 == 0.0
        assert provider.odom_rockchip_ts == 0.0
        assert provider.odom_subscriber_ts == 0.0

    def test_euler_from_quaternion_identity(self, mock_multiprocessing):
        """Test euler conversion with identity quaternion."""
        provider = ConcreteOdomProvider()

        # Identity quaternion (no rotation)
        roll, pitch, yaw = provider.euler_from_quaternion(0, 0, 0, 1)

        assert abs(roll) < 1e-10
        assert abs(pitch) < 1e-10
        assert abs(yaw) < 1e-10

    def test_euler_from_quaternion_90_deg_yaw(self, mock_multiprocessing):
        """Test euler conversion with 90 degree yaw rotation."""
        provider = ConcreteOdomProvider()

        # 90 degree rotation around z-axis
        # quaternion for 90 deg yaw: w=cos(45°), z=sin(45°)
        w = math.cos(math.pi / 4)
        z = math.sin(math.pi / 4)
        roll, pitch, yaw = provider.euler_from_quaternion(0, 0, z, w)

        assert abs(roll) < 1e-6
        assert abs(pitch) < 1e-6
        assert abs(yaw - math.pi / 2) < 1e-6  # 90 degrees in radians

    def test_euler_from_quaternion_180_deg_yaw(self, mock_multiprocessing):
        """Test euler conversion with 180 degree yaw rotation."""
        provider = ConcreteOdomProvider()

        # 180 degree rotation around z-axis
        roll, pitch, yaw = provider.euler_from_quaternion(0, 0, 1, 0)

        assert abs(roll) < 1e-6
        assert abs(pitch) < 1e-6
        assert abs(abs(yaw) - math.pi) < 1e-6  # 180 degrees in radians

    def test_position_property(self, mock_multiprocessing):
        """Test position property returns correct dictionary."""
        provider = ConcreteOdomProvider()

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
        assert position["odom_yaw_0_360"] == 0.0
        assert position["odom_yaw_m180_p180"] == 0.0
        assert position["body_height_cm"] == 0
        assert position["body_attitude"] is None

    def test_position_property_with_updated_values(self, mock_multiprocessing):
        """Test position property after updating internal state."""
        provider = ConcreteOdomProvider()

        provider.x = 1.5
        provider.y = 2.3
        provider.moving = True
        provider.odom_yaw_0_360 = 90.0
        provider.odom_yaw_m180_p180 = 45.0
        provider.body_height_cm = 30
        provider.body_attitude = RobotState.STANDING

        position = provider.position

        assert position["odom_x"] == 1.5
        assert position["odom_y"] == 2.3
        assert position["moving"] is True
        assert position["odom_yaw_0_360"] == 90.0
        assert position["odom_yaw_m180_p180"] == 45.0
        assert position["body_height_cm"] == 30
        assert position["body_attitude"] == RobotState.STANDING

    def test_process_odom_stops_on_event(self, mock_multiprocessing):
        """Test process_odom stops when stop event is set."""
        _, mock_queue_instance, _, mock_event_instance = mock_multiprocessing

        provider = ConcreteOdomProvider()

        mock_event_instance.is_set.return_value = True

        provider.process_odom()

        mock_queue_instance.get.assert_not_called()

    def test_process_odom_with_pose_data(self, mock_multiprocessing):
        """Test process_odom processes pose data correctly."""
        _, mock_queue_instance, _, mock_event_instance = mock_multiprocessing

        provider = ConcreteOdomProvider()

        mock_pose_data = MagicMock()
        mock_pose_data.pose.position.x = 1.0
        mock_pose_data.pose.position.y = 2.0
        mock_pose_data.pose.position.z = 0.0
        mock_pose_data.pose.orientation.x = 0.0
        mock_pose_data.pose.orientation.y = 0.0
        mock_pose_data.pose.orientation.z = 0.0
        mock_pose_data.pose.orientation.w = 1.0
        mock_pose_data.header.stamp.sec = 100
        mock_pose_data.header.stamp.nanosec = 500000000
        del mock_pose_data.pose.pose

        call_count = [0]

        def side_effect_get(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_pose_data
            raise Exception("Queue timeout")

        def side_effect_is_set():
            return call_count[0] > 0

        mock_queue_instance.get.side_effect = side_effect_get
        mock_event_instance.is_set.side_effect = side_effect_is_set

        with patch("providers.odom_provider_base.time.time", return_value=1000.0):
            provider.process_odom()

        assert provider.x == 1.0
        assert provider.y == 2.0
        assert provider.odom_rockchip_ts == 100.5

    def test_process_odom_with_pose_with_covariance(self, mock_multiprocessing):
        """Test process_odom handles PoseWithCovarianceStamped format."""
        _, mock_queue_instance, _, mock_event_instance = mock_multiprocessing

        provider = ConcreteOdomProvider()

        mock_pose_data = MagicMock()
        mock_pose_data.pose.pose.position.x = 1.5
        mock_pose_data.pose.pose.position.y = 2.5
        mock_pose_data.pose.pose.position.z = 0.0
        mock_pose_data.pose.pose.orientation.x = 0.0
        mock_pose_data.pose.pose.orientation.y = 0.0
        mock_pose_data.pose.pose.orientation.z = 0.0
        mock_pose_data.pose.pose.orientation.w = 1.0
        mock_pose_data.header.stamp.sec = 200
        mock_pose_data.header.stamp.nanosec = 0

        call_count = [0]

        def side_effect_get(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_pose_data
            raise Exception("Queue timeout")

        def side_effect_is_set():
            return call_count[0] > 0

        mock_queue_instance.get.side_effect = side_effect_get
        mock_event_instance.is_set.side_effect = side_effect_is_set

        with patch("providers.odom_provider_base.time.time", return_value=2000.0):
            provider.process_odom()

        assert provider.x == 1.5
        assert provider.y == 2.5
        assert provider.odom_rockchip_ts == 200.0
        assert provider.odom_subscriber_ts == 2000.0

    def test_process_odom_detects_movement(self, mock_multiprocessing):
        """Test process_odom detects robot movement."""
        _, mock_queue_instance, _, mock_event_instance = mock_multiprocessing

        provider = ConcreteOdomProvider()

        mock_pose_data1 = MagicMock()
        mock_pose_data1.pose.position.x = 0.0
        mock_pose_data1.pose.position.y = 0.0
        mock_pose_data1.pose.position.z = 0.0
        mock_pose_data1.pose.orientation.x = 0.0
        mock_pose_data1.pose.orientation.y = 0.0
        mock_pose_data1.pose.orientation.z = 0.0
        mock_pose_data1.pose.orientation.w = 1.0
        mock_pose_data1.header.stamp.sec = 100
        mock_pose_data1.header.stamp.nanosec = 0
        del mock_pose_data1.pose.pose

        mock_pose_data2 = MagicMock()
        mock_pose_data2.pose.position.x = 0.5  # Significant movement
        mock_pose_data2.pose.position.y = 0.5
        mock_pose_data2.pose.position.z = 0.0
        mock_pose_data2.pose.orientation.x = 0.0
        mock_pose_data2.pose.orientation.y = 0.0
        mock_pose_data2.pose.orientation.z = 0.0
        mock_pose_data2.pose.orientation.w = 1.0
        mock_pose_data2.header.stamp.sec = 101
        mock_pose_data2.header.stamp.nanosec = 0
        del mock_pose_data2.pose.pose

        call_count = [0]

        def side_effect_get(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_pose_data1
            elif call_count[0] == 2:
                return mock_pose_data2
            raise Exception("Queue timeout")

        def side_effect_is_set():
            return call_count[0] > 1

        mock_queue_instance.get.side_effect = side_effect_get
        mock_event_instance.is_set.side_effect = side_effect_is_set

        with patch("providers.odom_provider_base.time.time", return_value=1000.0):
            provider.process_odom()

        # After significant movement, should be detected as moving
        assert provider.moving is True

    def test_update_body_state_default(self, mock_multiprocessing):
        """Test _update_body_state default implementation does nothing."""
        provider = ConcreteOdomProvider()

        mock_pose = MagicMock()
        initial_height = provider.body_height_cm
        initial_attitude = provider.body_attitude

        provider._update_body_state(mock_pose)

        assert provider.body_height_cm == initial_height
        assert provider.body_attitude == initial_attitude

    def test_stop(self, mock_multiprocessing):
        """Test stop method cleans up resources."""
        _, _, _, mock_event_instance = mock_multiprocessing

        provider = ConcreteOdomProvider()

        mock_reader_thread = MagicMock()
        mock_processor_thread = MagicMock()

        provider._odom_reader_thread = mock_reader_thread
        provider._odom_processor_thread = mock_processor_thread

        provider.stop()

        mock_event_instance.set.assert_called_once()
        mock_reader_thread.terminate.assert_called_once()
        mock_reader_thread.join.assert_called_once()
        mock_processor_thread.join.assert_called_once()

    def test_stop_without_threads(self, mock_multiprocessing):
        """Test stop method when threads are None."""
        _, _, _, mock_event_instance = mock_multiprocessing

        provider = ConcreteOdomProvider()

        provider._odom_reader_thread = None
        provider._odom_processor_thread = None

        provider.stop()
        mock_event_instance.set.assert_called_once()

    def test_yaw_conversion_positive(self, mock_multiprocessing):
        """Test yaw angle conversion for positive angles."""
        _, mock_queue_instance, _, mock_event_instance = mock_multiprocessing

        provider = ConcreteOdomProvider()

        # Create pose with yaw rotation (positive)
        mock_pose_data = MagicMock()
        mock_pose_data.pose.position.x = 0.0
        mock_pose_data.pose.position.y = 0.0
        mock_pose_data.pose.position.z = 0.0
        # 45 degree yaw rotation
        mock_pose_data.pose.orientation.x = 0.0
        mock_pose_data.pose.orientation.y = 0.0
        mock_pose_data.pose.orientation.z = math.sin(math.pi / 8)  # 22.5 deg
        mock_pose_data.pose.orientation.w = math.cos(math.pi / 8)
        mock_pose_data.header.stamp.sec = 100
        mock_pose_data.header.stamp.nanosec = 0
        del mock_pose_data.pose.pose

        call_count = [0]

        def side_effect_get(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_pose_data
            raise Exception("Queue timeout")

        def side_effect_is_set():
            return call_count[0] > 0

        mock_queue_instance.get.side_effect = side_effect_get
        mock_event_instance.is_set.side_effect = side_effect_is_set

        with patch("providers.odom_provider_base.time.time", return_value=1000.0):
            provider.process_odom()

        assert provider.odom_yaw_m180_p180 > 0
        assert 0 <= provider.odom_yaw_0_360 <= 360

    def test_yaw_conversion_negative(self, mock_multiprocessing):
        """Test yaw angle conversion for negative angles."""
        _, mock_queue_instance, _, mock_event_instance = mock_multiprocessing

        provider = ConcreteOdomProvider()

        mock_pose_data = MagicMock()
        mock_pose_data.pose.position.x = 0.0
        mock_pose_data.pose.position.y = 0.0
        mock_pose_data.pose.position.z = 0.0
        mock_pose_data.pose.orientation.x = 0.0
        mock_pose_data.pose.orientation.y = 0.0
        mock_pose_data.pose.orientation.z = -math.sin(math.pi / 8)
        mock_pose_data.pose.orientation.w = math.cos(math.pi / 8)
        mock_pose_data.header.stamp.sec = 100
        mock_pose_data.header.stamp.nanosec = 0
        del mock_pose_data.pose.pose

        call_count = [0]

        def side_effect_get(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_pose_data
            raise Exception("Queue timeout")

        def side_effect_is_set():
            return call_count[0] > 0

        mock_queue_instance.get.side_effect = side_effect_get
        mock_event_instance.is_set.side_effect = side_effect_is_set

        with patch("providers.odom_provider_base.time.time", return_value=1000.0):
            provider.process_odom()

        assert provider.odom_yaw_m180_p180 < 0
        assert provider.odom_yaw_0_360 > 0
