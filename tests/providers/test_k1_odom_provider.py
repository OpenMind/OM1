import math
import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from providers.k1_odom_provider import K1OdomProvider, RobotState, k1_odom_processor, rad_to_deg


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    K1OdomProvider.reset()  # type: ignore
    yield
    K1OdomProvider.reset()  # type: ignore


@pytest.fixture
def mock_multiprocessing():
    """
    Mock multiprocessing and threading components.

    Returns
    -------
    tuple
        Tuple containing mock instances for multiprocessing components.
    """
    with (
        patch("providers.k1_odom_provider.mp.Queue") as mock_queue,
        patch("providers.k1_odom_provider.mp.Process") as mock_process,
        patch("providers.k1_odom_provider.threading.Thread") as mock_thread,
        patch("providers.k1_odom_provider.threading.Event") as mock_event,
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


class TestK1OdomProvider:
    """Test cases for K1OdomProvider."""

    def test_initialization_with_default_topic(self, mock_multiprocessing):
        """Test initialization with default topic."""
        provider = K1OdomProvider()

        assert provider.topic == "odometer_state"

    def test_initialization_with_custom_topic(self, mock_multiprocessing):
        """Test initialization with custom topic."""
        provider = K1OdomProvider(topic="custom_topic")

        assert provider.topic == "custom_topic"

    def test_singleton_pattern(self, mock_multiprocessing):
        """Test that K1OdomProvider follows singleton pattern."""
        provider1 = K1OdomProvider(topic="topic_1")
        provider2 = K1OdomProvider(topic="topic_2")

        assert provider1 is provider2
        # First instance topic should be preserved
        assert provider1.topic == "topic_1"

    def test_initialization_starts_processes(self, mock_multiprocessing):
        """Test that initialization starts reader and processor threads."""
        _, _, _, mock_process_instance, _, mock_thread_instance = mock_multiprocessing

        K1OdomProvider()

        # Should have started the process and thread during initialization
        assert mock_process_instance.start.call_count >= 1
        assert mock_thread_instance.start.call_count >= 1


class TestStart:
    """Test start method."""

    def test_start_creates_reader_process(self, mock_multiprocessing):
        """Test that start creates and starts reader process."""
        _, _, _, mock_process_instance, _, _ = mock_multiprocessing

        K1OdomProvider(topic="test_topic")

        # Should have started the process during initialization
        assert mock_process_instance.start.call_count >= 1

    def test_start_creates_processor_thread(self, mock_multiprocessing):
        """Test that start creates and starts processor thread."""
        _, _, _, _, _, mock_thread_instance = mock_multiprocessing

        K1OdomProvider(topic="test_topic")

        assert mock_thread_instance.start.call_count >= 1

    def test_start_already_running_reader(self, mock_multiprocessing):
        """Test that start doesn't restart reader if already running."""
        _, _, _, mock_process_instance, _, _ = mock_multiprocessing

        provider = K1OdomProvider(topic="test_topic")

        mock_process_instance.is_alive.return_value = True
        initial_start_count = mock_process_instance.start.call_count

        provider.start()

        assert mock_process_instance.start.call_count == initial_start_count

    def test_start_already_running_processor(self, mock_multiprocessing):
        """Test that start doesn't restart processor if already running."""
        _, _, _, _, _, mock_thread_instance = mock_multiprocessing

        provider = K1OdomProvider(topic="test_topic")

        mock_thread_instance.is_alive.return_value = True
        mock_thread_instance.start.call_count

        provider.start()

    def test_start_without_topic(self, mock_multiprocessing):
        """Test that start logs error when topic is not specified."""
        _, _, _, mock_process_instance, _, _ = mock_multiprocessing

        with patch("providers.k1_odom_provider.logging") as mock_logging:
            provider = K1OdomProvider()
            provider.topic = None  # type: ignore  # Simulate missing topic

            provider.start()

            mock_logging.error.assert_called()


class TestUpdateBodyState:
    """Test _update_body_state method."""

    def test_update_body_state_sets_standing(self, mock_multiprocessing):
        """Test _update_body_state sets robot to standing."""
        provider = K1OdomProvider()

        provider._update_body_state(None)

        assert provider.body_attitude == RobotState.STANDING
        assert provider.body_height_cm == 70

    def test_update_body_state_with_any_pose(self, mock_multiprocessing):
        """Test _update_body_state always assumes standing for K1."""
        provider = K1OdomProvider()

        # Test with mock pose data
        mock_pose = Mock()
        provider._update_body_state(mock_pose)

        assert provider.body_attitude == RobotState.STANDING
        assert provider.body_height_cm == 70


class TestProcessOdom:
    """Test process_odom method."""

    def test_process_odom_with_valid_data(self, mock_multiprocessing):
        """Test process_odom processes valid odometry data."""
        _, mock_queue_instance, _, _, _, _ = mock_multiprocessing

        provider = K1OdomProvider()

        # Prepare test data
        test_odom_data = {"x": 1.5, "y": 2.5, "theta": 0.5, "timestamp": time.time()}

        mock_queue_instance.get.return_value = test_odom_data

        with patch("providers.k1_odom_provider.time.time", return_value=1234567890.0):
            # Simulate one iteration of process_odom
            provider._stop_event.is_set.return_value = False  # type: ignore
            provider._stop_event.is_set.side_effect = [False, True]  # type: ignore  # Run once then stop

            provider.process_odom()

        # Verify data was processed
        assert provider.x == round(1.5, 4)
        assert provider.y == round(2.5, 4)

    def test_process_odom_calculates_movement(self, mock_multiprocessing):
        """Test process_odom correctly calculates movement delta."""
        _, mock_queue_instance, _, _, _, _ = mock_multiprocessing

        provider = K1OdomProvider()

        # First position
        test_odom_data1 = {"x": 0.0, "y": 0.0, "theta": 0.0, "timestamp": time.time()}
        # Second position (moved 0.1m in x direction)
        test_odom_data2 = {"x": 0.1, "y": 0.0, "theta": 0.0, "timestamp": time.time()}

        mock_queue_instance.get.side_effect = [test_odom_data1, test_odom_data2]

        with patch("providers.k1_odom_provider.time.time", return_value=1234567890.0):
            # Simulate two iterations
            provider._stop_event.is_set.side_effect = [False, False, True]  # type: ignore

            provider.process_odom()

        # Should detect movement
        assert provider.moving is True

    def test_process_odom_stationary_robot(self, mock_multiprocessing):
        """Test process_odom detects stationary robot."""
        _, mock_queue_instance, _, _, _, _ = mock_multiprocessing

        provider = K1OdomProvider()

        # Same position data - need multiple iterations for move_history to decay
        test_odom_data = {"x": 1.0, "y": 1.0, "theta": 0.0, "timestamp": time.time()}

        # Simulate multiple iterations with same position
        mock_queue_instance.get.side_effect = [test_odom_data] * 5

        with patch("providers.k1_odom_provider.time.time", return_value=1234567890.0):
            # Simulate multiple iterations for move_history to decay below threshold
            provider._stop_event.is_set.side_effect = [False] * 5 + [True]  # type: ignore

            provider.process_odom()

        # Should detect no movement after decay
        assert provider.moving is False
        assert provider.move_history < 0.01

    def test_process_odom_converts_theta_to_degrees(self, mock_multiprocessing):
        """Test process_odom converts theta from radians to degrees."""
        _, mock_queue_instance, _, _, _, _ = mock_multiprocessing

        provider = K1OdomProvider()

        # theta = pi/2 radians = 90 degrees
        test_odom_data = {"x": 0.0, "y": 0.0, "theta": math.pi / 2, "timestamp": time.time()}

        mock_queue_instance.get.return_value = test_odom_data

        with patch("providers.k1_odom_provider.time.time", return_value=1234567890.0):
            provider._stop_event.is_set.side_effect = [False, True]  # type: ignore

            provider.process_odom()

        # Should be approximately 90 degrees
        assert abs(provider.odom_yaw_m180_p180 - 90.0) < 0.1

    def test_process_odom_normalizes_yaw_to_m180_p180(self, mock_multiprocessing):
        """Test process_odom normalizes yaw to [-180, 180] range."""
        _, mock_queue_instance, _, _, _, _ = mock_multiprocessing

        provider = K1OdomProvider()

        # theta = 3.5 radians = ~200 degrees, should normalize to -160
        test_odom_data = {"x": 0.0, "y": 0.0, "theta": 3.5, "timestamp": time.time()}

        mock_queue_instance.get.return_value = test_odom_data

        with patch("providers.k1_odom_provider.time.time", return_value=1234567890.0):
            provider._stop_event.is_set.side_effect = [False, True]  # type: ignore

            provider.process_odom()

        # Should be in [-180, 180] range
        assert -180.0 <= provider.odom_yaw_m180_p180 <= 180.0

    def test_process_odom_provides_0_360_representation(self, mock_multiprocessing):
        """Test process_odom provides [0, 360] yaw representation."""
        _, mock_queue_instance, _, _, _, _ = mock_multiprocessing

        provider = K1OdomProvider()

        test_odom_data = {"x": 0.0, "y": 0.0, "theta": -math.pi / 2, "timestamp": time.time()}

        mock_queue_instance.get.return_value = test_odom_data

        with patch("providers.k1_odom_provider.time.time", return_value=1234567890.0):
            provider._stop_event.is_set.side_effect = [False, True]  # type: ignore

            provider.process_odom()

        # Should be in [0, 360] range
        assert 0.0 <= provider.odom_yaw_0_360 <= 360.0

    def test_process_odom_updates_timestamps(self, mock_multiprocessing):
        """Test process_odom updates timestamps correctly."""
        _, mock_queue_instance, _, _, _, _ = mock_multiprocessing

        provider = K1OdomProvider()

        test_timestamp = 9999.0
        test_odom_data = {"x": 0.0, "y": 0.0, "theta": 0.0, "timestamp": test_timestamp}

        mock_queue_instance.get.return_value = test_odom_data

        with patch("providers.k1_odom_provider.time.time", return_value=1234567890.0):
            provider._stop_event.is_set.side_effect = [False, True]  # type: ignore

            provider.process_odom()

        assert provider.odom_subscriber_ts == 1234567890.0
        assert provider.odom_rockchip_ts == test_timestamp

    def test_process_odom_handles_queue_timeout(self, mock_multiprocessing):
        """Test process_odom handles queue timeout gracefully."""
        _, mock_queue_instance, _, _, _, _ = mock_multiprocessing

        provider = K1OdomProvider()

        # Simulate queue timeout
        from queue import Empty

        mock_queue_instance.get.side_effect = Empty()

        provider._stop_event.is_set.side_effect = [False, True]  # type: ignore

        # Should not raise exception
        provider.process_odom()

    def test_process_odom_handles_invalid_data(self, mock_multiprocessing):
        """Test process_odom handles invalid data types."""
        _, mock_queue_instance, _, _, _, _ = mock_multiprocessing

        provider = K1OdomProvider()

        # Invalid data type (not a dict)
        mock_queue_instance.get.return_value = "invalid_data"

        with patch("providers.k1_odom_provider.logging") as mock_logging:
            provider._stop_event.is_set.side_effect = [False, True]  # type: ignore

            provider.process_odom()

            # Should log warning about unexpected data type
            mock_logging.warning.assert_called()

    def test_process_odom_uses_move_history_decay(self, mock_multiprocessing):
        """Test process_odom uses decay kernel for move history."""
        _, mock_queue_instance, _, _, _, _ = mock_multiprocessing

        provider = K1OdomProvider()

        # First with movement
        test_odom_data1 = {"x": 0.0, "y": 0.0, "theta": 0.0, "timestamp": time.time()}
        test_odom_data2 = {"x": 0.1, "y": 0.0, "theta": 0.0, "timestamp": time.time()}

        mock_queue_instance.get.side_effect = [test_odom_data1, test_odom_data2]

        with patch("providers.k1_odom_provider.time.time", return_value=1234567890.0):
            provider._stop_event.is_set.side_effect = [False, False, True]  # type: ignore

            provider.process_odom()

        # move_history should be non-zero due to decay kernel
        assert provider.move_history > 0.0


class TestK1OdomProcessor:
    """Test k1_odom_processor function."""

    def test_k1_odom_processor_opens_zenoh_session(self):
        """Test k1_odom_processor opens a Zenoh session."""
        mock_queue = MagicMock()

        with (
            patch("providers.k1_odom_provider.open_zenoh_session") as mock_zenoh,
            patch("providers.k1_odom_provider.setup_logging"),
            patch("providers.k1_odom_provider.time.sleep") as mock_sleep,
        ):
            mock_session = Mock()
            mock_zenoh.return_value = mock_session
            mock_sleep.side_effect = KeyboardInterrupt()  # Stop the infinite loop

            try:
                k1_odom_processor("test_topic", mock_queue)
            except KeyboardInterrupt:
                pass

            mock_zenoh.assert_called_once()
            mock_session.declare_subscriber.assert_called_once()

    def test_k1_odom_processor_subscribes_to_topic(self):
        """Test k1_odom_processor subscribes to the correct topic."""
        mock_queue = MagicMock()

        with (
            patch("providers.k1_odom_provider.open_zenoh_session") as mock_zenoh,
            patch("providers.k1_odom_provider.setup_logging"),
            patch("providers.k1_odom_provider.time.sleep") as mock_sleep,
        ):
            mock_session = Mock()
            mock_zenoh.return_value = mock_session
            mock_sleep.side_effect = KeyboardInterrupt()

            try:
                k1_odom_processor("my_custom_topic", mock_queue)
            except KeyboardInterrupt:
                pass

            # Verify subscriber was declared with the correct topic
            args, _ = mock_session.declare_subscriber.call_args
            assert args[0] == "my_custom_topic"

    def test_k1_odom_processor_handles_zenoh_error(self):
        """Test k1_odom_processor handles Zenoh connection errors."""
        mock_queue = MagicMock()

        with (
            patch("providers.k1_odom_provider.open_zenoh_session") as mock_zenoh,
            patch("providers.k1_odom_provider.setup_logging"),
            patch("providers.k1_odom_provider.logging") as mock_logging,
        ):
            mock_zenoh.side_effect = Exception("Connection failed")

            result = k1_odom_processor("test_topic", mock_queue)

            assert result is None
            mock_logging.error.assert_called()

    def test_k1_odom_processor_sets_up_logging(self):
        """Test k1_odom_processor sets up logging."""
        mock_queue = MagicMock()

        with (
            patch("providers.k1_odom_provider.open_zenoh_session"),
            patch("providers.k1_odom_provider.setup_logging") as mock_setup_logging,
            patch("providers.k1_odom_provider.time.sleep") as mock_sleep,
        ):
            mock_sleep.side_effect = KeyboardInterrupt()

            try:
                k1_odom_processor("test_topic", mock_queue, logging_config=None)
            except KeyboardInterrupt:
                pass

            mock_setup_logging.assert_called_once_with("k1_odom_processor", logging_config=None)


class TestPosition:
    """Test position property."""

    def test_position_property_returns_dict(self, mock_multiprocessing):
        """Test position property returns a dictionary with odometry data."""
        provider = K1OdomProvider()

        position = provider.position

        assert isinstance(position, dict)
        assert "odom_x" in position
        assert "odom_y" in position
        assert "odom_yaw_m180_p180" in position
        assert "odom_yaw_0_360" in position
        assert "moving" in position
        assert "body_attitude" in position

    def test_position_property_values(self, mock_multiprocessing):
        """Test position property returns correct values."""
        provider = K1OdomProvider()

        # Set some values
        provider.x = 5.0
        provider.y = 10.0
        provider.odom_yaw_m180_p180 = 45.0
        provider.odom_yaw_0_360 = 45.0
        provider.moving = True
        provider.body_attitude = RobotState.STANDING

        position = provider.position

        assert position["odom_x"] == 5.0
        assert position["odom_y"] == 10.0
        assert position["odom_yaw_m180_p180"] == 45.0
        assert position["odom_yaw_0_360"] == 45.0
        assert position["moving"] is True
        assert position["body_attitude"] == RobotState.STANDING


class TestRadToDegConstant:
    """Test rad_to_deg constant."""

    def test_rad_to_deg_value(self):
        """Test rad_to_deg constant has correct value."""
        # rad_to_deg should be approximately 180/pi = 57.2958
        expected = 180.0 / math.pi
        assert abs(rad_to_deg - expected) < 0.01

    def test_rad_to_deg_conversion(self):
        """Test rad_to_deg constant works for conversion."""
        # Test converting pi radians to degrees
        radians = math.pi
        degrees = radians * rad_to_deg

        assert abs(degrees - 180.0) < 0.1


class TestIntegration:
    """Integration tests for K1OdomProvider."""

    def test_provider_initialization_and_start(self, mock_multiprocessing):
        """Test complete initialization and start sequence."""
        _, _, mock_process_class, mock_process_instance, mock_thread_class, mock_thread_instance = mock_multiprocessing

        provider = K1OdomProvider(topic="integration_test")

        # Verify initialization - body_attitude is None until odom is processed
        assert provider.topic == "integration_test"
        assert provider.body_attitude is None  # Not set until process_odom runs
        assert provider.body_height_cm == 0  # Not set until _update_body_state is called
        assert provider.x == 0.0
        assert provider.y == 0.0
        assert provider.moving is False

        # Verify processes were started
        assert mock_process_instance.start.call_count >= 1
        assert mock_thread_instance.start.call_count >= 1

    def test_complete_odom_processing_cycle(self, mock_multiprocessing):
        """Test a complete odometry processing cycle."""
        _, mock_queue_instance, _, _, _, _ = mock_multiprocessing

        provider = K1OdomProvider(topic="test_topic")

        # Simulate receiving odometry data
        test_data = {"x": 1.0, "y": 2.0, "theta": math.pi / 4, "timestamp": 12345.0}

        mock_queue_instance.get.return_value = test_data

        with patch("providers.k1_odom_provider.time.time", return_value=1234567890.0):
            provider._stop_event.is_set.side_effect = [False, True]  # type: ignore

            provider.process_odom()

        # Verify all data was processed correctly
        assert provider.x == 1.0
        assert provider.y == 2.0
        assert provider.body_attitude == RobotState.STANDING
        assert provider.odom_subscriber_ts == 1234567890.0
        assert provider.odom_rockchip_ts == 12345.0

        # Verify position property returns correct values
        position = provider.position
        assert position["odom_x"] == 1.0
        assert position["odom_y"] == 2.0
        assert position["body_attitude"] == RobotState.STANDING
