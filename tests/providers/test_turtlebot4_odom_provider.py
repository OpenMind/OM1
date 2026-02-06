from unittest.mock import MagicMock, patch

import pytest

from providers.turtlebot4_odom_provider import TurtleBot4OdomProvider


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    TurtleBot4OdomProvider.reset()  # type: ignore
    yield
    TurtleBot4OdomProvider.reset()  # type: ignore


@pytest.fixture
def mock_multiprocessing():
    """Mock multiprocessing and threading components."""
    with (
        patch("providers.turtlebot4_odom_provider.mp.Queue") as mock_queue,
        patch("providers.turtlebot4_odom_provider.mp.Process") as mock_process,
        patch("providers.turtlebot4_odom_provider.threading.Thread") as mock_thread,
        patch("providers.turtlebot4_odom_provider.threading.Event") as mock_event,
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


class TestTurtleBot4OdomProvider:
    """Test cases for TurtleBot4OdomProvider."""

    def test_initialization_with_urid(self, mock_multiprocessing):
        """Test initialization with URID."""
        provider = TurtleBot4OdomProvider(URID="test_robot_123")

        assert provider.URID == "test_robot_123"

    def test_initialization_without_urid(self, mock_multiprocessing):
        """Test initialization without URID (None)."""
        provider = TurtleBot4OdomProvider(URID=None)

        assert provider.URID is None

    def test_initialization_default_urid(self, mock_multiprocessing):
        """Test initialization with default URID parameter."""
        provider = TurtleBot4OdomProvider()

        assert provider.URID is None

    def test_singleton_pattern(self, mock_multiprocessing):
        """Test that TurtleBot4OdomProvider follows singleton pattern."""
        provider1 = TurtleBot4OdomProvider(URID="robot_1")
        provider2 = TurtleBot4OdomProvider(URID="robot_2")

        assert provider1 is provider2
        # First instance URID should be preserved
        assert provider1.URID == "robot_1"

    def test_start_creates_reader_thread(self, mock_multiprocessing):
        """Test that start creates and starts reader thread."""
        _, _, _, mock_process_instance, _, _ = mock_multiprocessing

        TurtleBot4OdomProvider(URID="test_robot")

        # Should have started the process during initialization
        assert mock_process_instance.start.call_count >= 1

    def test_start_creates_processor_thread(self, mock_multiprocessing):
        """Test that start creates and starts processor thread."""
        _, _, _, _, _, mock_thread_instance = mock_multiprocessing

        TurtleBot4OdomProvider(URID="test_robot")

        # Should have started the thread during initialization
        assert mock_thread_instance.start.call_count >= 1

    def test_start_already_running(self, mock_multiprocessing):
        """Test that start doesn't restart if already running."""
        _, _, _, mock_process_instance, _, mock_thread_instance = mock_multiprocessing

        provider = TurtleBot4OdomProvider(URID="test_robot")

        # Simulate threads already running
        mock_process_instance.is_alive.return_value = True
        mock_thread_instance.is_alive.return_value = True

        # Reset call counts
        mock_process_instance.start.reset_mock()
        mock_thread_instance.start.reset_mock()

        # Call start again
        provider.start()

        # Should not have started new threads
        mock_process_instance.start.assert_not_called()
        mock_thread_instance.start.assert_not_called()

    def test_start_with_processor_thread_running(self, mock_multiprocessing):
        """Test start when processor thread is already running."""
        _, _, _, mock_process_instance, _, mock_thread_instance = mock_multiprocessing

        provider = TurtleBot4OdomProvider(URID="test_robot")

        # Simulate only processor thread running
        mock_process_instance.is_alive.return_value = False
        mock_thread_instance.is_alive.return_value = True

        mock_process_instance.start.reset_mock()
        mock_thread_instance.start.reset_mock()

        provider.start()

        # Reader thread should start, processor should not
        mock_process_instance.start.assert_called_once()
        mock_thread_instance.start.assert_not_called()

    def test_stop(self, mock_multiprocessing):
        """Test stop method cleans up resources."""
        _, _, _, mock_process_instance, _, mock_thread_instance = mock_multiprocessing

        provider = TurtleBot4OdomProvider(URID="test_robot")
        provider.stop()

        # Stop event should be set
        assert provider._stop_event.set.called  # type: ignore

        # Threads should be terminated/joined
        mock_process_instance.terminate.assert_called_once()
        mock_process_instance.join.assert_called_once()
        mock_thread_instance.join.assert_called_once()

    def test_position_property(self, mock_multiprocessing):
        """Test position property returns correct data structure."""
        provider = TurtleBot4OdomProvider(URID="test_robot")

        position = provider.position

        # Verify all expected keys are present
        assert "odom_x" in position
        assert "odom_y" in position
        assert "moving" in position
        assert "odom_yaw_0_360" in position
        assert "odom_yaw_m180_p180" in position
        assert "body_height_cm" in position
        assert "body_attitude" in position
        assert "odom_rockchip_ts" in position
        assert "odom_subscriber_ts" in position

        # Verify initial values
        assert position["odom_x"] == 0.0
        assert position["odom_y"] == 0.0
        assert position["moving"] is False

    def test_initialization_inherits_from_base(self, mock_multiprocessing):
        """Test that initialization properly inherits from OdomProviderBase."""
        provider = TurtleBot4OdomProvider(URID="test_robot")

        # Check base class attributes are initialized
        assert hasattr(provider, "data_queue")
        assert hasattr(provider, "_odom_reader_thread")
        assert hasattr(provider, "_odom_processor_thread")
        assert hasattr(provider, "_stop_event")
        assert hasattr(provider, "x")
        assert hasattr(provider, "y")
        assert hasattr(provider, "moving")

    def test_start_logging_with_urid(self, mock_multiprocessing, caplog):
        """Test that start logs the URID."""
        with caplog.at_level("INFO"):
            TurtleBot4OdomProvider(URID="robot_xyz")

        assert "Starting TurtleBot4 Odom Provider with URID: robot_xyz" in caplog.text

    def test_start_logging_without_urid(self, mock_multiprocessing, caplog):
        """Test that start logs when URID is None."""
        with caplog.at_level("INFO"):
            TurtleBot4OdomProvider(URID=None)

        assert "Starting TurtleBot4 Odom Provider with URID: None" in caplog.text

    @patch("providers.turtlebot4_odom_provider.get_logging_config")
    def test_start_passes_logging_config(
        self, mock_get_logging_config, mock_multiprocessing
    ):
        """Test that start passes logging config to the processor."""
        _, _, mock_process, _, _, _ = mock_multiprocessing

        mock_logging_config = MagicMock()
        mock_get_logging_config.return_value = mock_logging_config

        TurtleBot4OdomProvider(URID="test_robot")

        # Verify Process was called with logging config
        call_args = mock_process.call_args
        assert call_args is not None
        args = call_args[1]["args"]
        assert args[2] == mock_logging_config

    def test_multiple_start_calls_with_running_threads(
        self, mock_multiprocessing, caplog
    ):
        """Test multiple start calls when threads are already running."""
        _, _, _, mock_process_instance, _, mock_thread_instance = mock_multiprocessing

        provider = TurtleBot4OdomProvider(URID="test_robot")

        # Simulate both threads running
        mock_process_instance.is_alive.return_value = True
        mock_thread_instance.is_alive.return_value = True

        with caplog.at_level("WARNING"):
            provider.start()

        assert "TurtleBot4 Odom Provider is already running" in caplog.text
