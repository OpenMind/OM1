from unittest.mock import MagicMock, patch

import pytest

from providers.unitree_g1_odom_provider import UnitreeG1OdomProvider


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    UnitreeG1OdomProvider.reset()  # type: ignore
    yield
    # Stop the provider before resetting to clean up threads
    try:
        provider = UnitreeG1OdomProvider._instances.get(UnitreeG1OdomProvider)  # type: ignore
        if provider:
            provider._stop_event.set()
            provider.stop()
    except Exception:
        pass
    UnitreeG1OdomProvider.reset()  # type: ignore


@pytest.fixture
def mock_multiprocessing():
    """Mock multiprocessing and threading components."""
    with (
        patch("providers.unitree_g1_odom_provider.mp.Queue") as mock_queue,
        patch("providers.unitree_g1_odom_provider.mp.Process") as mock_process,
        patch("providers.unitree_g1_odom_provider.threading.Thread") as mock_thread,
        patch("providers.unitree_g1_odom_provider.threading.Event") as mock_event,
        patch("providers.odom_provider_base.mp.Queue") as mock_base_queue,
        patch("providers.odom_provider_base.threading.Event") as mock_base_event,
    ):
        mock_queue_instance = MagicMock()
        mock_process_instance = MagicMock()
        mock_thread_instance = MagicMock()
        mock_event_instance = MagicMock()

        mock_queue.return_value = mock_queue_instance
        mock_base_queue.return_value = mock_queue_instance
        mock_process.return_value = mock_process_instance
        mock_thread.return_value = mock_thread_instance
        mock_event.return_value = mock_event_instance
        mock_base_event.return_value = mock_event_instance

        mock_process_instance.is_alive.return_value = False
        mock_thread_instance.is_alive.return_value = False
        mock_event_instance.is_set.return_value = True
        mock_process_instance.join.return_value = None
        mock_thread_instance.join.return_value = None

        yield (
            mock_queue,
            mock_queue_instance,
            mock_process,
            mock_process_instance,
            mock_thread,
            mock_thread_instance,
        )


class TestUnitreeG1OdomProvider:
    """Test cases for UnitreeG1OdomProvider."""

    def test_initialization_with_channel(self, mock_multiprocessing):
        """Test initialization with channel."""
        provider = UnitreeG1OdomProvider(channel="test_channel")

        assert provider.channel == "test_channel"

    def test_initialization_without_channel(self, mock_multiprocessing):
        """Test initialization without channel (None)."""
        provider = UnitreeG1OdomProvider(channel=None)

        assert provider.channel is None

    def test_initialization_default_channel(self, mock_multiprocessing):
        """Test initialization with default channel parameter."""
        provider = UnitreeG1OdomProvider()

        assert provider.channel is None

    def test_singleton_pattern(self, mock_multiprocessing):
        """Test that UnitreeG1OdomProvider follows singleton pattern."""
        provider1 = UnitreeG1OdomProvider(channel="channel_1")
        provider2 = UnitreeG1OdomProvider(channel="channel_2")

        assert provider1 is provider2
        assert provider1.channel == "channel_1"

    def test_start_creates_reader_thread(self, mock_multiprocessing):
        """Test that start creates and starts reader thread."""
        _, _, _, mock_process_instance, _, _ = mock_multiprocessing

        UnitreeG1OdomProvider(channel="test_channel")
        assert mock_process_instance.start.call_count >= 1

    def test_start_creates_processor_thread(self, mock_multiprocessing):
        """Test that start creates and starts processor thread."""
        _, _, _, _, _, mock_thread_instance = mock_multiprocessing

        UnitreeG1OdomProvider(channel="test_channel")
        assert mock_thread_instance.start.call_count >= 1

    def test_start_already_running(self, mock_multiprocessing):
        """Test that start doesn't restart if already running."""
        _, _, _, mock_process_instance, _, mock_thread_instance = mock_multiprocessing

        provider = UnitreeG1OdomProvider(channel="test_channel")

        mock_process_instance.is_alive.return_value = True
        mock_thread_instance.is_alive.return_value = True

        mock_process_instance.start.reset_mock()
        mock_thread_instance.start.reset_mock()

        provider.start()

        mock_process_instance.start.assert_not_called()
        mock_thread_instance.start.assert_not_called()

    def test_start_without_channel(self, mock_multiprocessing, caplog):
        """Test that start logs error when channel is not specified."""
        _, _, _, mock_process_instance, _, mock_thread_instance = mock_multiprocessing

        UnitreeG1OdomProvider(channel=None)

        mock_process_instance.start.assert_not_called()
        mock_thread_instance.start.assert_not_called()

    def test_start_with_processor_thread_running(self, mock_multiprocessing):
        """Test start when processor thread is already running."""
        _, _, _, mock_process_instance, _, mock_thread_instance = mock_multiprocessing

        provider = UnitreeG1OdomProvider(channel="test_channel")

        mock_process_instance.is_alive.return_value = False
        mock_thread_instance.is_alive.return_value = True

        mock_process_instance.start.reset_mock()
        mock_thread_instance.start.reset_mock()

        provider.start()

        mock_process_instance.start.assert_called_once()
        mock_thread_instance.start.assert_not_called()

    def test_stop(self, mock_multiprocessing):
        """Test stop method cleans up resources."""
        _, _, _, mock_process_instance, _, mock_thread_instance = mock_multiprocessing

        provider = UnitreeG1OdomProvider(channel="test_channel")
        provider.stop()

        assert provider._stop_event.set.called  # type: ignore

        mock_process_instance.terminate.assert_called_once()
        mock_process_instance.join.assert_called_once()
        mock_thread_instance.join.assert_called_once()

    def test_position_property(self, mock_multiprocessing):
        """Test position property returns correct data structure."""
        provider = UnitreeG1OdomProvider(channel="test_channel")

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

    def test_initialization_inherits_from_base(self, mock_multiprocessing):
        """Test that initialization properly inherits from OdomProviderBase."""
        provider = UnitreeG1OdomProvider(channel="test_channel")

        # Check base class attributes are initialized
        assert hasattr(provider, "data_queue")
        assert hasattr(provider, "_odom_reader_thread")
        assert hasattr(provider, "_odom_processor_thread")
        assert hasattr(provider, "_stop_event")
        assert hasattr(provider, "x")
        assert hasattr(provider, "y")
        assert hasattr(provider, "moving")

    def test_start_logging_with_channel(self, mock_multiprocessing, caplog):
        """Test that start logs the channel."""
        with caplog.at_level("INFO"):
            UnitreeG1OdomProvider(channel="channel_xyz")

        assert (
            "Starting Unitree G1 Odom Provider on channel: channel_xyz" in caplog.text
        )

    def test_start_logging_without_channel(self, mock_multiprocessing, caplog):
        """Test that start logs error when channel is None."""
        with caplog.at_level("ERROR"):
            UnitreeG1OdomProvider(channel=None)

        assert "Channel must be specified to start the G1 Odom Provider" in caplog.text

    @patch("providers.unitree_g1_odom_provider.get_logging_config")
    def test_start_passes_logging_config(
        self, mock_get_logging_config, mock_multiprocessing
    ):
        """Test that start passes logging config to the processor."""
        _, _, mock_process, _, _, _ = mock_multiprocessing

        mock_logging_config = MagicMock()
        mock_get_logging_config.return_value = mock_logging_config

        UnitreeG1OdomProvider(channel="test_channel")

        call_args = mock_process.call_args
        assert call_args is not None
        args = call_args[1]["args"]
        assert args[2] == mock_logging_config

    def test_multiple_start_calls_with_running_threads(
        self, mock_multiprocessing, caplog
    ):
        """Test multiple start calls when threads are already running."""
        _, _, _, mock_process_instance, _, mock_thread_instance = mock_multiprocessing

        provider = UnitreeG1OdomProvider(channel="test_channel")

        mock_process_instance.is_alive.return_value = True
        mock_thread_instance.is_alive.return_value = True

        with caplog.at_level("WARNING"):
            provider.start()

        assert "G1 Odom Provider is already running" in caplog.text
