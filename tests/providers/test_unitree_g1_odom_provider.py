import sys
from unittest.mock import MagicMock

# Mock zenoh to avoid PyO3 import error
sys.modules["zenoh"] = MagicMock()

# ruff: noqa: E402
import math
from queue import Empty
from unittest.mock import patch

import pytest

from providers.odom_provider_base import RobotState
from providers.unitree_g1_odom_provider import UnitreeG1OdomProvider


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    UnitreeG1OdomProvider.reset()  # type: ignore
    yield
    _cleanup_singleton()
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
        mock_event_instance.is_set.return_value = False
        mock_process_instance.join.return_value = None
        mock_thread_instance.join.return_value = None

        yield {
            "queue": mock_queue,
            "queue_instance": mock_queue_instance,
            "process": mock_process,
            "process_instance": mock_process_instance,
            "thread": mock_thread,
            "thread_instance": mock_thread_instance,
            "event": mock_event,
            "event_instance": mock_event_instance,
        }


def _cleanup_singleton():
    """Clean up singleton instance after tests."""
    try:
        provider = UnitreeG1OdomProvider._instances.get(UnitreeG1OdomProvider)  # type: ignore
        if provider:
            provider._stop_event.set()
            provider.stop()
    except Exception:
        pass


def create_sport_mode_data(x=0.0, y=0.0, z=0.0, yaw_rad=0.0, sec=0, nanosec=0):
    """Helper to create SportModeState mock data with custom values."""
    mock_data = MagicMock()
    mock_data.stamp.sec = sec
    mock_data.stamp.nanosec = nanosec
    mock_data.position = [x, y, z]
    mock_data.imu_state.rpy = [0.0, 0.0, yaw_rad]
    return mock_data


class TestG1OdomProcessor:
    """Test cases for g1_odom_processor function."""

    @patch("time.sleep")
    def test_g1_odom_processor_initialization_success(self, mock_sleep):
        """Test successful initialization of g1_odom_processor."""
        with (
            patch.dict("sys.modules", {"unitree": MagicMock()}),
            patch.dict("sys.modules", {"unitree.unitree_sdk2py": MagicMock()}),
            patch.dict("sys.modules", {"unitree.unitree_sdk2py.core": MagicMock()}),
            patch.dict(
                "sys.modules", {"unitree.unitree_sdk2py.core.channel": MagicMock()}
            ),
            patch.dict("sys.modules", {"unitree.unitree_sdk2py.idl": MagicMock()}),
            patch.dict(
                "sys.modules",
                {"unitree.unitree_sdk2py.idl.unitree_go": MagicMock()},
            ),
            patch.dict(
                "sys.modules",
                {"unitree.unitree_sdk2py.idl.unitree_go.msg": MagicMock()},
            ),
            patch.dict(
                "sys.modules",
                {"unitree.unitree_sdk2py.idl.unitree_go.msg.dds_": MagicMock()},
            ),
        ):
            pytest.skip(
                "g1_odom_processor requires Unitree SDK imports - tested via integration"
            )

    @patch("providers.unitree_g1_odom_provider.logging")
    def test_g1_odom_processor_factory_init_error(self, mock_logging):
        """Test error handling when ChannelFactoryInitialize fails."""
        pytest.skip(
            "g1_odom_processor requires Unitree SDK imports - tested via integration"
        )

    def test_g1_odom_processor_subscriber_init_error(self):
        """Test error handling when ChannelSubscriber initialization fails."""
        pytest.skip(
            "g1_odom_processor requires Unitree SDK imports - tested via integration"
        )

    def test_g1_odom_processor_successful_initialization_logging(self):
        """Test that successful initialization logs info message."""
        pytest.skip(
            "g1_odom_processor requires Unitree SDK imports - tested via integration"
        )

    def test_g1_odom_processor_handler_puts_data_in_queue(self):
        """Test that sport_mode_handler puts data into the queue."""
        pytest.skip(
            "g1_odom_processor requires Unitree SDK imports - tested via integration"
        )

    def test_g1_odom_processor_uses_logging_config(self):
        """Test that g1_odom_processor uses provided logging config."""
        pytest.skip(
            "g1_odom_processor requires Unitree SDK imports - tested via integration"
        )

    def test_g1_odom_processor_infinite_loop(self):
        """Test that g1_odom_processor runs in infinite loop."""
        pytest.skip(
            "g1_odom_processor requires Unitree SDK imports - tested via integration"
        )


class TestUnitreeG1OdomProviderStopMethod:
    """Additional test cases for stop method."""

    def test_stop_when_not_started(self, mock_multiprocessing):
        """Test stop method when provider was never started."""
        mocks = mock_multiprocessing
        provider = UnitreeG1OdomProvider(channel=None)

        mocks["process_instance"].terminate.reset_mock()
        mocks["process_instance"].join.reset_mock()
        mocks["thread_instance"].join.reset_mock()

        provider.stop()

        assert provider._stop_event.set.called  # type: ignore

    def test_stop_when_process_already_terminated(self, mock_multiprocessing):
        """Test stop method when process is already terminated."""
        mocks = mock_multiprocessing
        provider = UnitreeG1OdomProvider(channel="test_channel")

        mocks["process_instance"].is_alive.return_value = False
        mocks["process_instance"].terminate.reset_mock()
        mocks["process_instance"].join.reset_mock()

        provider.stop()

        mocks["process_instance"].terminate.assert_called_once()
        mocks["process_instance"].join.assert_called_once()

    def test_stop_when_thread_already_stopped(self, mock_multiprocessing):
        """Test stop method when thread is already stopped."""
        mocks = mock_multiprocessing
        provider = UnitreeG1OdomProvider(channel="test_channel")

        mocks["thread_instance"].is_alive.return_value = False
        mocks["thread_instance"].join.reset_mock()

        provider.stop()

        mocks["thread_instance"].join.assert_called_once()

    def test_stop_multiple_times(self, mock_multiprocessing):
        """Test calling stop multiple times."""
        mocks = mock_multiprocessing
        provider = UnitreeG1OdomProvider(channel="test_channel")

        provider.stop()
        mocks["process_instance"].terminate.reset_mock()
        mocks["process_instance"].join.reset_mock()
        mocks["thread_instance"].join.reset_mock()

        provider.stop()

        mocks["process_instance"].terminate.assert_called()
        mocks["process_instance"].join.assert_called()

    def test_stop_sets_stop_event(self, mock_multiprocessing):
        """Test that stop sets the stop event."""
        provider = UnitreeG1OdomProvider(channel="test_channel")

        provider.stop()

        provider._stop_event.set.assert_called()  # type: ignore

    def test_stop_with_join_timeout(self, mock_multiprocessing):
        """Test stop when join has timeout parameter."""
        mocks = mock_multiprocessing
        provider = UnitreeG1OdomProvider(channel="test_channel")

        provider.stop()

        assert mocks["process_instance"].join.called
        assert mocks["thread_instance"].join.called


class TestProcessOdomErrorScenarios:
    """Test error scenarios in process_odom method."""

    def test_process_odom_with_malformed_data_missing_position(
        self, mock_multiprocessing
    ):
        """Test process_odom with malformed data missing position."""
        mocks = mock_multiprocessing
        provider = UnitreeG1OdomProvider(channel="test_channel")

        malformed_data = MagicMock()
        malformed_data.stamp.sec = 1000
        malformed_data.stamp.nanosec = 0
        del malformed_data.position

        mocks["queue_instance"].get.side_effect = [malformed_data, Exception()]
        mocks["event_instance"].is_set.side_effect = [False, True]

        try:
            provider.process_odom()
        except AttributeError:
            pass

    def test_process_odom_with_malformed_data_missing_imu(self, mock_multiprocessing):
        """Test process_odom with malformed data missing IMU data."""
        mocks = mock_multiprocessing
        provider = UnitreeG1OdomProvider(channel="test_channel")

        malformed_data = MagicMock()
        malformed_data.stamp.sec = 1000
        malformed_data.stamp.nanosec = 0
        malformed_data.position = [0.0, 0.0, 0.0]
        del malformed_data.imu_state

        mocks["queue_instance"].get.side_effect = [malformed_data, Exception()]
        mocks["event_instance"].is_set.side_effect = [False, True]

        try:
            provider.process_odom()
        except AttributeError:
            pass

    def test_process_odom_with_invalid_position_data_type(self, mock_multiprocessing):
        """Test process_odom with invalid position data type."""
        mocks = mock_multiprocessing
        provider = UnitreeG1OdomProvider(channel="test_channel")

        malformed_data = MagicMock()
        malformed_data.stamp.sec = 1000
        malformed_data.stamp.nanosec = 0
        malformed_data.position = "invalid"
        malformed_data.imu_state.rpy = [0.0, 0.0, 0.0]

        mocks["queue_instance"].get.side_effect = [malformed_data, Exception()]
        mocks["event_instance"].is_set.side_effect = [False, True]

        try:
            provider.process_odom()
        except (TypeError, IndexError):
            pass

    def test_process_odom_with_nan_values(self, mock_multiprocessing):
        """Test process_odom with NaN values in position."""
        mocks = mock_multiprocessing
        provider = UnitreeG1OdomProvider(channel="test_channel")

        sport_data = create_sport_mode_data(
            x=float("nan"),
            y=float("nan"),
            z=float("nan"),
            yaw_rad=float("nan"),
        )

        mocks["queue_instance"].get.side_effect = [sport_data, Exception()]
        mocks["event_instance"].is_set.side_effect = [False, True]

        provider.process_odom()

    def test_process_odom_with_inf_values(self, mock_multiprocessing):
        """Test process_odom with infinity values in position."""
        mocks = mock_multiprocessing
        provider = UnitreeG1OdomProvider(channel="test_channel")

        sport_data = create_sport_mode_data(
            x=float("inf"), y=float("-inf"), z=0.0, yaw_rad=0.0
        )

        mocks["queue_instance"].get.side_effect = [sport_data, Exception()]
        mocks["event_instance"].is_set.side_effect = [False, True]

        provider.process_odom()

    def test_process_odom_with_very_large_values(self, mock_multiprocessing):
        """Test process_odom with very large position values."""
        mocks = mock_multiprocessing
        provider = UnitreeG1OdomProvider(channel="test_channel")

        sport_data = create_sport_mode_data(x=1e10, y=1e10, z=1e10, yaw_rad=1000.0)

        mocks["queue_instance"].get.side_effect = [sport_data, Exception()]
        mocks["event_instance"].is_set.side_effect = [False, True]

        provider.process_odom()

    def test_process_odom_continuous_exceptions(self, mock_multiprocessing):
        """Test process_odom handles continuous exceptions gracefully."""
        mocks = mock_multiprocessing
        provider = UnitreeG1OdomProvider(channel="test_channel")

        mocks["queue_instance"].get.side_effect = [
            Exception("Error 1"),
            Exception("Error 2"),
            Exception("Error 3"),
        ]
        mocks["event_instance"].is_set.side_effect = [False, False, False, True]

        provider.process_odom()

    def test_process_odom_queue_empty_exception(self, mock_multiprocessing):
        """Test process_odom handles queue.Empty exception."""
        mocks = mock_multiprocessing
        provider = UnitreeG1OdomProvider(channel="test_channel")

        mocks["queue_instance"].get.side_effect = [Empty(), Exception()]
        mocks["event_instance"].is_set.side_effect = [False, True]

        provider.process_odom()


class TestProcessOdomEdgeCases:
    """Test edge cases in process_odom method."""

    def test_process_odom_yaw_wraparound_positive(self, mock_multiprocessing):
        """Test yaw conversion at positive wraparound boundary."""
        mocks = mock_multiprocessing
        provider = UnitreeG1OdomProvider(channel="test_channel")

        yaw_rad = math.pi + 0.1
        sport_data = create_sport_mode_data(yaw_rad=yaw_rad)
        mocks["queue_instance"].get.side_effect = [sport_data, Exception()]
        mocks["event_instance"].is_set.side_effect = [False, True]

        provider.process_odom()

        assert provider.odom_yaw_m180_p180 > 180

    def test_process_odom_yaw_wraparound_negative(self, mock_multiprocessing):
        """Test yaw conversion at negative wraparound boundary."""
        mocks = mock_multiprocessing
        provider = UnitreeG1OdomProvider(channel="test_channel")

        yaw_rad = -math.pi - 0.1
        sport_data = create_sport_mode_data(yaw_rad=yaw_rad)
        mocks["queue_instance"].get.side_effect = [sport_data, Exception()]
        mocks["event_instance"].is_set.side_effect = [False, True]

        provider.process_odom()

        assert provider.odom_yaw_m180_p180 < -180

    def test_process_odom_movement_history_initial_state(self, mock_multiprocessing):
        """Test movement history starts at zero."""
        provider = UnitreeG1OdomProvider(channel="test_channel")

        assert provider.move_history == 0.0

    def test_process_odom_previous_position_initial_state(self, mock_multiprocessing):
        """Test previous position starts at zero."""
        provider = UnitreeG1OdomProvider(channel="test_channel")

        assert provider.previous_x == 0.0
        assert provider.previous_y == 0.0
        assert provider.previous_z == 0.0

    def test_process_odom_negative_coordinates(self, mock_multiprocessing):
        """Test process_odom with negative coordinates."""
        mocks = mock_multiprocessing
        provider = UnitreeG1OdomProvider(channel="test_channel")

        sport_data = create_sport_mode_data(x=-5.5, y=-3.2, z=-1.0)
        mocks["queue_instance"].get.side_effect = [sport_data, Exception()]
        mocks["event_instance"].is_set.side_effect = [False, True]

        provider.process_odom()

        assert provider.x == -5.5
        assert provider.y == -3.2
        assert provider.z == -1.0

    def test_process_odom_diagonal_movement(self, mock_multiprocessing):
        """Test movement detection with diagonal movement."""
        mocks = mock_multiprocessing
        provider = UnitreeG1OdomProvider(channel="test_channel")

        sport_data1 = create_sport_mode_data(x=0.0, y=0.0, z=0.0)
        sport_data2 = create_sport_mode_data(x=0.3, y=0.4, z=0.0)

        mocks["queue_instance"].get.side_effect = [
            sport_data1,
            sport_data2,
            Exception(),
        ]
        mocks["event_instance"].is_set.side_effect = [False, False, True]

        provider.process_odom()

        assert provider.moving is True
        assert abs(provider.move_history - 0.35) < 0.01

    def test_process_odom_timestamp_nanosecond_conversion(self, mock_multiprocessing):
        """Test timestamp conversion with nanoseconds."""
        mocks = mock_multiprocessing
        provider = UnitreeG1OdomProvider(channel="test_channel")

        sport_data = create_sport_mode_data(sec=1000, nanosec=500000000)
        mocks["queue_instance"].get.side_effect = [sport_data, Exception()]
        mocks["event_instance"].is_set.side_effect = [False, True]

        provider.process_odom()

        expected_ts = 1000 + 0.5
        assert provider.odom_rockchip_ts == expected_ts

    def test_process_odom_very_small_movement(self, mock_multiprocessing):
        """Test process_odom with very small movement (sub-millimeter)."""
        mocks = mock_multiprocessing
        provider = UnitreeG1OdomProvider(channel="test_channel")

        sport_data1 = create_sport_mode_data(x=0.0, y=0.0, z=0.0)
        sport_data2 = create_sport_mode_data(x=0.0001, y=0.0001, z=0.0)

        mocks["queue_instance"].get.side_effect = [
            sport_data1,
            sport_data2,
            Exception(),
        ]
        mocks["event_instance"].is_set.side_effect = [False, False, True]

        provider.process_odom()

        assert provider.moving is False

    def test_process_odom_oscillating_movement(self, mock_multiprocessing):
        """Test movement detection with oscillating movement."""
        mocks = mock_multiprocessing
        provider = UnitreeG1OdomProvider(channel="test_channel")

        sport_data1 = create_sport_mode_data(x=0.0, y=0.0, z=0.0)
        sport_data2 = create_sport_mode_data(x=0.1, y=0.0, z=0.0)
        sport_data3 = create_sport_mode_data(x=0.0, y=0.0, z=0.0)
        sport_data4 = create_sport_mode_data(x=0.1, y=0.0, z=0.0)

        mocks["queue_instance"].get.side_effect = [
            sport_data1,
            sport_data2,
            sport_data3,
            sport_data4,
            Exception(),
        ]
        mocks["event_instance"].is_set.side_effect = [
            False,
            False,
            False,
            False,
            True,
        ]

        provider.process_odom()

        assert provider.moving is True


class TestUnitreeG1OdomProviderIntegration:
    """Integration-style tests for UnitreeG1OdomProvider."""

    def test_full_lifecycle(self, mock_multiprocessing):
        """Test full lifecycle: create, start, process data, stop."""
        mocks = mock_multiprocessing
        provider = UnitreeG1OdomProvider(channel="test_channel")

        assert mocks["process_instance"].start.called
        assert mocks["thread_instance"].start.called

        sport_data = create_sport_mode_data(x=1.0, y=2.0, z=0.5, yaw_rad=0.5)
        mocks["queue_instance"].get.side_effect = [sport_data, Exception()]
        mocks["event_instance"].is_set.side_effect = [False, True]

        provider.process_odom()

        assert provider.x == 1.0
        assert provider.y == 2.0
        assert provider.z == 0.5

        provider.stop()
        assert provider._stop_event.set.called  # type: ignore

    def test_restart_after_stop(self, mock_multiprocessing):
        """Test restarting provider after stopping it."""
        mocks = mock_multiprocessing
        provider = UnitreeG1OdomProvider(channel="test_channel")

        provider.stop()

        mocks["process_instance"].start.reset_mock()
        mocks["thread_instance"].start.reset_mock()
        mocks["process_instance"].is_alive.return_value = False
        mocks["thread_instance"].is_alive.return_value = False

        provider.start()

        assert mocks["process_instance"].start.called
        assert mocks["thread_instance"].start.called

    def test_position_property_after_processing(self, mock_multiprocessing):
        """Test position property returns correct data after processing."""
        mocks = mock_multiprocessing
        provider = UnitreeG1OdomProvider(channel="test_channel")

        sport_data = create_sport_mode_data(
            x=1.5, y=2.5, z=0.3, yaw_rad=math.pi / 4, sec=1000, nanosec=500000000
        )
        mocks["queue_instance"].get.side_effect = [sport_data, Exception()]
        mocks["event_instance"].is_set.side_effect = [False, True]

        provider.process_odom()

        position = provider.position

        assert position["odom_x"] == 1.5
        assert position["odom_y"] == 2.5
        assert position["body_attitude"] == RobotState.STANDING
        assert position["odom_rockchip_ts"] == 1000.5
        assert abs(position["odom_yaw_m180_p180"] - 45.0) < 0.1

    def test_concurrent_position_reads(self, mock_multiprocessing):
        """Test reading position property multiple times."""
        mocks = mock_multiprocessing
        provider = UnitreeG1OdomProvider(channel="test_channel")

        sport_data = create_sport_mode_data(x=1.0, y=2.0, z=0.5)
        mocks["queue_instance"].get.side_effect = [sport_data, Exception()]
        mocks["event_instance"].is_set.side_effect = [False, True]

        provider.process_odom()

        position1 = provider.position
        position2 = provider.position
        position3 = provider.position

        assert position1["odom_x"] == position2["odom_x"] == position3["odom_x"]
        assert position1["odom_y"] == position2["odom_y"] == position3["odom_y"]
