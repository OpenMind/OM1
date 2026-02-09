import logging
from unittest.mock import MagicMock, patch

import pytest

from providers.unitree_go2_odom_provider import RobotState, UnitreeGo2OdomProvider


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    UnitreeGo2OdomProvider.reset()  # type: ignore
    yield
    UnitreeGo2OdomProvider.reset()  # type: ignore


@pytest.fixture
def mock_multiprocessing():
    with (
        patch("providers.unitree_go2_odom_provider.mp.Queue") as mock_queue,
        patch("providers.unitree_go2_odom_provider.mp.Process") as mock_process,
        patch("providers.unitree_go2_odom_provider.threading.Thread") as mock_thread,
        patch("providers.unitree_go2_odom_provider.threading.Event") as mock_event,
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

        yield mock_queue, mock_queue_instance, mock_process, mock_process_instance, mock_thread, mock_thread_instance


def test_initialization_with_channel(mock_multiprocessing):
    provider = UnitreeGo2OdomProvider(channel="test")
    assert provider.channel == "test"


def test_singleton_pattern(mock_multiprocessing):
    provider1 = UnitreeGo2OdomProvider(channel="test")
    provider2 = UnitreeGo2OdomProvider(channel="test2")
    assert provider1 is provider2


def test_start_processes_and_threads(mock_multiprocessing):
    _, _, _, mock_process_instance, _, mock_thread_instance = mock_multiprocessing
    UnitreeGo2OdomProvider(channel="test")
    assert mock_process_instance.start.call_count >= 1
    assert mock_thread_instance.start.call_count >= 1


def test_start_already_running(mock_multiprocessing):
    _, _, _, mock_process_instance, _, mock_thread_instance = mock_multiprocessing
    provider = UnitreeGo2OdomProvider(channel="test")
    mock_process_instance.is_alive.return_value = True
    mock_thread_instance.is_alive.return_value = True
    mock_process_instance.start.reset_mock()
    mock_thread_instance.start.reset_mock()
    provider.start()
    mock_process_instance.start.assert_not_called()
    mock_thread_instance.start.assert_not_called()


def test_stop_terminates_processes(mock_multiprocessing):
    _, _, _, mock_process_instance, _, mock_thread_instance = mock_multiprocessing
    provider = UnitreeGo2OdomProvider(channel="test")
    provider.stop()
    assert provider._stop_event.set.called  # type: ignore
    mock_process_instance.terminate.assert_called_once()
    mock_process_instance.join.assert_called_once()
    mock_thread_instance.join.assert_called_once()


def test_robot_state_enum_values():
    assert RobotState.STANDING.value == "standing"
    assert RobotState.SITTING.value == "sitting"


def test_position_property_defaults(mock_multiprocessing):
    provider = UnitreeGo2OdomProvider(channel="test")
    position = provider.position
    assert position["odom_x"] == 0.0
    assert position["odom_y"] == 0.0
    assert position["moving"] is False
    assert "odom_yaw_0_360" in position
    assert "odom_yaw_m180_p180" in position
    assert "body_height_cm" in position
    assert "body_attitude" in position
    assert "odom_rockchip_ts" in position
    assert "odom_subscriber_ts" in position


def test_update_body_state_standing(mock_multiprocessing):
    provider = UnitreeGo2OdomProvider(channel="test")
    mock_pose = MagicMock()
    mock_pose.position.z = 0.25  # 25 cm
    provider._update_body_state(mock_pose)
    assert provider.body_height_cm == 25
    assert provider.body_attitude == RobotState.STANDING


def test_update_body_state_sitting(mock_multiprocessing):
    provider = UnitreeGo2OdomProvider(channel="test")
    mock_pose = MagicMock()
    mock_pose.position.z = 0.15  # 15 cm
    provider._update_body_state(mock_pose)
    assert provider.body_height_cm == 15
    assert provider.body_attitude == RobotState.SITTING


def test_update_body_state_lying_down(mock_multiprocessing):
    provider = UnitreeGo2OdomProvider(channel="test")
    mock_pose = MagicMock()
    mock_pose.position.z = 0.02  # 2 cm
    provider._update_body_state(mock_pose)
    assert provider.body_height_cm == 2


def test_start_already_running_logs_warning(caplog, mock_multiprocessing):
    _, _, _, mock_process_instance, _, mock_thread_instance = mock_multiprocessing
    provider = UnitreeGo2OdomProvider(channel="test")
    mock_process_instance.is_alive.return_value = True
    mock_thread_instance.is_alive.return_value = True
    with caplog.at_level(logging.WARNING):
        provider.start()
        assert "Go2 Odom Provider is already running." in caplog.text


def test_start_processor_already_running_logs_warning(caplog, mock_multiprocessing):
    _, _, _, mock_process_instance, _, mock_thread_instance = mock_multiprocessing
    provider = UnitreeGo2OdomProvider(channel="test")
    provider._odom_processor_thread = MagicMock()
    provider._odom_processor_thread.is_alive.return_value = True
    with caplog.at_level(logging.WARNING):
        provider.start()
        assert "Odom processor thread is already running." in caplog.text
