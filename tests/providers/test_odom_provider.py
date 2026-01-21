from unittest.mock import MagicMock, patch

import pytest

from providers.odom_provider import OdomProvider, RobotState


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    OdomProvider.reset()  # type: ignore
    yield
    OdomProvider.reset()  # type: ignore


@pytest.fixture
def mock_multiprocessing():
    with (
        patch("providers.odom_provider.mp.Queue") as mock_queue,
        patch("providers.odom_provider.mp.Process") as mock_process,
        patch("providers.odom_provider.threading.Thread") as mock_thread,
        patch("providers.odom_provider.threading.Event") as mock_event,
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


def test_initialization_zenoh(mock_multiprocessing):
    mock_queue, _, _, _, _, _ = mock_multiprocessing

    provider = OdomProvider(channel="test", use_zenoh=True)

    assert provider.channel == "test"
    assert provider.use_zenoh is True
    assert provider.URID == ""
    mock_queue.assert_called_once()


def test_initialization_cyclonedds(mock_multiprocessing):
    provider = OdomProvider(channel="test", use_zenoh=False)

    assert provider.channel == "test"
    assert provider.use_zenoh is False
    assert provider.URID == ""


def test_singleton_pattern(mock_multiprocessing):
    provider1 = OdomProvider(channel="test")
    provider2 = OdomProvider(channel="test2")
    assert provider1 is provider2


def test_start(mock_multiprocessing):
    _, _, _, mock_process_instance, _, mock_thread_instance = mock_multiprocessing

    OdomProvider(channel="test")

    assert mock_process_instance.start.call_count >= 1
    assert mock_thread_instance.start.call_count >= 1


def test_start_already_running(mock_multiprocessing):
    _, _, _, mock_process_instance, _, mock_thread_instance = mock_multiprocessing

    provider = OdomProvider(channel="test")

    mock_process_instance.is_alive.return_value = True
    mock_thread_instance.is_alive.return_value = True

    mock_process_instance.start.reset_mock()
    mock_thread_instance.start.reset_mock()

    provider.start()

    mock_process_instance.start.assert_not_called()
    mock_thread_instance.start.assert_not_called()


def test_stop(mock_multiprocessing):
    _, _, _, mock_process_instance, _, mock_thread_instance = mock_multiprocessing

    provider = OdomProvider(channel="test")
    provider.stop()

    assert provider._stop_event.set.called  # type: ignore
    mock_process_instance.terminate.assert_called_once()
    mock_process_instance.join.assert_called_once()
    mock_thread_instance.join.assert_called_once()


def test_robot_state_enum():
    assert RobotState.STANDING.value == "standing"
    assert RobotState.SITTING.value == "sitting"


def test_position_property(mock_multiprocessing):
    provider = OdomProvider(channel="test")

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
