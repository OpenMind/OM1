from unittest.mock import MagicMock, patch

import pytest

from providers.unitree_go2_odom_zenoh_provider import UnitreeGo2OdomZenohProvider


@pytest.fixture(autouse=True)
def reset_singleton():
    UnitreeGo2OdomZenohProvider.reset()  # type: ignore
    yield
    UnitreeGo2OdomZenohProvider.reset()  # type: ignore


@pytest.fixture
def patches():
    with (
        patch("providers.unitree_go2_odom_zenoh_provider.mp.Process") as mock_process_class,
        patch("providers.unitree_go2_odom_zenoh_provider.threading.Thread") as mock_thread_class,
    ):
        proc = MagicMock()
        proc.is_alive.return_value = True
        mock_process_class.return_value = proc
        the = MagicMock()
        the.is_alive.return_value = True
        mock_thread_class.return_value = the
        yield {
            "process_class": mock_process_class,
            "process": proc,
            "thread_class": mock_thread_class,
            "thread": the,
        }


def test_initialization(patches):
    provider = UnitreeGo2OdomZenohProvider()
    assert provider.topic == "utlidar/robot_pose"
    assert provider.api_key is None
    assert provider.use_sim is False
    patches["process"].start.assert_called_once()
    patches["thread"].start.assert_called_once()


def test_initialization_custom(patches):
    UnitreeGo2OdomZenohProvider.reset()  # type: ignore[attr-defined]
    provider = UnitreeGo2OdomZenohProvider(api_key="k", topic="odom", use_sim=True)
    assert provider.topic == "odom"
    assert provider.api_key == "k"
    assert provider.use_sim is True


def test_start_is_idempotent(patches):
    provider = UnitreeGo2OdomZenohProvider()
    initial_process_calls = patches["process_class"].call_count
    initial_thread_calls = patches["thread_class"].call_count
    provider.start()
    # Should not spawn new ones because reader_proc and processor_thread are alive
    assert patches["process_class"].call_count == initial_process_calls
    assert patches["thread_class"].call_count == initial_thread_calls


def test_update_body_state_standing(patches):
    from providers.odom_provider_base import RobotState

    provider = UnitreeGo2OdomZenohProvider()
    pose = MagicMock()
    pose.position.z = 0.30  # cm > 24
    provider._update_body_state(pose)
    assert provider.body_attitude is RobotState.STANDING
    assert provider.body_height_cm == 30


def test_update_body_state_sitting(patches):
    from providers.odom_provider_base import RobotState

    provider = UnitreeGo2OdomZenohProvider()
    pose = MagicMock()
    pose.position.z = 0.10  # 10 cm > 3 but < 24
    provider._update_body_state(pose)
    assert provider.body_attitude is RobotState.SITTING


def test_processor_function_subscribes(monkeypatch):
    """Run _go2_odom_zenoh_processor directly and verify it subscribes."""
    import multiprocessing as mp

    from providers.unitree_go2_odom_zenoh_provider import (
        _go2_odom_zenoh_processor,
    )

    session = MagicMock()
    with (
        patch(
            "providers.unitree_go2_odom_zenoh_provider.open_zenoh_session",
            return_value=session,
        ),
        patch(
            "providers.unitree_go2_odom_zenoh_provider.load_session_config",
        ),
        patch(
            "providers.unitree_go2_odom_zenoh_provider.setup_logging",
        ),
        patch("providers.unitree_go2_odom_zenoh_provider.threading.Event") as mock_event_cls,
    ):
        # threading.Event().wait() blocks forever; make it return immediately.
        mock_event = MagicMock()
        mock_event.wait.return_value = None
        mock_event_cls.return_value = mock_event

        data_queue: mp.Queue = mp.Queue()
        _go2_odom_zenoh_processor(None, "odom", False, data_queue)

    session.declare_subscriber.assert_called_once()


def test_processor_session_failure_exits_cleanly():
    import multiprocessing as mp

    from providers.unitree_go2_odom_zenoh_provider import (
        _go2_odom_zenoh_processor,
    )

    with (
        patch(
            "providers.unitree_go2_odom_zenoh_provider.open_zenoh_session",
            side_effect=RuntimeError("fail"),
        ),
        patch(
            "providers.unitree_go2_odom_zenoh_provider.load_session_config",
        ),
        patch(
            "providers.unitree_go2_odom_zenoh_provider.setup_logging",
        ),
    ):
        data_queue: mp.Queue = mp.Queue()
        # Should not raise
        _go2_odom_zenoh_processor(None, "odom", False, data_queue)
