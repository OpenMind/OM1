from queue import Queue
from unittest.mock import MagicMock, patch

import pytest

from providers.unitree_go2_state_zenoh_provider import (
    UnitreeGo2StateZenohProvider,
    _state_zenoh_processor,
    state_machine_codes,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    UnitreeGo2StateZenohProvider.reset()  # type: ignore
    yield
    UnitreeGo2StateZenohProvider.reset()  # type: ignore


@pytest.fixture
def patches():
    with (
        patch("providers.unitree_go2_state_zenoh_provider.mp.Process") as mock_process_class,
        patch("providers.unitree_go2_state_zenoh_provider.threading.Thread") as mock_thread_class,
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


def test_initialization_default_topic(patches):
    provider = UnitreeGo2StateZenohProvider()
    assert provider.go2_state is None
    assert provider.go2_state_code is None
    assert provider.go2_action_progress == 0
    patches["process"].start.assert_called_once()
    patches["thread"].start.assert_called_once()


def test_start_is_idempotent(patches):
    provider = UnitreeGo2StateZenohProvider()
    initial_process_calls = patches["process_class"].call_count
    initial_thread_calls = patches["thread_class"].call_count
    provider.start()
    assert patches["process_class"].call_count == initial_process_calls
    assert patches["thread_class"].call_count == initial_thread_calls


def test_state_properties(patches):
    provider = UnitreeGo2StateZenohProvider()
    provider.go2_state = "Sit"
    provider.go2_state_code = 1007
    provider.go2_action_progress = 50
    assert provider.state == "Sit"
    assert provider.state_code == 1007
    assert provider.action_progress == 50


def test_processor_loop_consumes_data(patches):
    from queue import Queue

    provider = UnitreeGo2StateZenohProvider()
    provider.data_queue = Queue()  # type: ignore[assignment]
    sample = {
        "go2_sport_mode_state_msg": "msg",
        "go2_state_code": 1007,
        "go2_state": "Sit",
        "go2_action_progress": 75,
    }
    provider.data_queue.put(sample)
    # is_set returns False once (so the body runs once) then True (loop exits).
    provider._stop_event = MagicMock()
    provider._stop_event.is_set.side_effect = [False, True]
    provider._processor_loop()
    assert provider.go2_state == "Sit"
    assert provider.go2_state_code == 1007
    assert provider.go2_action_progress == 75


def test_stop_signals_stop(patches):
    provider = UnitreeGo2StateZenohProvider()
    provider._reader_proc = patches["process"]
    provider._processor_thread = patches["thread"]
    provider.stop()
    patches["process"].terminate.assert_called_once()


def test_state_machine_codes_present():
    assert state_machine_codes[1007] == "Sit"
    assert state_machine_codes[1015] == "Regular Walking"
    assert state_machine_codes.get(99999) is None


def test_state_zenoh_processor_subscribes(patches):
    """Run the in-process worker function with a pre-stopped control queue."""
    data_queue: Queue = Queue()
    control_queue: Queue = Queue()
    control_queue.put("STOP")

    session = MagicMock()
    with (
        patch(
            "providers.unitree_go2_state_zenoh_provider.open_zenoh_session",
            return_value=session,
        ),
        patch("providers.unitree_go2_state_zenoh_provider.setup_logging"),
        patch(
            "providers.unitree_go2_state_zenoh_provider.time.sleep",
            return_value=None,
        ),
    ):
        _state_zenoh_processor(None, False, data_queue, control_queue)  # type: ignore[arg-type]

    session.declare_subscriber.assert_called_once()
    topic_arg = session.declare_subscriber.call_args[0][0]
    assert topic_arg == "sportmodestate"


def test_state_zenoh_processor_session_failure():
    data_queue: Queue = Queue()
    control_queue: Queue = Queue()
    with (
        patch(
            "providers.unitree_go2_state_zenoh_provider.open_zenoh_session",
            side_effect=RuntimeError("fail"),
        ),
        patch("providers.unitree_go2_state_zenoh_provider.setup_logging"),
    ):
        _state_zenoh_processor(None, False, data_queue, control_queue)  # type: ignore[arg-type]
