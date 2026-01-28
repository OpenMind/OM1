import threading
import time

from simulators.base import Simulator, SimulatorConfig


def test_simulator_init():
    """Test simulator initialization with config."""
    config = SimulatorConfig(name="test_sim")
    simulator = Simulator(config)
    assert simulator.name == "test_sim"
    assert simulator.config == config


def test_simulator_init_default_name():
    """Test simulator initialization with default name."""
    config = SimulatorConfig()
    simulator = Simulator(config)
    assert simulator.name == "Simulator"
    assert simulator.config == config


def test_simulator_config_kwargs():
    """Test simulator config with additional kwargs."""
    config = SimulatorConfig(name="test_sim", host="localhost", port=8000)
    name = getattr(config, "name", None)
    host = getattr(config, "host", None)
    port = getattr(config, "port", None)
    assert name == "test_sim"
    assert host == "localhost"
    assert port == 8000


def test_sleep_without_stop_event():
    """Test that sleep works normally when no stop event is set."""
    config = SimulatorConfig()
    simulator = Simulator(config)

    start_time = time.time()
    result = simulator.sleep(0.1)
    duration = time.time() - start_time

    assert result is True
    assert duration >= 0.1


def test_sleep_with_stop_event_not_triggered():
    """Test that sleep completes normally when stop event is set but not triggered."""
    config = SimulatorConfig()
    simulator = Simulator(config)
    stop_event = threading.Event()
    simulator.set_stop_event(stop_event)

    start_time = time.time()
    result = simulator.sleep(0.1)
    duration = time.time() - start_time

    assert result is True
    assert duration >= 0.1


def test_sleep_interrupted_by_stop_event():
    """Test that sleep is interrupted when stop event is set during sleep."""
    config = SimulatorConfig()
    simulator = Simulator(config)
    stop_event = threading.Event()
    simulator.set_stop_event(stop_event)

    def trigger_stop():
        time.sleep(0.05)
        stop_event.set()

    # Start a thread to trigger the stop event
    stop_thread = threading.Thread(target=trigger_stop)
    stop_thread.start()

    start_time = time.time()
    result = simulator.sleep(1.0)
    duration = time.time() - start_time

    stop_thread.join()

    assert result is False
    assert duration < 0.2
    assert simulator.should_stop() is True


def test_sleep_already_stopped():
    """Test that sleep returns immediately when stop event is already set."""
    config = SimulatorConfig()
    simulator = Simulator(config)
    stop_event = threading.Event()
    stop_event.set()
    simulator.set_stop_event(stop_event)

    start_time = time.time()
    result = simulator.sleep(1.0)
    duration = time.time() - start_time

    assert result is False
    assert duration < 0.1
    assert simulator.should_stop() is True


def test_should_stop_without_event():
    """Test that should_stop returns False when no stop event is set."""
    config = SimulatorConfig()
    simulator = Simulator(config)

    assert simulator.should_stop() is False


def test_should_stop_with_event_not_set():
    """Test that should_stop returns False when stop event exists but is not set."""
    config = SimulatorConfig()
    simulator = Simulator(config)
    stop_event = threading.Event()
    simulator.set_stop_event(stop_event)

    assert simulator.should_stop() is False


def test_should_stop_with_event_set():
    """Test that should_stop returns True when stop event is set."""
    config = SimulatorConfig()
    simulator = Simulator(config)
    stop_event = threading.Event()
    simulator.set_stop_event(stop_event)
    stop_event.set()

    assert simulator.should_stop() is True
