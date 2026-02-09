import threading
import time

from backgrounds.base import Background, BackgroundConfig


def test_background_init():
    """Test background initialization with config."""
    config = BackgroundConfig()
    background = Background(config)
    assert background.name == "Background"
    assert background.config == config


def test_background_init_default_name():
    """Test background initialization with default name."""
    config = BackgroundConfig()
    background = Background(config)
    assert background.name == "Background"
    assert background.config == config


def test_sleep_without_stop_event():
    """Test that sleep works normally when no stop event is set."""
    config = BackgroundConfig()
    background = Background(config)

    start_time = time.time()
    result = background.sleep(0.1)
    duration = time.time() - start_time

    assert result is True
    assert duration >= 0.1


def test_sleep_with_stop_event_not_triggered():
    """Test that sleep completes normally when stop event is set but not triggered."""
    config = BackgroundConfig()
    background = Background(config)
    stop_event = threading.Event()
    background.set_stop_event(stop_event)

    start_time = time.time()
    result = background.sleep(0.1)
    duration = time.time() - start_time

    assert result is True
    assert duration >= 0.1


def test_sleep_interrupted_by_stop_event():
    """Test that sleep is interrupted when stop event is set during sleep."""
    config = BackgroundConfig()
    background = Background(config)
    stop_event = threading.Event()
    background.set_stop_event(stop_event)

    def trigger_stop():
        time.sleep(0.05)
        stop_event.set()

    # Start a thread to trigger the stop event
    stop_thread = threading.Thread(target=trigger_stop)
    stop_thread.start()

    start_time = time.time()
    result = background.sleep(1.0)  # Sleep for a long time, but should be interrupted
    duration = time.time() - start_time

    stop_thread.join()

    assert result is False  # Should return False when interrupted
    assert duration < 0.2  # Should wake up quickly, well before the 1 second sleep
    assert background.should_stop() is True


def test_sleep_already_stopped():
    """Test that sleep returns immediately when stop event is already set."""
    config = BackgroundConfig()
    background = Background(config)
    stop_event = threading.Event()
    stop_event.set()  # Set before sleep
    background.set_stop_event(stop_event)

    start_time = time.time()
    result = background.sleep(1.0)
    duration = time.time() - start_time

    assert result is False
    assert duration < 0.1  # Should return almost immediately
    assert background.should_stop() is True


def test_should_stop_without_event():
    """Test that should_stop returns False when no stop event is set."""
    config = BackgroundConfig()
    background = Background(config)

    assert background.should_stop() is False


def test_should_stop_with_event_not_set():
    """Test that should_stop returns False when stop event exists but is not set."""
    config = BackgroundConfig()
    background = Background(config)
    stop_event = threading.Event()
    background.set_stop_event(stop_event)

    assert background.should_stop() is False


def test_should_stop_with_event_set():
    """Test that should_stop returns True when stop event is set."""
    config = BackgroundConfig()
    background = Background(config)
    stop_event = threading.Event()
    background.set_stop_event(stop_event)
    stop_event.set()

    assert background.should_stop() is True
