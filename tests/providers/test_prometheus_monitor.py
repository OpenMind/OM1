import time
from unittest.mock import MagicMock, patch

import pytest

from providers.prometheus_monitor import HealthStatus, PrometheusMonitor, ProviderState


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instance between tests."""
    PrometheusMonitor.reset()  # type: ignore
    yield
    PrometheusMonitor.reset()  # type: ignore


@pytest.fixture
def monitor():
    """Create a PrometheusMonitor instance with short timeouts for testing."""
    with patch("providers.prometheus_monitor.start_http_server"):
        m = PrometheusMonitor(heartbeat_timeout=1.0, check_interval=0.1)
        yield m
        m.stop()


def test_singleton_pattern():
    """Test that PrometheusMonitor is a singleton."""
    with patch("providers.prometheus_monitor.start_http_server"):
        m1 = PrometheusMonitor()
        m2 = PrometheusMonitor()
        assert m1 is m2


def test_register_provider(monitor):
    """Test registering a provider."""
    monitor.register("TestProvider", metadata={"type": "test"})

    status = monitor.get_status("TestProvider")
    assert status == HealthStatus.HEALTHY


def test_register_with_recovery_callback(monitor):
    """Test registering a provider with recovery callback."""
    callback = MagicMock(return_value=True)
    monitor.register("TestProvider", recovery_callback=callback)

    assert monitor._providers["TestProvider"].recovery_callback == callback


def test_unregister_provider(monitor):
    """Test unregistering a provider."""
    monitor.register("TestProvider")
    assert monitor.get_status("TestProvider") is not None

    monitor.unregister("TestProvider")
    assert monitor.get_status("TestProvider") is None


def test_heartbeat_updates_timestamp(monitor):
    """Test that heartbeat updates the last_heartbeat timestamp."""
    monitor.register("TestProvider")
    initial_time = monitor._providers["TestProvider"].last_heartbeat

    time.sleep(0.01)
    monitor.heartbeat("TestProvider")

    assert monitor._providers["TestProvider"].last_heartbeat > initial_time


def test_heartbeat_recovers_unhealthy_provider(monitor):
    """Test that heartbeat recovers an unhealthy provider."""
    monitor.register("TestProvider")

    with monitor._lock:
        monitor._providers["TestProvider"].status = HealthStatus.UNHEALTHY

    monitor.heartbeat("TestProvider")

    assert monitor._providers["TestProvider"].status == HealthStatus.HEALTHY


def test_report_error_increments_counter(monitor):
    """Test that report_error increments error count."""
    monitor.register("TestProvider")

    monitor.report_error("TestProvider", "Test error")

    assert monitor._providers["TestProvider"].error_count == 1


def test_report_error_multiple_times(monitor):
    """Test reporting multiple errors."""
    monitor.register("TestProvider")

    monitor.report_error("TestProvider", "Error 1")
    monitor.report_error("TestProvider", "Error 2")
    monitor.report_error("TestProvider", "Error 3")

    assert monitor._providers["TestProvider"].error_count == 3


def test_get_all_statuses(monitor):
    """Test getting status of all providers."""
    monitor.register("Provider1")
    monitor.register("Provider2")

    statuses = monitor.get_all_statuses()

    assert len(statuses) == 2
    assert statuses["Provider1"] == HealthStatus.HEALTHY
    assert statuses["Provider2"] == HealthStatus.HEALTHY


def test_provider_becomes_unhealthy_without_heartbeat(monitor):
    """Test that provider becomes unhealthy without heartbeat."""
    monitor.register("TestProvider")

    with monitor._lock:
        monitor._providers["TestProvider"].last_heartbeat = time.time() - 10

    monitor._perform_health_check()

    assert monitor._providers["TestProvider"].status == HealthStatus.UNHEALTHY


def test_recovery_callback_called_on_unhealthy(monitor):
    """Test that recovery callback is called when provider becomes unhealthy."""
    callback = MagicMock(return_value=True)
    monitor.register("TestProvider", recovery_callback=callback)

    with monitor._lock:
        monitor._providers["TestProvider"].last_heartbeat = time.time() - 10
        monitor._providers["TestProvider"].status = HealthStatus.HEALTHY

    monitor._perform_health_check()

    callback.assert_called_once()


def test_recovery_callback_success_restores_health(monitor):
    """Test that successful recovery restores health status."""
    callback = MagicMock(return_value=True)
    monitor.register("TestProvider", recovery_callback=callback)

    with monitor._lock:
        monitor._providers["TestProvider"].last_heartbeat = time.time() - 10
        monitor._providers["TestProvider"].status = HealthStatus.HEALTHY

    monitor._perform_health_check()

    assert monitor._providers["TestProvider"].status == HealthStatus.HEALTHY


def test_recovery_callback_failure_keeps_unhealthy(monitor):
    """Test that failed recovery keeps provider unhealthy."""
    callback = MagicMock(return_value=False)
    monitor.register("TestProvider", recovery_callback=callback)

    with monitor._lock:
        monitor._providers["TestProvider"].last_heartbeat = time.time() - 10
        monitor._providers["TestProvider"].status = HealthStatus.HEALTHY

    monitor._perform_health_check()

    assert monitor._providers["TestProvider"].status == HealthStatus.UNHEALTHY


def test_recovery_callback_exception_handled(monitor):
    """Test that exceptions in recovery callback are handled."""
    callback = MagicMock(side_effect=Exception("Recovery failed"))
    monitor.register("TestProvider", recovery_callback=callback)

    with monitor._lock:
        monitor._providers["TestProvider"].last_heartbeat = time.time() - 10
        monitor._providers["TestProvider"].status = HealthStatus.HEALTHY

    monitor._perform_health_check()

    assert monitor._providers["TestProvider"].status == HealthStatus.UNHEALTHY


def test_start_creates_check_thread(monitor):
    """Test that start creates the health check thread."""
    monitor.start(port=19090)

    assert monitor._running is True
    assert monitor._check_thread is not None
    assert monitor._check_thread.is_alive()


def test_stop_terminates_check_thread(monitor):
    """Test that stop terminates the health check thread."""
    monitor.start(port=19091)
    thread = monitor._check_thread

    monitor.stop()

    assert monitor._running is False
    # Give extra time for thread to terminate
    thread.join(timeout=1.0)
    assert not thread.is_alive()


def test_heartbeat_for_unregistered_provider(monitor):
    """Test that heartbeat for unregistered provider is ignored."""
    monitor.heartbeat("UnknownProvider")
    # Should not raise exception


def test_report_error_for_unregistered_provider(monitor):
    """Test that report_error for unregistered provider is ignored."""
    monitor.report_error("UnknownProvider", "Error")
    # Should not raise exception


def test_uptime_metric_updated(monitor):
    """Test that uptime metric is updated during health check."""
    initial_uptime = monitor._uptime_gauge._value.get()

    time.sleep(0.1)
    monitor._perform_health_check()

    # Uptime should have increased
    assert monitor._uptime_gauge._value.get() > initial_uptime


def test_provider_state_dataclass():
    """Test ProviderState dataclass initialization."""
    state = ProviderState(name="Test")

    assert state.name == "Test"
    assert state.metadata == {}
    assert state.recovery_callback is None
    assert state.last_heartbeat == 0.0
    assert state.error_count == 0
    assert state.status == HealthStatus.UNKNOWN


def test_health_status_enum():
    """Test HealthStatus enum values."""
    assert HealthStatus.HEALTHY.value == "healthy"
    assert HealthStatus.UNHEALTHY.value == "unhealthy"
    assert HealthStatus.UNKNOWN.value == "unknown"


def test_register_updates_existing_provider(monitor):
    """Test that re-registering updates existing provider."""
    callback1 = MagicMock()
    callback2 = MagicMock()

    monitor.register("TestProvider", recovery_callback=callback1)
    monitor.register("TestProvider", recovery_callback=callback2)

    assert monitor._providers["TestProvider"].recovery_callback == callback2


def test_concurrent_heartbeats(monitor):
    """Test concurrent heartbeat calls are thread-safe."""
    import threading

    monitor.register("TestProvider")

    def send_heartbeats():
        for _ in range(100):
            monitor.heartbeat("TestProvider")

    threads = [threading.Thread(target=send_heartbeats) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Should complete without errors
    assert monitor._providers["TestProvider"].status == HealthStatus.HEALTHY
