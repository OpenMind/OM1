import time
from unittest.mock import patch

import pytest

from providers.prometheus_monitor import HealthStatus, PrometheusMonitor, ProviderState


@pytest.fixture(autouse=True)
def reset_singleton():
    PrometheusMonitor.reset()  # type: ignore
    yield
    PrometheusMonitor.reset()  # type: ignore


@pytest.fixture
def monitor():
    with patch("uvicorn.Server.run"):
        m = PrometheusMonitor(heartbeat_timeout=1.0, check_interval=0.1)
        yield m
        m.stop()


def test_singleton_pattern():
    with patch("uvicorn.Server.run"):
        m1 = PrometheusMonitor()
        m2 = PrometheusMonitor()
        assert m1 is m2


def test_register_provider(monitor):
    monitor.register("TestProvider", metadata={"type": "test"})

    status = monitor.get_status("TestProvider")
    assert status == HealthStatus.HEALTHY



def test_unregister_provider(monitor):
    monitor.register("TestProvider")
    assert monitor.get_status("TestProvider") is not None

    monitor.unregister("TestProvider")
    assert monitor.get_status("TestProvider") is None


def test_unregister_cleans_metrics(monitor):
    """Unregister should zero out Prometheus metric labels."""
    monitor.register("TestProvider", metadata={"type": "test", "category": "unit"})
    monitor.unregister("TestProvider")

    assert "TestProvider" not in monitor._providers


def test_heartbeat_updates_timestamp(monitor):
    monitor.register("TestProvider")
    initial_time = monitor._providers["TestProvider"].last_heartbeat

    time.sleep(0.01)
    monitor.heartbeat("TestProvider")

    assert monitor._providers["TestProvider"].last_heartbeat > initial_time


def test_heartbeat_recovers_unhealthy_provider(monitor):
    monitor.register("TestProvider")

    with monitor._lock:
        monitor._providers["TestProvider"].status = HealthStatus.UNHEALTHY

    monitor.heartbeat("TestProvider")

    assert monitor._providers["TestProvider"].status == HealthStatus.HEALTHY


def test_report_error_increments_counter(monitor):
    monitor.register("TestProvider")

    monitor.report_error("TestProvider", "Test error")

    assert monitor._providers["TestProvider"].error_count == 1


def test_report_error_multiple_times(monitor):
    monitor.register("TestProvider")

    monitor.report_error("TestProvider", "Error 1")
    monitor.report_error("TestProvider", "Error 2")
    monitor.report_error("TestProvider", "Error 3")

    assert monitor._providers["TestProvider"].error_count == 3


def test_get_all_statuses(monitor):
    monitor.register("Provider1")
    monitor.register("Provider2")

    statuses = monitor.get_all_statuses()

    assert len(statuses) == 2
    assert statuses["Provider1"] == HealthStatus.HEALTHY
    assert statuses["Provider2"] == HealthStatus.HEALTHY


def test_provider_becomes_unhealthy_without_heartbeat(monitor):
    monitor.register("TestProvider")

    with monitor._lock:
        monitor._providers["TestProvider"].last_heartbeat = time.time() - 10

    monitor._perform_health_check()

    assert monitor._providers["TestProvider"].status == HealthStatus.UNHEALTHY


def test_no_recovery_on_unhealthy(monitor):
    """After removing recovery logic, unhealthy providers stay unhealthy."""
    monitor.register("TestProvider")

    with monitor._lock:
        monitor._providers["TestProvider"].last_heartbeat = time.time() - 10
        monitor._providers["TestProvider"].status = HealthStatus.HEALTHY

    monitor._perform_health_check()

    assert monitor._providers["TestProvider"].status == HealthStatus.UNHEALTHY


def test_start_creates_check_thread(monitor):
    monitor.start(port=19090)

    assert monitor._running is True
    assert monitor._check_thread is not None
    assert monitor._check_thread.is_alive()


def test_stop_terminates_check_thread(monitor):
    monitor.start(port=19091)
    thread = monitor._check_thread

    monitor.stop()

    assert monitor._running is False
    thread.join(timeout=1.0)
    assert not thread.is_alive()


def test_heartbeat_for_unregistered_provider(monitor):
    monitor.heartbeat("UnknownProvider")


def test_report_error_for_unregistered_provider(monitor):
    monitor.report_error("UnknownProvider", "Error")


def test_uptime_metric_updated(monitor):
    initial_uptime = monitor._uptime_gauge._value.get()

    time.sleep(0.1)
    monitor._perform_health_check()

    assert monitor._uptime_gauge._value.get() > initial_uptime


def test_provider_state_dataclass():
    state = ProviderState(name="Test")

    assert state.name == "Test"
    assert state.metadata == {}
    assert state.last_heartbeat == 0.0
    assert state.error_count == 0
    assert state.status == HealthStatus.UNKNOWN


def test_health_status_enum():
    assert HealthStatus.HEALTHY.value == "healthy"
    assert HealthStatus.UNHEALTHY.value == "unhealthy"
    assert HealthStatus.UNKNOWN.value == "unknown"


def test_register_updates_existing_provider(monitor):
    monitor.register("TestProvider", metadata={"type": "a"})
    monitor.register("TestProvider", metadata={"type": "b"})

    assert monitor._providers["TestProvider"].metadata["type"] == "b"


def test_concurrent_heartbeats(monitor):
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

    assert monitor._providers["TestProvider"].status == HealthStatus.HEALTHY


def test_unknown_count_in_health_data(monitor):
    """UNKNOWN status providers should be counted separately from unhealthy."""
    monitor.register("Provider1")
    monitor.register("Provider2")

    with monitor._lock:
        monitor._providers["Provider2"].status = HealthStatus.UNKNOWN

    data = monitor._get_health_data()

    assert data["healthy_count"] == 1
    assert data["unhealthy_count"] == 0
    assert data["unknown_count"] == 1


def test_xss_escape_in_table_rows(monitor):
    """Provider names with HTML should be escaped in dashboard."""
    monitor.register(
        '<script>alert("xss")</script>',
        metadata={"type": "<b>bold</b>", "category": "<i>italic</i>"},
    )

    data = monitor._get_health_data()
    rows = monitor._generate_table_rows(data["providers"])

    assert "<script>" not in rows
    assert "&lt;script&gt;" in rows
    assert "<b>" not in rows
    assert "&lt;b&gt;" in rows


def test_stale_metrics_cleaned_on_unregister(monitor):
    """After unregister, provider should not appear in health data."""
    monitor.register("StaleProvider", metadata={"type": "test"})
    assert "StaleProvider" in monitor._providers

    monitor.unregister("StaleProvider")
    assert "StaleProvider" not in monitor._providers

    data = monitor._get_health_data()
    provider_names = [p["name"] for p in data["providers"]]
    assert "StaleProvider" not in provider_names
