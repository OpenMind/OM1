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


class TestAutoRegistrationSensor:
    """Test that Sensor subclasses auto-register with PrometheusMonitor."""

    def test_sensor_auto_registers_on_init(self):
        """Sensor subclass should auto-register with its class name."""
        with patch("uvicorn.Server.run"):
            from inputs.base import Sensor, SensorConfig

            class MySensor(Sensor):
                def __init__(self, config):
                    super().__init__(config)

                async def _raw_to_text(self, raw_input):
                    return None

                async def raw_to_text(self, raw_input):
                    pass

                def formatted_latest_buffer(self):
                    return None

            monitor = PrometheusMonitor()
            config = SensorConfig()
            MySensor(config)

            assert "MySensor" in monitor._providers
            assert monitor._providers["MySensor"].metadata["type"] == "input"

    def test_sensor_auto_heartbeat_on_formatted_latest_buffer(self):
        """formatted_latest_buffer returning non-None should trigger heartbeat."""
        with patch("uvicorn.Server.run"):
            from inputs.base import Sensor, SensorConfig

            class HeartbeatSensor(Sensor):
                def __init__(self, config):
                    super().__init__(config)

                async def _raw_to_text(self, raw_input):
                    return None

                async def raw_to_text(self, raw_input):
                    pass

                def formatted_latest_buffer(self):
                    return "some data"

            monitor = PrometheusMonitor()
            config = SensorConfig()
            sensor = HeartbeatSensor(config)

            initial_hb = monitor._providers["HeartbeatSensor"].last_heartbeat
            time.sleep(0.01)
            sensor.formatted_latest_buffer()

            assert monitor._providers["HeartbeatSensor"].last_heartbeat > initial_hb

    def test_sensor_no_heartbeat_on_none_result(self):
        """formatted_latest_buffer returning None should NOT trigger heartbeat."""
        with patch("uvicorn.Server.run"):
            from inputs.base import Sensor, SensorConfig

            class NullSensor(Sensor):
                def __init__(self, config):
                    super().__init__(config)

                async def _raw_to_text(self, raw_input):
                    return None

                async def raw_to_text(self, raw_input):
                    pass

                def formatted_latest_buffer(self):
                    return None

            monitor = PrometheusMonitor()
            config = SensorConfig()
            sensor = NullSensor(config)

            initial_hb = monitor._providers["NullSensor"].last_heartbeat
            time.sleep(0.01)
            sensor.formatted_latest_buffer()

            assert monitor._providers["NullSensor"].last_heartbeat == initial_hb


class TestAutoRegistrationAction:
    """Test that ActionConnector subclasses auto-register."""

    def test_action_connector_auto_registers_on_init(self):
        """ActionConnector subclass should auto-register."""
        with patch("uvicorn.Server.run"):
            from actions.base import ActionConfig, ActionConnector

            class MyConnector(ActionConnector):
                async def connect(self, output_interface):
                    pass

            monitor = PrometheusMonitor()
            config = ActionConfig()
            MyConnector(config)

            assert "MyConnector" in monitor._providers
            assert monitor._providers["MyConnector"].metadata["type"] == "action"

    @pytest.mark.asyncio
    async def test_action_connector_auto_heartbeat_on_connect(self):
        """connect() should auto-send heartbeat after successful return."""
        with patch("uvicorn.Server.run"):
            from actions.base import ActionConfig, ActionConnector

            class HBConnector(ActionConnector):
                async def connect(self, output_interface):
                    pass

            monitor = PrometheusMonitor()
            config = ActionConfig()
            connector = HBConnector(config)

            initial_hb = monitor._providers["HBConnector"].last_heartbeat
            time.sleep(0.01)
            await connector.connect("dummy")

            assert monitor._providers["HBConnector"].last_heartbeat > initial_hb


class TestAutoRegistrationLLM:
    """Test that LLM subclasses auto-register."""

    def test_llm_auto_registers_on_init(self):
        """LLM subclass should auto-register with its class name."""
        with patch("uvicorn.Server.run"):
            from llm import LLM, LLMConfig

            class MyLLM(LLM):
                async def ask(self, prompt, messages=[]):
                    return None

            monitor = PrometheusMonitor()
            config = LLMConfig()
            MyLLM(config=config)

            assert "MyLLM" in monitor._providers
            assert monitor._providers["MyLLM"].metadata["type"] == "llm"

    @pytest.mark.asyncio
    async def test_llm_auto_heartbeat_on_ask_success(self):
        """ask() returning successfully should trigger heartbeat."""
        with patch("uvicorn.Server.run"):
            from llm import LLM, LLMConfig

            class SuccessLLM(LLM):
                async def ask(self, prompt, messages=[]):
                    return None

            monitor = PrometheusMonitor()
            config = LLMConfig()
            llm = SuccessLLM(config=config)

            initial_hb = monitor._providers["SuccessLLM"].last_heartbeat
            time.sleep(0.01)
            await llm.ask("hello")

            assert monitor._providers["SuccessLLM"].last_heartbeat > initial_hb

    @pytest.mark.asyncio
    async def test_llm_auto_error_report_on_ask_exception(self):
        """ask() raising exception should report error and re-raise."""
        with patch("uvicorn.Server.run"):
            from llm import LLM, LLMConfig

            class FailLLM(LLM):
                async def ask(self, prompt, messages=[]):
                    raise ValueError("LLM failed")

            monitor = PrometheusMonitor()
            config = LLMConfig()
            llm = FailLLM(config=config)

            with pytest.raises(ValueError, match="LLM failed"):
                await llm.ask("hello")

            assert monitor._providers["FailLLM"].error_count == 1


class TestAutoRegistrationSimulator:
    """Test that Simulator subclasses auto-register."""

    def test_simulator_auto_registers_on_init(self):
        """Simulator subclass should auto-register."""
        with patch("uvicorn.Server.run"):
            from simulators.base import Simulator, SimulatorConfig

            class MySim(Simulator):
                def sim(self, actions):
                    pass

            monitor = PrometheusMonitor()
            config = SimulatorConfig()
            MySim(config)

            assert "MySim" in monitor._providers
            assert monitor._providers["MySim"].metadata["type"] == "simulator"

    def test_simulator_auto_heartbeat_on_sim(self):
        """sim() should auto-send heartbeat after call."""
        with patch("uvicorn.Server.run"):
            from simulators.base import Simulator, SimulatorConfig

            class HBSim(Simulator):
                def sim(self, actions):
                    pass

            monitor = PrometheusMonitor()
            config = SimulatorConfig()
            sim = HBSim(config)

            initial_hb = monitor._providers["HBSim"].last_heartbeat
            time.sleep(0.01)
            sim.sim([])

            assert monitor._providers["HBSim"].last_heartbeat > initial_hb


class TestBaseClassPrometheusIntegration:
    """Test that base classes properly set up Prometheus monitor."""

    def test_sensor_base_has_monitor(self):
        """Test that Sensor base class initializes _monitor."""
        with patch("uvicorn.Server.run"):
            from inputs.base import Sensor, SensorConfig

            class TestSensor(Sensor):
                def __init__(self, config):
                    super().__init__(config)

                async def _raw_to_text(self, raw_input):
                    return None

                async def raw_to_text(self, raw_input):
                    pass

                def formatted_latest_buffer(self):
                    return None

            config = SensorConfig()
            sensor = TestSensor(config)

            assert hasattr(sensor, "_monitor")
            assert sensor._monitor is not None

    def test_action_connector_base_has_monitor(self):
        """Test that ActionConnector base class initializes _monitor."""
        with patch("uvicorn.Server.run"):
            from actions.base import ActionConfig, ActionConnector

            class TestConnector(ActionConnector):
                async def connect(self, output_interface):
                    pass

            config = ActionConfig()
            connector = TestConnector(config)

            assert hasattr(connector, "_monitor")
            assert connector._monitor is not None

    def test_simulator_base_has_monitor(self):
        """Test that Simulator base class initializes _monitor."""
        with patch("uvicorn.Server.run"):
            from simulators.base import Simulator, SimulatorConfig

            config = SimulatorConfig()
            simulator = Simulator(config)

            assert hasattr(simulator, "_monitor")
            assert simulator._monitor is not None
