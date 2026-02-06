"""
Integration tests for Prometheus monitoring across components.

These tests verify that base classes automatically register providers
and send heartbeats via __init_subclass__ wrapping.
"""

import time
from unittest.mock import patch

import pytest


class TestAutoRegistrationSensor:
    """Test that Sensor subclasses auto-register with PrometheusMonitor."""

    @pytest.fixture(autouse=True)
    def reset_singletons(self):
        from providers.prometheus_monitor import PrometheusMonitor

        PrometheusMonitor.reset()  # type: ignore
        yield
        try:
            PrometheusMonitor().stop()
        except Exception:
            pass
        PrometheusMonitor.reset()  # type: ignore

    def test_sensor_auto_registers_on_init(self):
        """Sensor subclass should auto-register with its class name."""
        with patch("uvicorn.Server.run"):
            from inputs.base import Sensor, SensorConfig
            from providers.prometheus_monitor import PrometheusMonitor

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
            from providers.prometheus_monitor import PrometheusMonitor

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
            from providers.prometheus_monitor import PrometheusMonitor

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

    @pytest.fixture(autouse=True)
    def reset_singletons(self):
        from providers.prometheus_monitor import PrometheusMonitor

        PrometheusMonitor.reset()  # type: ignore
        yield
        try:
            PrometheusMonitor().stop()
        except Exception:
            pass
        PrometheusMonitor.reset()  # type: ignore

    def test_action_connector_auto_registers_on_init(self):
        """ActionConnector subclass should auto-register."""
        with patch("uvicorn.Server.run"):
            from actions.base import ActionConfig, ActionConnector
            from providers.prometheus_monitor import PrometheusMonitor

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
            from providers.prometheus_monitor import PrometheusMonitor

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

    @pytest.fixture(autouse=True)
    def reset_singletons(self):
        from providers.prometheus_monitor import PrometheusMonitor

        PrometheusMonitor.reset()  # type: ignore
        yield
        try:
            PrometheusMonitor().stop()
        except Exception:
            pass
        PrometheusMonitor.reset()  # type: ignore

    def test_llm_auto_registers_on_init(self):
        """LLM subclass should auto-register with its class name."""
        with patch("uvicorn.Server.run"):
            from llm import LLM, LLMConfig
            from providers.prometheus_monitor import PrometheusMonitor

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
            from providers.prometheus_monitor import PrometheusMonitor

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
            from providers.prometheus_monitor import PrometheusMonitor

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

    @pytest.fixture(autouse=True)
    def reset_singletons(self):
        from providers.prometheus_monitor import PrometheusMonitor

        PrometheusMonitor.reset()  # type: ignore
        yield
        try:
            PrometheusMonitor().stop()
        except Exception:
            pass
        PrometheusMonitor.reset()  # type: ignore

    def test_simulator_auto_registers_on_init(self):
        """Simulator subclass should auto-register."""
        with patch("uvicorn.Server.run"):
            from providers.prometheus_monitor import PrometheusMonitor
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
            from providers.prometheus_monitor import PrometheusMonitor
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

    @pytest.fixture(autouse=True)
    def reset_singletons(self):
        from providers.prometheus_monitor import PrometheusMonitor

        PrometheusMonitor.reset()  # type: ignore
        yield
        try:
            PrometheusMonitor().stop()
        except Exception:
            pass
        PrometheusMonitor.reset()  # type: ignore

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
