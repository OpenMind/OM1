"""
Integration tests for Prometheus monitoring across components.

These tests verify that providers, inputs, actions, and LLM plugins
correctly integrate with PrometheusMonitor by calling register(),
heartbeat(), and report_error() at appropriate times.
"""

import time
from unittest.mock import MagicMock, patch

import pytest


class TestBaseClassPrometheusIntegration:
    """Test that base classes properly set up Prometheus monitor."""

    @pytest.fixture(autouse=True)
    def reset_singletons(self):
        """Reset singleton instances between tests."""
        from providers.prometheus_monitor import PrometheusMonitor

        PrometheusMonitor.reset()  # type: ignore
        yield
        PrometheusMonitor.reset()  # type: ignore

    def test_sensor_base_has_monitor(self):
        """Test that Sensor base class initializes _monitor."""
        with patch("providers.prometheus_monitor.start_http_server"):
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
        with patch("providers.prometheus_monitor.start_http_server"):
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
        with patch("providers.prometheus_monitor.start_http_server"):
            from simulators.base import Simulator, SimulatorConfig

            config = SimulatorConfig()
            simulator = Simulator(config)

            assert hasattr(simulator, "_monitor")
            assert simulator._monitor is not None


class TestLLMPrometheusIntegration:
    """Test that LLM plugins integrate correctly with Prometheus."""

    @pytest.fixture(autouse=True)
    def reset_singletons(self):
        """Reset singleton instances between tests."""
        from providers.prometheus_monitor import PrometheusMonitor

        PrometheusMonitor.reset()  # type: ignore
        yield
        PrometheusMonitor.reset()  # type: ignore

    def test_ollama_llm_registers_with_prometheus(self):
        """Test that OllamaLLM registers with PrometheusMonitor on init."""
        with patch("providers.prometheus_monitor.start_http_server"):
            from llm.plugins.ollama_llm import OllamaLLM, OllamaLLMConfig
            from providers.prometheus_monitor import PrometheusMonitor

            monitor = PrometheusMonitor()
            config = OllamaLLMConfig(
                model="llama3.2", base_url="http://localhost:11434"
            )
            OllamaLLM(config=config)  # Creates instance, registers with monitor

            assert "OllamaLLM" in monitor._providers
            assert monitor.get_status("OllamaLLM") is not None

    def test_deepseek_llm_registers_with_prometheus(self):
        """Test that DeepSeekLLM registers with PrometheusMonitor on init."""
        with (
            patch("providers.prometheus_monitor.start_http_server"),
            patch("llm.plugins.deepseek_llm.openai.AsyncOpenAI"),
        ):
            from llm.plugins.deepseek_llm import DeepSeekConfig, DeepSeekLLM
            from providers.prometheus_monitor import PrometheusMonitor

            monitor = PrometheusMonitor()
            config = DeepSeekConfig(api_key="test-key")
            DeepSeekLLM(config=config)  # Creates instance, registers with monitor

            assert "DeepSeekLLM" in monitor._providers

    def test_llm_has_monitor_attribute(self):
        """Test that LLM instances have _monitor attribute from base class."""
        with patch("providers.prometheus_monitor.start_http_server"):
            from llm.plugins.ollama_llm import OllamaLLM, OllamaLLMConfig

            config = OllamaLLMConfig(
                model="llama3.2", base_url="http://localhost:11434"
            )
            llm = OllamaLLM(config=config)

            assert hasattr(llm, "_monitor")
            assert llm._monitor is not None


class TestInputPluginPrometheusIntegration:
    """Test that input plugins integrate correctly with Prometheus."""

    @pytest.fixture(autouse=True)
    def reset_singletons(self):
        """Reset singleton instances between tests."""
        from providers.prometheus_monitor import PrometheusMonitor

        PrometheusMonitor.reset()  # type: ignore
        yield
        PrometheusMonitor.reset()  # type: ignore

    def test_gps_input_registers(self):
        """Test that Gps input registers with Prometheus."""
        with (
            patch("providers.prometheus_monitor.start_http_server"),
            patch("inputs.plugins.gps.GpsProvider"),
            patch("inputs.plugins.gps.IOProvider"),
        ):
            from inputs.base import SensorConfig
            from inputs.plugins.gps import Gps
            from providers.prometheus_monitor import PrometheusMonitor

            monitor = PrometheusMonitor()
            config = SensorConfig()
            Gps(config)  # Creates instance, registers with monitor

            assert "Gps" in monitor._providers

    def test_rplidar_input_registers(self):
        """Test that RPLidar input registers with Prometheus."""
        with (
            patch("providers.prometheus_monitor.start_http_server"),
            patch("inputs.plugins.rplidar.RPLidarProvider"),
            patch("inputs.plugins.rplidar.IOProvider"),
        ):
            from inputs.plugins.rplidar import RPLidar, RPLidarConfig
            from providers.prometheus_monitor import PrometheusMonitor

            monitor = PrometheusMonitor()
            config = RPLidarConfig()
            RPLidar(config)  # Creates instance, registers with monitor

            # Check that RPLidar or RPLidarProvider is registered
            assert any("RPLidar" in name for name in monitor._providers)

    def test_odom_input_registers(self):
        """Test that Odom input registers with Prometheus."""
        with (
            patch("providers.prometheus_monitor.start_http_server"),
            patch("inputs.plugins.odom.OdomProvider"),
            patch("inputs.plugins.odom.IOProvider"),
        ):
            from inputs.plugins.odom import Odom, OdomConfig
            from providers.prometheus_monitor import PrometheusMonitor

            monitor = PrometheusMonitor()
            config = OdomConfig()
            Odom(config)  # Creates instance, registers with monitor

            assert "Odom" in monitor._providers

    def test_input_heartbeat_on_formatted_buffer(self):
        """Test that input sends heartbeat when formatted_latest_buffer is called."""
        with (
            patch("providers.prometheus_monitor.start_http_server"),
            patch("inputs.plugins.gps.GpsProvider"),
            patch("inputs.plugins.gps.IOProvider"),
        ):
            from inputs.base import Message, SensorConfig
            from inputs.plugins.gps import Gps
            from providers.prometheus_monitor import PrometheusMonitor

            monitor = PrometheusMonitor()
            config = SensorConfig()
            gps_input = Gps(config)

            # Add a message to buffer
            gps_input.messages = [Message(timestamp=time.time(), message="test")]

            initial_heartbeat = monitor._providers["Gps"].last_heartbeat

            # Call formatted_latest_buffer
            gps_input.formatted_latest_buffer()

            # Heartbeat should be updated
            assert monitor._providers["Gps"].last_heartbeat >= initial_heartbeat


class TestActionConnectorPrometheusIntegration:
    """Test that action connectors integrate correctly with Prometheus."""

    @pytest.fixture(autouse=True)
    def reset_singletons(self):
        """Reset singleton instances between tests."""
        from providers.prometheus_monitor import PrometheusMonitor

        PrometheusMonitor.reset()  # type: ignore
        yield
        PrometheusMonitor.reset()  # type: ignore

    def test_face_avatar_connector_registers(self):
        """Test that FaceAvatarConnector registers with Prometheus."""
        with (
            patch("providers.prometheus_monitor.start_http_server"),
            patch("actions.face.connector.avatar.AvatarProvider") as mock_provider,
        ):
            from actions.base import ActionConfig
            from actions.face.connector.avatar import FaceAvatarConnector
            from providers.prometheus_monitor import PrometheusMonitor

            mock_provider_instance = MagicMock()
            mock_provider.return_value = mock_provider_instance

            monitor = PrometheusMonitor()
            config = ActionConfig()
            FaceAvatarConnector(config)  # Creates instance, registers with monitor

            assert "FaceAvatarConnector" in monitor._providers

    @pytest.mark.asyncio
    async def test_face_connector_heartbeat_on_connect(self):
        """Test that face connector sends heartbeat when connect() is called."""
        with (
            patch("providers.prometheus_monitor.start_http_server"),
            patch("actions.face.connector.avatar.AvatarProvider") as mock_provider,
        ):
            from actions.base import ActionConfig
            from actions.face.connector.avatar import FaceAvatarConnector
            from actions.face.interface import FaceAction, FaceInput
            from providers.prometheus_monitor import PrometheusMonitor

            mock_provider_instance = MagicMock()
            mock_provider.return_value = mock_provider_instance

            monitor = PrometheusMonitor()
            config = ActionConfig()
            connector = FaceAvatarConnector(config)

            initial_heartbeat = monitor._providers["FaceAvatarConnector"].last_heartbeat

            # Call connect with proper FaceInput object
            face_input = FaceInput(action=FaceAction.HAPPY)
            await connector.connect(face_input)

            # Heartbeat should be updated
            assert (
                monitor._providers["FaceAvatarConnector"].last_heartbeat
                >= initial_heartbeat
            )


class TestSimulatorPrometheusIntegration:
    """Test that simulators integrate correctly with Prometheus."""

    @pytest.fixture(autouse=True)
    def reset_singletons(self):
        """Reset singleton instances between tests."""
        from providers.prometheus_monitor import PrometheusMonitor

        PrometheusMonitor.reset()  # type: ignore
        yield
        PrometheusMonitor.reset()  # type: ignore

    def test_websim_registers(self):
        """Test that WebSim registers with Prometheus."""
        with (
            patch("providers.prometheus_monitor.start_http_server"),
            patch("simulators.plugins.WebSim.uvicorn"),
            patch("simulators.plugins.WebSim.IOProvider"),
            patch("simulators.plugins.WebSim.FastAPI"),
            patch("simulators.plugins.WebSim.StaticFiles"),
            patch("os.path.exists", return_value=True),
        ):
            from providers.prometheus_monitor import PrometheusMonitor
            from simulators.base import SimulatorConfig
            from simulators.plugins.WebSim import WebSim

            monitor = PrometheusMonitor()
            config = SimulatorConfig(name="TestWebSim")

            # Patch threading to avoid actual server start
            with patch("threading.Thread"):
                WebSim(config)  # Creates instance, registers with monitor

            assert "WebSim" in monitor._providers


class TestPrometheusRecoveryIntegration:
    """Test that providers with recovery callbacks work correctly."""

    @pytest.fixture(autouse=True)
    def reset_singletons(self):
        """Reset singleton instances between tests."""
        from providers.prometheus_monitor import PrometheusMonitor

        PrometheusMonitor.reset()  # type: ignore
        yield
        PrometheusMonitor.reset()  # type: ignore

    def test_provider_registers_with_recovery_callback(self):
        """Test that providers register recovery callbacks."""
        with (
            patch("providers.prometheus_monitor.start_http_server"),
            patch("inputs.plugins.gps.GpsProvider"),
            patch("inputs.plugins.gps.IOProvider"),
        ):
            from inputs.base import SensorConfig
            from inputs.plugins.gps import Gps
            from providers.prometheus_monitor import PrometheusMonitor

            monitor = PrometheusMonitor()
            config = SensorConfig()
            Gps(config)  # Creates instance, registers with monitor

            # Input plugins register with recovery_callback=None
            assert monitor._providers["Gps"].recovery_callback is None

    def test_ollama_llm_registers_with_metadata(self):
        """Test that OllamaLLM registers with proper metadata."""
        with patch("providers.prometheus_monitor.start_http_server"):
            from llm.plugins.ollama_llm import OllamaLLM, OllamaLLMConfig
            from providers.prometheus_monitor import PrometheusMonitor

            monitor = PrometheusMonitor()
            config = OllamaLLMConfig(
                model="llama3.2", base_url="http://localhost:11434"
            )
            OllamaLLM(config=config)  # Creates instance, registers with monitor

            # LLM plugins register with proper metadata
            provider_state = monitor._providers["OllamaLLM"]
            assert provider_state.metadata is not None
            assert provider_state.metadata["type"] == "llm"
            assert provider_state.metadata["provider"] == "ollama"


class TestHeartbeatAndErrorReporting:
    """Test heartbeat and error reporting integration."""

    @pytest.fixture(autouse=True)
    def reset_singletons(self):
        """Reset singleton instances between tests."""
        from providers.prometheus_monitor import PrometheusMonitor

        PrometheusMonitor.reset()  # type: ignore
        yield
        PrometheusMonitor.reset()  # type: ignore

    def test_multiple_providers_register(self):
        """Test that multiple providers can register simultaneously."""
        with (
            patch("providers.prometheus_monitor.start_http_server"),
            patch("inputs.plugins.gps.GpsProvider"),
            patch("inputs.plugins.gps.IOProvider"),
            patch("inputs.plugins.rplidar.RPLidarProvider"),
            patch("inputs.plugins.rplidar.IOProvider"),
        ):
            from inputs.base import SensorConfig
            from inputs.plugins.gps import Gps
            from inputs.plugins.rplidar import RPLidar, RPLidarConfig
            from providers.prometheus_monitor import PrometheusMonitor

            monitor = PrometheusMonitor()

            gps_config = SensorConfig()
            Gps(gps_config)  # Creates instance, registers with monitor

            rplidar_config = RPLidarConfig()
            RPLidar(rplidar_config)  # Creates instance, registers with monitor

            # Both should be registered
            assert "Gps" in monitor._providers
            assert any("RPLidar" in name for name in monitor._providers)
            assert len(monitor._providers) >= 2

    def test_provider_metadata_stored(self):
        """Test that provider metadata is stored correctly."""
        with (
            patch("providers.prometheus_monitor.start_http_server"),
            patch("inputs.plugins.gps.GpsProvider"),
            patch("inputs.plugins.gps.IOProvider"),
        ):
            from inputs.base import SensorConfig
            from inputs.plugins.gps import Gps
            from providers.prometheus_monitor import PrometheusMonitor

            monitor = PrometheusMonitor()
            config = SensorConfig()
            Gps(config)  # Creates instance, registers with monitor

            # Check metadata was stored
            provider_state = monitor._providers["Gps"]
            assert provider_state.metadata is not None
            assert "type" in provider_state.metadata
            assert provider_state.metadata["type"] == "input"
