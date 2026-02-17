"""Tests for SafetySimulator."""

from unittest.mock import MagicMock, patch

from src.simulators.plugins.safety_simulator import SafetySimulator


class TestSafetySimulator:

    @patch("simulators.load_simulator")
    def test_init_success(self, mock_load):
        mock_sim = MagicMock()
        mock_load.return_value = mock_sim
        sim = SafetySimulator("WebSim", {})
        assert sim.simulator is mock_sim
        mock_load.assert_called_once_with({"type": "WebSim", "config": {}})

    @patch("simulators.load_simulator")
    def test_init_failure(self, mock_load):
        mock_load.side_effect = Exception("Failed")
        sim = SafetySimulator("WebSim", {})
        assert sim.simulator is None

    def test_simulate_no_simulator(self):
        sim = SafetySimulator("WebSim", {})
        sim.simulator = None
        safe, reason, suggestion = sim.simulate("move", {}, {})
        assert safe is True
        assert reason == ""
        assert suggestion == {}

    def test_simulate_placeholder(self):
        sim = SafetySimulator("WebSim", {})
        sim.simulator = MagicMock()
        safe, reason, suggestion = sim.simulate("move", {"action": "forwards"}, {})
        assert safe is True
        assert reason == ""
        assert suggestion == {}

    def test_simulate_exception_returns_unsafe(self):
        """Exception inside simulate() try block → returns False. Covers lines 80-83."""
        sim = SafetySimulator("WebSim", {})
        sim.simulator = MagicMock()

        # Patch logging.debug to raise so the try block throws
        with patch("src.simulators.plugins.safety_simulator.logging") as mock_log:
            mock_log.debug.side_effect = Exception("sim crash")
            safe, reason, suggestion = sim.simulate("move", {}, {})

        assert safe is False
        assert "sim crash" in reason
        assert suggestion == {}
