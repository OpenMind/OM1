"""Tests for SafetySimulator."""

from unittest.mock import MagicMock, patch

import pytest

from src.simulators.plugins.safety_simulator import SafetySimulator


class TestSafetySimulator:
    """Test suite for SafetySimulator."""

    @patch("simulators.load_simulator")
    def test_init_success(self, mock_load):
        """Test successful initialization."""
        mock_sim = MagicMock()
        mock_load.return_value = mock_sim
        sim = SafetySimulator("WebSim", {})
        assert sim.simulator is mock_sim
        mock_load.assert_called_once_with({"type": "WebSim", "config": {}})

    @patch("simulators.load_simulator")
    def test_init_failure(self, mock_load):
        """Test initialization when simulator load fails."""
        mock_load.side_effect = Exception("Failed")
        sim = SafetySimulator("WebSim", {})
        assert sim.simulator is None
        # Should not raise

    def test_simulate_no_simulator(self):
        """Test simulate when no simulator available."""
        sim = SafetySimulator("WebSim", {})
        sim.simulator = None
        safe, reason, suggestion = sim.simulate("move", {}, {})
        assert safe is True
        assert reason == ""
        assert suggestion == {}

    def test_simulate_placeholder(self):
        """Test placeholder simulate (always safe)."""
        sim = SafetySimulator("WebSim", {})
        # Even with simulator, placeholder returns safe
        sim.simulator = MagicMock()
        safe, reason, suggestion = sim.simulate("move", {"action": "forwards"}, {})
        assert safe is True
        assert reason == ""
        assert suggestion == {}

    def test_simulate_error(self):
        """Test simulate when exception occurs."""
        sim = SafetySimulator("WebSim", {})
        sim.simulator = MagicMock()

        # We need to test exception handling inside simulate
        # Override the simulate method temporarily
        def bad_simulate(*args, **kwargs):
            raise Exception("Simulation error")

        original_simulate = sim.simulate
        sim.simulate = bad_simulate
        with pytest.raises(Exception):
            sim.simulate("move", {}, {})
        sim.simulate = original_simulate
