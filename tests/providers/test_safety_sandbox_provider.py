"""Tests for SafetySandboxProvider."""

from unittest.mock import MagicMock, patch

from src.providers.safety_sandbox_provider import SafetySandboxProvider


class TestSafetySandboxProvider:
    """Test suite for SafetySandboxProvider."""

    def test_init_default(self):
        """Test initialization with default config."""
        config = {}
        provider = SafetySandboxProvider(config)
        assert provider.enabled is True
        assert provider.simulator_type == "WebSim"
        assert provider.simulation_timeout == 2.0
        assert provider.obstacle_margin == 0.3
        assert provider.simulator is not None

    def test_init_custom(self):
        """Test initialization with custom config."""
        config = {
            "enabled": False,
            "simulator": "TestSim",
            "simulation_timeout": 5.0,
            "obstacle_margin": 0.5,
        }
        provider = SafetySandboxProvider(config)
        assert provider.enabled is False
        assert provider.simulator_type == "TestSim"
        assert provider.simulation_timeout == 5.0
        assert provider.obstacle_margin == 0.5

    def test_verify_disabled(self):
        """Test verify returns safe immediately when disabled."""
        config = {"enabled": False}
        provider = SafetySandboxProvider(config)
        safe, reason, suggestion = provider.verify("move", {"action": "forwards"})
        assert safe is True
        assert reason == ""
        assert suggestion == {}

    @patch("providers.robot_state_provider.RobotStateProvider")
    def test_verify_no_robot_state(self, mock_state_provider):
        """Test verify when robot_state not provided and can't get from provider."""
        config = {"enabled": True}
        provider = SafetySandboxProvider(config)
        # Mock simulator to return safe
        provider.simulator.simulate = MagicMock(return_value=(True, "", {}))
        # Make RobotStateProvider raise exception
        mock_state_provider.side_effect = Exception("Not available")
        safe, reason, suggestion = provider.verify("move", {"action": "forwards"})
        assert safe is True

    @patch("providers.robot_state_provider.RobotStateProvider")
    def test_verify_with_robot_state(self, mock_state_provider):
        """Test verify with robot_state provided."""
        config = {"enabled": True}
        provider = SafetySandboxProvider(config)
        # Mock simulator
        provider.simulator.simulate = MagicMock(return_value=(True, "", {}))
        robot_state = {"position": {"x": 1.0, "y": 2.0}}
        safe, reason, suggestion = provider.verify(
            "move", {"action": "forwards"}, robot_state=robot_state
        )
        provider.simulator.simulate.assert_called_once_with(
            "move", {"action": "forwards"}, robot_state
        )
        assert safe is True

    def test_verify_blocked(self):
        """Test verify when simulator returns unsafe."""
        config = {"enabled": True}
        provider = SafetySandboxProvider(config)
        provider.simulator.simulate = MagicMock(
            return_value=(False, "Obstacle detected", {})
        )
        safe, reason, suggestion = provider.verify(
            "move", {"action": "forwards"}, robot_state={}
        )
        assert safe is False
        assert reason == "Obstacle detected"
        assert suggestion == {}
