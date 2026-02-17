"""Tests for SafetySandboxProvider."""

from unittest.mock import MagicMock, patch

from src.providers.safety_sandbox_provider import SafetySandboxProvider


class TestSafetySandboxProvider:

    def test_init_default(self):
        provider = SafetySandboxProvider({})
        assert provider.enabled is True
        assert provider.simulator_type == "WebSim"
        assert provider.simulation_timeout == 2.0
        assert provider.obstacle_margin == 0.3
        assert provider.simulator is not None

    def test_init_custom(self):
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
        provider = SafetySandboxProvider({"enabled": False})
        safe, reason, suggestion = provider.verify("move", {"action": "forwards"})
        assert safe is True
        assert reason == ""
        assert suggestion == {}

    def test_verify_with_robot_state(self):
        provider = SafetySandboxProvider({"enabled": True})
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
        provider = SafetySandboxProvider({"enabled": True})
        provider.simulator.simulate = MagicMock(
            return_value=(False, "Obstacle detected", {})
        )
        safe, reason, suggestion = provider.verify(
            "move", {"action": "forwards"}, robot_state={}
        )
        assert safe is False
        assert reason == "Obstacle detected"

    def test_verify_no_robot_state_provider_raises(self):
        """robot_state=None, RobotStateProvider raises → returns safe. Covers line 46 except path."""
        provider = SafetySandboxProvider({"enabled": True})
        with patch("providers.robot_state_provider.RobotStateProvider") as mock_cls:
            mock_cls.side_effect = Exception("Not available")
            safe, reason, suggestion = provider.verify("move", {})
        assert safe is True

    def test_verify_no_robot_state_provider_succeeds(self):
        """robot_state=None, RobotStateProvider succeeds → uses current_state_dict. Covers line 46."""
        provider = SafetySandboxProvider({"enabled": True})
        provider.simulator.simulate = MagicMock(return_value=(True, "", {}))
        mock_state = {"position": {"x": 0.0}}
        mock_instance = MagicMock()
        mock_instance.current_state_dict = mock_state
        with patch(
            "providers.robot_state_provider.RobotStateProvider",
            return_value=mock_instance,
        ):
            safe, reason, suggestion = provider.verify("move", {})
        provider.simulator.simulate.assert_called_once_with("move", {}, mock_state)
        assert safe is True
