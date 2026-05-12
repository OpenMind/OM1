"""
Safety Sandbox Provider for verifying action safety before execution.
"""

import logging
from typing import Any, Dict, Optional, Tuple

from simulators.plugins.safety_simulator import SafetySimulator


class SafetySandboxProvider:
    """
    Provider that verifies safety of robot actions through simulation.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the safety sandbox provider with configuration.
        """
        self.enabled = config.get("enabled", True)
        self.simulator_type = config.get("simulator", "WebSim")
        self.simulation_timeout = config.get("simulation_timeout", 2.0)
        self.obstacle_margin = config.get("obstacle_margin", 0.3)
        self.simulator = SafetySimulator(self.simulator_type, {})

        logging.info(f"SafetySandboxProvider initialized (enabled={self.enabled})")

    def verify(
        self,
        action_name: str,
        params: Dict[str, Any],
        robot_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Verify if an action is safe to execute.
        """
        if not self.enabled:
            return True, "", {}

        # If robot_state not provided, try to get from RobotStateProvider
        if robot_state is None:
            try:
                from providers.robot_state_provider import RobotStateProvider

                provider = RobotStateProvider()
                robot_state = provider.current_state_dict
            except Exception as e:
                logging.warning(f"Could not get robot state: {e}")
                # If we can't get state, assume safe (or could fail closed)
                return True, "", {}

        # Run simulation
        safe, reason, suggestion = self.simulator.simulate(
            action_name, params, robot_state
        )
        if not safe:
            logging.info(f"Action {action_name} blocked: {reason}")
        return safe, reason, suggestion
