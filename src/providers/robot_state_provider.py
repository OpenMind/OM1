"""
Robot State Provider that consolidates state from various sources.
"""

import logging
import threading
import time
from typing import Any, Dict, Optional

from .robot_state import RobotState
from .singleton import singleton

# Import existing providers (will be imported lazily to avoid circular imports)
# from .unitree_go2_odom_provider import UnitreeGo2OdomProvider
# from .unitree_go2_state_provider import UnitreeGo2StateProvider
# from .unitree_go2_amcl_provider import UnitreeGo2AMCLProvider
# from .unitree_go2_rplidar_provider import UnitreeGo2RPLidarProvider
# from .teleops_status_provider import TeleopsStatusProvider


@singleton
class RobotStateProvider:
    """
    Singleton provider that consolidates robot state from multiple sources.

    This provider polls various other providers and maintains a unified
    RobotState object that represents the current state of the robot.
    """

    def __init__(self):
        """Initialize the RobotStateProvider."""
        self._lock = threading.Lock()
        self._state = RobotState()
        self._running = False
        self._update_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # References to other providers (will be initialized on demand)
        self._odom_provider = None
        self._state_provider = None
        self._amcl_provider = None
        self._lidar_provider = None
        self._teleops_provider = None

        logging.info("RobotStateProvider initialized")

    def start(self):
        """Start the background update thread."""
        if self._running:
            logging.warning("RobotStateProvider already running")
            return

        self._running = True
        self._stop_event.clear()
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._update_thread.start()
        logging.info("RobotStateProvider started")

    def stop(self):
        """Stop the background update thread."""
        self._running = False
        self._stop_event.set()
        if self._update_thread:
            self._update_thread.join(timeout=2.0)
        logging.info("RobotStateProvider stopped")

    def _update_loop(self):
        """Background thread that periodically updates state from all providers."""
        while not self._stop_event.is_set():
            try:
                self._update_state()
            except Exception as e:
                logging.error(f"Error updating robot state: {e}")

            # Update at ~10Hz
            self._stop_event.wait(timeout=0.1)

    def _update_state(self):
        """Update state from all available providers."""
        with self._lock:
            new_state = RobotState()

            # Update from odometry provider
            if self._odom_provider and hasattr(self._odom_provider, "position"):
                pos = self._odom_provider.position
                if pos:
                    new_state.position.x = pos.get("odom_x", 0.0)
                    new_state.position.y = pos.get("odom_y", 0.0)
                    new_state.position.yaw = pos.get("odom_yaw_0_360", 0.0)
                    new_state.is_moving = pos.get("moving", False)

            # Update from state provider (body state)
            if self._state_provider and hasattr(self._state_provider, "state"):
                body_state = self._state_provider.state
                if body_state:
                    new_state.body_state = body_state

            # Update from AMCL (localization)
            if self._amcl_provider:
                if hasattr(self._amcl_provider, "is_localized"):
                    new_state.is_localized = self._amcl_provider.is_localized
                if hasattr(self._amcl_provider, "pose"):
                    pose = self._amcl_provider.pose
                    if pose and hasattr(pose, "position"):
                        new_state.localization_pose = {
                            "x": pose.position.x,
                            "y": pose.position.y,
                            "z": pose.position.z,
                        }

            # Update from LiDAR (safe paths)
            if self._lidar_provider and hasattr(
                self._lidar_provider, "movement_options"
            ):
                movement = self._lidar_provider.movement_options
                paths = []
                if movement.get("advance"):
                    paths.append("move forwards")
                if movement.get("turn_left"):
                    paths.append("turn left")
                if movement.get("turn_right"):
                    paths.append("turn right")
                if movement.get("retreat"):
                    paths.append("move back")
                new_state.safe_paths = paths
                new_state.obstacles_nearby = (
                    len(paths) < 4
                )  # if not all paths available

            # TODO: Update from battery (via TeleopsStatusProvider)

            new_state.timestamp = time.time()
            self._state = new_state

    def register_providers(
        self,
        odom=None,
        state_prov=None,
        amcl=None,
        lidar=None,
        teleops=None,
    ):
        """Register provider instances to use for state updates."""
        with self._lock:
            if odom:
                self._odom_provider = odom
            if state_prov:
                self._state_provider = state_prov
            if amcl:
                self._amcl_provider = amcl
            if lidar:
                self._lidar_provider = lidar
            if teleops:
                self._teleops_provider = teleops

    @property
    def current_state(self) -> RobotState:
        """Get the current robot state."""
        with self._lock:
            return self._state

    @property
    def current_state_dict(self) -> Dict[str, Any]:
        """Get current state as dictionary."""
        return self.current_state.to_dict()
