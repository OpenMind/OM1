"""
Robot State data structure for consolidating robot state information.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Position:
    """2D position with orientation."""

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0  # in degrees, 0-360


@dataclass
class BatteryStatus:
    """Battery status information."""

    percentage: float = 100.0
    voltage: float = 0.0
    temperature: float = 0.0
    charging: bool = False


@dataclass
class RobotState:
    """Complete robot state consolidated from various providers."""

    # Timestamp
    timestamp: float = field(default_factory=time.time)

    # Position and movement
    position: Position = field(default_factory=Position)
    is_moving: bool = False
    body_state: str = "unknown"  # "standing", "sitting", etc.

    # Battery
    battery: BatteryStatus = field(default_factory=BatteryStatus)

    # Localization
    is_localized: bool = False
    localization_pose: Optional[Dict[str, float]] = None  # from AMCL

    # Environment perception
    safe_paths: List[str] = field(
        default_factory=list
    )  # e.g., ["move forwards", "turn left"]
    obstacles_nearby: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy serialization."""
        return {
            "timestamp": self.timestamp,
            "position": {
                "x": self.position.x,
                "y": self.position.y,
                "yaw": self.position.yaw,
            },
            "is_moving": self.is_moving,
            "body_state": self.body_state,
            "battery": {
                "percentage": self.battery.percentage,
                "voltage": self.battery.voltage,
                "temperature": self.battery.temperature,
                "charging": self.battery.charging,
            },
            "is_localized": self.is_localized,
            "localization_pose": self.localization_pose,
            "safe_paths": self.safe_paths,
            "obstacles_nearby": self.obstacles_nearby,
        }
