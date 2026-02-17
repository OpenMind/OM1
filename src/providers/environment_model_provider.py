"""
Environment Model Provider for maintaining a representation of the robot's surroundings.

This provider aggregates data from LiDAR, cameras, and other sensors to build
a model of the environment (obstacles, map, etc.) that can be used for safety
simulation and path planning.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .singleton import singleton
from .unitree_go2_amcl_provider import UnitreeGo2AMCLProvider
from .unitree_go2_rplidar_provider import UnitreeGo2RPLidarProvider


@dataclass
class Obstacle:
    """Represents an obstacle in the environment."""

    x: float
    y: float
    radius: float = 0.2  # approximate radius in meters
    confidence: float = 1.0


@dataclass
class EnvironmentModel:
    """
    Current model of the environment.
    """

    timestamp: float = field(default_factory=time.time)
    obstacles: List[Obstacle] = field(default_factory=list)
    map_origin_x: float = 0.0
    map_origin_y: float = 0.0
    map_resolution: float = 0.05  # meters per pixel
    map_width: int = 100
    map_height: int = 100
    occupancy_grid: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "obstacles": [
                {"x": o.x, "y": o.y, "radius": o.radius} for o in self.obstacles
            ],
            "map_origin_x": self.map_origin_x,
            "map_origin_y": self.map_origin_y,
            "map_resolution": self.map_resolution,
            "map_width": self.map_width,
            "map_height": self.map_height,
            # occupancy grid not serialized by default (too large)
        }


@singleton
class EnvironmentModelProvider:
    """
    Provider that builds and maintains a model of the robot's environment.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._model = EnvironmentModel()
        self._running = False
        self._update_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._lidar_provider: Optional[UnitreeGo2RPLidarProvider] = None
        self._amcl_provider: Optional[UnitreeGo2AMCLProvider] = None

        logging.info("EnvironmentModelProvider initialized")

    def start(self):
        """Start the background update thread."""
        if self._running:
            logging.warning("EnvironmentModelProvider already running")
            return
        self._running = True
        self._stop_event.clear()
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._update_thread.start()
        logging.info("EnvironmentModelProvider started")

    def stop(self):
        """Stop the background update thread."""
        self._running = False
        self._stop_event.set()
        if self._update_thread:
            self._update_thread.join(timeout=2.0)
        logging.info("EnvironmentModelProvider stopped")

    def _update_loop(self):
        """Background thread that periodically updates the environment model."""
        while not self._stop_event.is_set():
            try:
                self._update_model()
            except Exception as e:
                logging.error(f"Error updating environment model: {e}")
            self._stop_event.wait(timeout=0.5)  # 2 Hz update

    def _update_model(self):
        """Update the environment model from available sensor data."""
        with self._lock:
            new_model = EnvironmentModel()

            # Update from LiDAR if available
            if self._lidar_provider and self._lidar_provider.raw_scan is not None:
                # Convert raw scan points to obstacles
                # raw_scan is a numpy array with columns [x, y, angle, distance]
                scan = self._lidar_provider.raw_scan
                if scan is not None and len(scan) > 0:
                    obstacles = []
                    for point in scan:
                        # Assuming point format: [x, y, angle, distance]
                        # x,y are in robot frame? We need to transform to world frame if possible
                        # For now, just store as obstacles in robot frame
                        obs = Obstacle(x=float(point[0]), y=float(point[1]), radius=0.1)
                        obstacles.append(obs)
                    new_model.obstacles = obstacles

            # Update from AMCL if available (for map origin, etc.)
            if self._amcl_provider and self._amcl_provider.pose:
                pose = self._amcl_provider.pose
                new_model.map_origin_x = pose.position.x
                new_model.map_origin_y = pose.position.y
                # In future, could integrate with a real map

            new_model.timestamp = time.time()
            self._model = new_model

    def register_providers(
        self,
        lidar: Optional[UnitreeGo2RPLidarProvider] = None,
        amcl: Optional[UnitreeGo2AMCLProvider] = None,
    ):
        """Register provider instances to use for model updates."""
        with self._lock:
            if lidar:
                self._lidar_provider = lidar
            if amcl:
                self._amcl_provider = amcl

    @property
    def current_model(self) -> EnvironmentModel:
        """Get the current environment model."""
        with self._lock:
            return self._model

    @property
    def obstacles(self) -> List[Obstacle]:
        """Get current obstacles."""
        with self._lock:
            return self._model.obstacles.copy()

    def check_collision(self, x: float, y: float, robot_radius: float = 0.3) -> bool:
        """
        Check if a point (x,y) collides with any obstacle.

        Args:
            x, y: coordinates in robot frame (or world frame, depending on model)
            robot_radius: radius of the robot

        Returns
        -------
            True if collision, False otherwise
        """
        with self._lock:
            for obs in self._model.obstacles:
                dx = x - obs.x
                dy = y - obs.y
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < (robot_radius + obs.radius):
                    return True
            return False
