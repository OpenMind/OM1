"""Tests for EnvironmentModelProvider."""

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.providers.environment_model_provider import (
    EnvironmentModelProvider,
    Obstacle,
)


@pytest.fixture
def provider():
    """Create an EnvironmentModelProvider instance and clean up."""
    # Reset singleton
    EnvironmentModelProvider._singleton_instance = None  # type: ignore
    prov = EnvironmentModelProvider()
    yield prov
    if prov._running:
        prov.stop()


class TestEnvironmentModelProvider:
    """Test suite for EnvironmentModelProvider."""

    def test_singleton(self):
        """Test that EnvironmentModelProvider is a singleton."""
        p1 = EnvironmentModelProvider()
        p2 = EnvironmentModelProvider()
        assert p1 is p2

    def test_start_stop(self, provider):
        """Test starting and stopping."""
        assert not provider._running
        provider.start()
        assert provider._running
        assert provider._update_thread is not None
        assert provider._update_thread.is_alive()
        provider.stop()
        assert not provider._running
        time.sleep(0.1)
        assert not provider._update_thread.is_alive()

    def test_register_providers(self, provider):
        """Test registering LiDAR and AMCL providers."""
        mock_lidar = MagicMock()
        mock_amcl = MagicMock()
        provider.register_providers(lidar=mock_lidar, amcl=mock_amcl)
        assert provider._lidar_provider is mock_lidar
        assert provider._amcl_provider is mock_amcl

    def test_update_model_with_lidar(self, provider):
        """Test updating model from LiDAR data."""
        provider.start()
        try:
            mock_lidar = MagicMock()
            # Simulate raw_scan as numpy array with points
            mock_lidar.raw_scan = np.array(
                [[0.5, 0.2, 30.0, 0.8], [-0.3, 0.4, 150.0, 0.5]]
            )
            provider.register_providers(lidar=mock_lidar)
            time.sleep(0.6)  # Allow update loop to run
            model = provider.current_model
            assert len(model.obstacles) == 2
            assert model.obstacles[0].x == 0.5
            assert model.obstacles[0].y == 0.2
            assert model.obstacles[0].radius == 0.1
        finally:
            provider.stop()

    def test_update_model_with_amcl(self, provider):
        """Test updating model from AMCL data."""
        provider.start()
        try:
            mock_amcl = MagicMock()
            mock_amcl.pose = MagicMock()
            mock_amcl.pose.position.x = 1.0
            mock_amcl.pose.position.y = 2.0
            provider.register_providers(amcl=mock_amcl)
            time.sleep(0.6)
            model = provider.current_model
            assert model.map_origin_x == 1.0
            assert model.map_origin_y == 2.0
        finally:
            provider.stop()

    def test_check_collision(self, provider):
        """Test collision checking."""
        # Manually set obstacles
        provider._model.obstacles = [Obstacle(x=1.0, y=1.0, radius=0.2)]
        # Point inside obstacle
        assert provider.check_collision(1.0, 1.0, robot_radius=0.3) is True
        # Point far away
        assert provider.check_collision(5.0, 5.0, robot_radius=0.3) is False
        # Point near but not colliding
        assert provider.check_collision(1.3, 1.0, robot_radius=0.1) is False

    def test_obstacles_property(self, provider):
        """Test obstacles property returns copy."""
        provider._model.obstacles = [Obstacle(x=1.0, y=1.0)]
        obs = provider.obstacles
        assert len(obs) == 1
        assert obs[0].x == 1.0
        # Modifying returned list should not affect internal
        obs.append(Obstacle(x=2.0, y=2.0))
        assert len(provider._model.obstacles) == 1
