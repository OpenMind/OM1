"""Tests for EnvironmentModelProvider."""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.providers.environment_model_provider import (
    EnvironmentModel,
    EnvironmentModelProvider,
    Obstacle,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    EnvironmentModelProvider._singleton_instance = None  # type: ignore
    yield
    instance = EnvironmentModelProvider._singleton_instance  # type: ignore
    if instance and instance._running:
        instance.stop()
    EnvironmentModelProvider._singleton_instance = None  # type: ignore


class TestEnvironmentModelProvider:

    def test_singleton(self):
        p1 = EnvironmentModelProvider()
        p2 = EnvironmentModelProvider()
        assert p1 is p2

    def test_start_stop(self):
        provider = EnvironmentModelProvider()
        assert not provider._running
        provider.start()
        assert provider._running
        assert provider._update_thread is not None
        assert provider._update_thread.is_alive()
        provider.stop()
        assert not provider._running
        time.sleep(0.1)
        assert provider._update_thread is not None
        assert not provider._update_thread.is_alive()

    def test_start_already_running(self):
        """start() when already running logs warning and returns. Covers lines 84-85."""
        provider = EnvironmentModelProvider()
        provider.start()
        try:
            with patch("src.providers.environment_model_provider.logging") as mock_log:
                provider.start()
                mock_log.warning.assert_called_with(
                    "EnvironmentModelProvider already running"
                )
        finally:
            provider.stop()

    def test_update_loop_exception_is_caught(self):
        """Exception in _update_model is caught and logged. Covers lines 105-106."""
        provider = EnvironmentModelProvider()
        with patch.object(provider, "_update_model", side_effect=Exception("boom")):
            with patch("src.providers.environment_model_provider.logging") as mock_log:
                provider.start()
                time.sleep(0.7)
                provider.stop()
                mock_log.error.assert_called()
                assert "boom" in mock_log.error.call_args[0][0]

    def test_register_providers(self):
        provider = EnvironmentModelProvider()
        mock_lidar = MagicMock()
        mock_amcl = MagicMock()
        provider.register_providers(lidar=mock_lidar, amcl=mock_amcl)
        assert provider._lidar_provider is mock_lidar
        assert provider._amcl_provider is mock_amcl

    def test_update_model_with_lidar(self):
        provider = EnvironmentModelProvider()
        provider.start()
        try:
            mock_lidar = MagicMock()
            mock_lidar.raw_scan = np.array(
                [[0.5, 0.2, 30.0, 0.8], [-0.3, 0.4, 150.0, 0.5]]
            )
            provider.register_providers(lidar=mock_lidar)
            time.sleep(0.6)
            model = provider.current_model
            assert len(model.obstacles) == 2
            assert model.obstacles[0].x == 0.5
            assert model.obstacles[0].y == 0.2
            assert model.obstacles[0].radius == 0.1
        finally:
            provider.stop()

    def test_update_model_with_amcl(self):
        provider = EnvironmentModelProvider()
        provider.start()
        try:
            mock_amcl = MagicMock()
            mock_amcl.pose.position.x = 1.0
            mock_amcl.pose.position.y = 2.0
            provider.register_providers(amcl=mock_amcl)
            time.sleep(0.6)
            model = provider.current_model
            assert model.map_origin_x == 1.0
            assert model.map_origin_y == 2.0
        finally:
            provider.stop()

    def test_check_collision(self):
        provider = EnvironmentModelProvider()
        provider._model.obstacles = [Obstacle(x=1.0, y=1.0, radius=0.2)]
        assert provider.check_collision(1.0, 1.0, robot_radius=0.3) is True
        assert provider.check_collision(5.0, 5.0, robot_radius=0.3) is False
        assert provider.check_collision(1.3, 1.0, robot_radius=0.1) is False

    def test_obstacles_property(self):
        provider = EnvironmentModelProvider()
        provider._model.obstacles = [Obstacle(x=1.0, y=1.0)]
        obs = provider.obstacles
        assert len(obs) == 1
        obs.append(Obstacle(x=2.0, y=2.0))
        assert len(provider._model.obstacles) == 1

    def test_environment_model_to_dict(self):
        """EnvironmentModel.to_dict() serializes correctly. Covers line 49."""
        model = EnvironmentModel(
            map_origin_x=1.0,
            map_origin_y=2.0,
            map_resolution=0.05,
            map_width=200,
            map_height=200,
        )
        model.obstacles = [Obstacle(x=0.5, y=0.3, radius=0.2)]
        result = model.to_dict()
        assert result["map_origin_x"] == 1.0
        assert result["map_origin_y"] == 2.0
        assert len(result["obstacles"]) == 1
        assert result["obstacles"][0] == {"x": 0.5, "y": 0.3, "radius": 0.2}
        assert "occupancy_grid" not in result
