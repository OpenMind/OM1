"""Tests for RobotStateProvider."""

import time
from unittest.mock import MagicMock

import pytest

from src.providers.robot_state import RobotState
from src.providers.robot_state_provider import RobotStateProvider


@pytest.fixture
def provider():
    """Create a RobotStateProvider instance and clean up after test."""
    prov = RobotStateProvider()
    # Reset singleton for testing (since it's a singleton)
    RobotStateProvider._singleton_instance = None  # type: ignore
    yield prov
    if prov._running:
        prov.stop()


class TestRobotStateProvider:
    """Test suite for RobotStateProvider."""

    def test_singleton(self):
        """Test that RobotStateProvider is a singleton."""
        p1 = RobotStateProvider()
        p2 = RobotStateProvider()
        assert p1 is p2

    def test_start_stop(self, provider):
        """Test starting and stopping the provider."""
        assert not provider._running
        provider.start()
        assert provider._running
        assert provider._update_thread is not None
        assert provider._update_thread.is_alive()
        provider.stop()
        assert not provider._running
        # Thread should have stopped
        time.sleep(0.1)
        assert not provider._update_thread.is_alive()

    def test_initial_state(self, provider):
        """Test initial state is default RobotState."""
        state = provider.current_state
        assert isinstance(state, RobotState)
        assert state.position.x == 0.0

    def test_register_providers(self, provider):
        """Test registering providers."""
        mock_odom = MagicMock()
        mock_state = MagicMock()
        provider.register_providers(odom=mock_odom, state_prov=mock_state)
        assert provider._odom_provider is mock_odom
        assert provider._state_provider is mock_state

    def test_update_state_with_odom(self, provider):
        """Test state update from odometry provider."""
        provider.start()
        try:
            mock_odom = MagicMock()
            mock_odom.position = {
                "odom_x": 1.0,
                "odom_y": 2.0,
                "odom_yaw_0_360": 90.0,
                "moving": True,
            }
            provider.register_providers(odom=mock_odom)
            time.sleep(0.2)  # Let update loop run
            state = provider.current_state
            assert state.position.x == 1.0
            assert state.position.y == 2.0
            assert state.position.yaw == 90.0
            assert state.is_moving is True
        finally:
            provider.stop()

    def test_update_state_with_state_provider(self, provider):
        """Test state update from state provider."""
        provider.start()
        try:
            mock_state_prov = MagicMock()
            mock_state_prov.state = "standing"
            provider.register_providers(state_prov=mock_state_prov)
            time.sleep(0.2)
            state = provider.current_state
            assert state.body_state == "standing"
        finally:
            provider.stop()

    def test_update_state_with_amcl(self, provider):
        """Test state update from AMCL provider."""
        provider.start()
        try:
            mock_amcl = MagicMock()
            mock_amcl.is_localized = True
            mock_amcl.pose = MagicMock()
            mock_amcl.pose.position.x = 10.0
            mock_amcl.pose.position.y = 20.0
            mock_amcl.pose.position.z = 0.5
            provider.register_providers(amcl=mock_amcl)
            time.sleep(0.2)
            state = provider.current_state
            assert state.is_localized is True
            assert state.localization_pose == {"x": 10.0, "y": 20.0, "z": 0.5}
        finally:
            provider.stop()

    def test_update_state_with_lidar(self, provider):
        """Test state update from LiDAR provider."""
        provider.start()
        try:
            mock_lidar = MagicMock()
            mock_lidar.movement_options = {
                "advance": [3, 4, 5],
                "turn_left": [0, 1, 2],
                "turn_right": [6, 7, 8],
                "retreat": True,
            }
            provider.register_providers(lidar=mock_lidar)
            time.sleep(0.2)
            state = provider.current_state
            assert "move forwards" in state.safe_paths
            assert "turn left" in state.safe_paths
            assert "turn right" in state.safe_paths
            assert "move back" in state.safe_paths
            assert state.obstacles_nearby is False
        finally:
            provider.stop()

    def test_current_state_dict(self, provider):
        """Test current_state_dict property."""
        provider.start()
        try:
            state_dict = provider.current_state_dict
            assert isinstance(state_dict, dict)
            assert "position" in state_dict
        finally:
            provider.stop()
