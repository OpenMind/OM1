"""Tests for RobotStateProvider."""

import time
from unittest.mock import MagicMock, patch

import pytest

from src.providers.robot_state import RobotState
from src.providers.robot_state_provider import RobotStateProvider


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton before each test."""
    RobotStateProvider._singleton_instance = None  # type: ignore
    yield
    instance = RobotStateProvider._singleton_instance  # type: ignore
    if instance and instance._running:
        instance.stop()
    RobotStateProvider._singleton_instance = None  # type: ignore


class TestRobotStateProvider:
    """Test suite for RobotStateProvider."""

    def test_singleton(self):
        p1 = RobotStateProvider()
        p2 = RobotStateProvider()
        assert p1 is p2

    def test_start_stop(self):
        provider = RobotStateProvider()
        assert not provider._running
        provider.start()
        assert provider._running
        assert provider._update_thread is not None
        assert provider._update_thread.is_alive()
        provider.stop()
        assert not provider._running
        time.sleep(0.1)
        assert not provider._update_thread.is_alive()

    def test_start_already_running(self):
        """start() when already running logs warning and returns. Covers lines 50-51."""
        provider = RobotStateProvider()
        provider.start()
        try:
            with patch("src.providers.robot_state_provider.logging") as mock_log:
                provider.start()  # second call
                mock_log.warning.assert_called_with(
                    "RobotStateProvider already running"
                )
        finally:
            provider.stop()

    def test_initial_state(self):
        provider = RobotStateProvider()
        state = provider.current_state
        assert isinstance(state, RobotState)
        assert state.position.x == 0.0

    def test_register_providers(self):
        provider = RobotStateProvider()
        mock_odom = MagicMock()
        mock_state = MagicMock()
        provider.register_providers(odom=mock_odom, state_prov=mock_state)
        assert provider._odom_provider is mock_odom
        assert provider._state_provider is mock_state

    def test_register_providers_with_teleops(self):
        """register_providers sets teleops provider. Covers line 154."""
        provider = RobotStateProvider()
        mock_teleops = MagicMock()
        provider.register_providers(teleops=mock_teleops)
        assert provider._teleops_provider is mock_teleops

    def test_update_loop_exception_is_caught(self):
        """Exception in _update_state is caught and logged. Covers lines 72-73."""
        provider = RobotStateProvider()
        with patch.object(provider, "_update_state", side_effect=Exception("boom")):
            with patch("src.providers.robot_state_provider.logging") as mock_log:
                provider.start()
                time.sleep(0.2)
                provider.stop()
                mock_log.error.assert_called()
                args = mock_log.error.call_args[0][0]
                assert "boom" in args

    def test_update_state_with_odom(self):
        provider = RobotStateProvider()
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
            time.sleep(0.2)
            state = provider.current_state
            assert state.position.x == 1.0
            assert state.position.y == 2.0
            assert state.position.yaw == 90.0
            assert state.is_moving is True
        finally:
            provider.stop()

    def test_update_state_with_state_provider(self):
        provider = RobotStateProvider()
        provider.start()
        try:
            mock_state_prov = MagicMock()
            mock_state_prov.state = "standing"
            provider.register_providers(state_prov=mock_state_prov)
            time.sleep(0.2)
            assert provider.current_state.body_state == "standing"
        finally:
            provider.stop()

    def test_update_state_with_amcl(self):
        provider = RobotStateProvider()
        provider.start()
        try:
            mock_amcl = MagicMock()
            mock_amcl.is_localized = True
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

    def test_update_state_with_lidar(self):
        provider = RobotStateProvider()
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
            assert "move back" in state.safe_paths
            assert state.obstacles_nearby is False
        finally:
            provider.stop()

    def test_current_state_dict(self):
        provider = RobotStateProvider()
        provider.start()
        try:
            state_dict = provider.current_state_dict
            assert isinstance(state_dict, dict)
            assert "position" in state_dict
        finally:
            provider.stop()
