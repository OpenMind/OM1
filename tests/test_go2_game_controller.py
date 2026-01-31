"""
Unit tests for Go2GameControllerConnector.
"""

import sys
from unittest.mock import MagicMock, Mock, patch

import pytest


def test_go2_game_controller_config_defaults():
    module_name = "src.actions.move_game_controller.connector.go2_game_controller"
    if module_name in sys.modules:
        del sys.modules[module_name]

    sys.modules["unitree.unitree_sdk2py.go2.sport.sport_client"] = Mock()
    sys.modules["hid"] = Mock()
    sys.modules["zenoh"] = Mock()
    sys.modules["zenoh_msgs"] = Mock()
    sys.modules["providers.odom_provider"] = Mock()
    sys.modules["providers.unitree_go2_state_provider"] = Mock()

    from src.actions.move_game_controller.connector.go2_game_controller import (
        Go2GameControllerConfig,
    )

    config = Go2GameControllerConfig()

    assert config.speed_x == 0.9
    assert config.speed_yaw == 0.6
    assert config.yaw_correction == 0.0
    assert config.lateral_correction == 0.0
    assert config.unitree_ethernet is None


def test_go2_game_controller_config_custom_values():
    module_name = "src.actions.move_game_controller.connector.go2_game_controller"
    if module_name in sys.modules:
        del sys.modules[module_name]

    sys.modules["unitree.unitree_sdk2py.go2.sport.sport_client"] = Mock()
    sys.modules["hid"] = Mock()
    sys.modules["zenoh"] = Mock()
    sys.modules["zenoh_msgs"] = Mock()
    sys.modules["providers.odom_provider"] = Mock()
    sys.modules["providers.unitree_go2_state_provider"] = Mock()

    from src.actions.move_game_controller.connector.go2_game_controller import (
        Go2GameControllerConfig,
    )

    config = Go2GameControllerConfig(
        speed_x=1.0,
        speed_yaw=0.5,
        yaw_correction=0.1,
        lateral_correction=0.1,
        unitree_ethernet="eth0",
    )

    assert config.speed_x == 1.0
    assert config.speed_yaw == 0.5
    assert config.yaw_correction == 0.1
    assert config.lateral_correction == 0.1
    assert config.unitree_ethernet == "eth0"


def test_go2_game_controller_initialization_with_mocks():
    module_name = "src.actions.move_game_controller.connector.go2_game_controller"
    if module_name in sys.modules:
        del sys.modules[module_name]

    sys.modules["unitree.unitree_sdk2py.go2.sport.sport_client"] = Mock()
    mock_hid = MagicMock()
    mock_hid.enumerate.return_value = []
    sys.modules["hid"] = mock_hid
    sys.modules["zenoh"] = Mock()
    sys.modules["zenoh_msgs"] = Mock()
    sys.modules["providers.odom_provider"] = Mock()
    sys.modules["providers.unitree_go2_state_provider"] = Mock()

    with (
        patch(f"{module_name}.open_zenoh_session"),
        patch(f"{module_name}.OdomProvider"),
        patch(f"{module_name}.UnitreeGo2StateProvider"),
        patch(f"{module_name}.SportClient"),
    ):
        from src.actions.move_game_controller.connector.go2_game_controller import (
            Go2GameControllerConfig,
            Go2GameControllerConnector,
        )

        config = Go2GameControllerConfig(
            speed_x=1.0,
            speed_yaw=0.5,
            yaw_correction=0.1,
            lateral_correction=0.1,
            unitree_ethernet="eth0",
        )
        connector = Go2GameControllerConnector(config=config)

        assert connector.config == config
        assert connector.move_speed == 1.0
        assert connector.turn_speed == 0.5
        assert connector.gamepad is None
        assert connector.sony_dualsense is False
        assert connector.xbox is False
        assert connector.sony_edge is False


def test_init_controller_no_controllers_with_mocks():
    module_name = "src.actions.move_game_controller.connector.go2_game_controller"
    if module_name in sys.modules:
        del sys.modules[module_name]

    sys.modules["unitree.unitree_sdk2py.go2.sport.sport_client"] = Mock()
    mock_hid = MagicMock()
    mock_hid.enumerate.return_value = []
    sys.modules["hid"] = mock_hid
    sys.modules["zenoh"] = Mock()
    sys.modules["zenoh_msgs"] = Mock()
    sys.modules["providers.odom_provider"] = Mock()
    sys.modules["providers.unitree_go2_state_provider"] = Mock()

    with (
        patch(f"{module_name}.open_zenoh_session"),
        patch(f"{module_name}.OdomProvider"),
        patch(f"{module_name}.UnitreeGo2StateProvider"),
        patch(f"{module_name}.SportClient"),
    ):
        from src.actions.move_game_controller.connector.go2_game_controller import (
            Go2GameControllerConnector,
        )

        mock_config = Mock()
        mock_config.speed_x = 1.0
        mock_config.speed_yaw = 0.5

        connector = Go2GameControllerConnector(config=mock_config)
        connector._init_controller()

        assert connector.gamepad is None
        assert connector.sony_dualsense is False
        assert connector.xbox is False
        assert connector.sony_edge is False


@patch("time.sleep", return_value=None)
def test_tick_method_with_mocks(mock_sleep):
    module_name = "src.actions.move_game_controller.connector.go2_game_controller"
    if module_name in sys.modules:
        del sys.modules[module_name]

    sys.modules["unitree.unitree_sdk2py.go2.sport.sport_client"] = Mock()
    mock_hid = MagicMock()
    mock_hid.enumerate.return_value = []
    sys.modules["hid"] = mock_hid
    sys.modules["zenoh"] = Mock()
    sys.modules["zenoh_msgs"] = Mock()
    sys.modules["providers.odom_provider"] = Mock()
    sys.modules["providers.unitree_go2_state_provider"] = Mock()

    with (
        patch(f"{module_name}.open_zenoh_session"),
        patch(f"{module_name}.OdomProvider"),
        patch(f"{module_name}.UnitreeGo2StateProvider"),
        patch(f"{module_name}.SportClient"),
    ):
        from src.actions.move_game_controller.connector.go2_game_controller import (
            Go2GameControllerConnector,
        )

        mock_config = Mock()
        mock_config.speed_x = 1.0
        mock_config.speed_yaw = 0.5

        connector = Go2GameControllerConnector(config=mock_config)
        connector.tick()


@pytest.mark.asyncio
async def test_connect_method_with_mocks():
    module_name = "src.actions.move_game_controller.connector.go2_game_controller"
    if module_name in sys.modules:
        del sys.modules[module_name]

    sys.modules["unitree.unitree_sdk2py.go2.sport.sport_client"] = Mock()
    mock_hid = MagicMock()
    mock_hid.enumerate.return_value = []
    sys.modules["hid"] = mock_hid
    sys.modules["zenoh"] = Mock()
    sys.modules["zenoh_msgs"] = Mock()
    sys.modules["providers.odom_provider"] = Mock()
    sys.modules["providers.unitree_go2_state_provider"] = Mock()

    with (
        patch(f"{module_name}.open_zenoh_session"),
        patch(f"{module_name}.OdomProvider"),
        patch(f"{module_name}.UnitreeGo2StateProvider"),
        patch(f"{module_name}.SportClient"),
    ):
        from src.actions.move_game_controller.connector.go2_game_controller import (
            Go2GameControllerConnector,
            IDLEInput,
        )

        mock_config = Mock()
        mock_config.speed_x = 1.0
        mock_config.speed_yaw = 0.5

        connector = Go2GameControllerConnector(config=mock_config)
        mock_idle = Mock(spec=IDLEInput)
        await connector.connect(mock_idle)
