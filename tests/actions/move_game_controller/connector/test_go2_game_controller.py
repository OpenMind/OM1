"""Tests for the Move Game Controller Go2 connector."""

import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

# Mock modules at module load time BEFORE any other imports
mock_zenoh = MagicMock()
mock_zenoh_msgs = MagicMock()
mock_hid = MagicMock()
mock_unitree = MagicMock()
mock_sport_client = MagicMock()
mock_unitree.unitree_sdk2py.go2.sport.sport_client.SportClient = mock_sport_client

sys.modules["zenoh"] = mock_zenoh
sys.modules["zenoh_msgs"] = mock_zenoh_msgs
sys.modules["hid"] = mock_hid
sys.modules["unitree"] = mock_unitree
sys.modules["unitree.unitree_sdk2py"] = mock_unitree.unitree_sdk2py
sys.modules["unitree.unitree_sdk2py.go2"] = mock_unitree.unitree_sdk2py.go2
sys.modules["unitree.unitree_sdk2py.go2.sport"] = mock_unitree.unitree_sdk2py.go2.sport
sys.modules["unitree.unitree_sdk2py.go2.sport.sport_client"] = (
    mock_unitree.unitree_sdk2py.go2.sport.sport_client
)

from actions.move_game_controller.connector.go2_game_controller import (  # noqa: E402
    Go2GameControllerConfig,
    Go2GameControllerConnector,
)
from actions.move_game_controller.interface import IDLEInput  # noqa: E402


@pytest.fixture
def default_config():
    """Create a default config for testing."""
    return Go2GameControllerConfig()


@pytest.fixture
def custom_config():
    """Create a custom config for testing."""
    return Go2GameControllerConfig(
        speed_x=1.2,
        speed_yaw=0.8,
        yaw_correction=0.1,
        lateral_correction=0.05,
        unitree_ethernet="eth0",
    )


@pytest.fixture
def idle_input():
    """Create an IDLEInput instance."""
    return IDLEInput(action="test_action")


@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset all mock objects between tests."""
    mock_zenoh.reset_mock()
    mock_zenoh_msgs.reset_mock()
    mock_hid.reset_mock()
    mock_sport_client.reset_mock()
    yield


class TestGo2GameControllerConfig:
    """Test the Go2GameController configuration class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = Go2GameControllerConfig()
        assert config.speed_x == 0.9
        assert config.speed_yaw == 0.6
        assert config.yaw_correction == 0.0
        assert config.lateral_correction == 0.0
        assert config.unitree_ethernet is None

    def test_custom_config(self):
        """Test custom configuration values."""
        config = Go2GameControllerConfig(
            speed_x=1.5,
            speed_yaw=1.0,
            yaw_correction=0.2,
            lateral_correction=0.1,
            unitree_ethernet="eth1",
        )
        assert config.speed_x == 1.5
        assert config.speed_yaw == 1.0
        assert config.yaw_correction == 0.2
        assert config.lateral_correction == 0.1
        assert config.unitree_ethernet == "eth1"


class TestGo2GameControllerConnector:
    """Test the Go2GameController connector."""

    @patch(
        "actions.move_game_controller.connector.go2_game_controller.UnitreeGo2StateProvider"
    )
    @patch(
        "actions.move_game_controller.connector.go2_game_controller.UnitreeGo2OdomProvider"
    )
    @patch(
        "actions.move_game_controller.connector.go2_game_controller.open_zenoh_session"
    )
    @patch("actions.move_game_controller.connector.go2_game_controller.SportClient")
    @patch("actions.move_game_controller.connector.go2_game_controller.hid")
    def test_init(
        self,
        mock_hid_module,
        mock_sport_client_class,
        mock_zenoh_session,
        mock_odom_provider,
        mock_state_provider,
        default_config,
    ):
        """Test initialization of Go2GameControllerConnector."""
        mock_client_instance = Mock()
        mock_sport_client_class.return_value = mock_client_instance
        mock_hid_module.enumerate.return_value = []

        connector = Go2GameControllerConnector(default_config)

        assert connector.move_speed == 0.9
        assert connector.turn_speed == 0.6
        mock_client_instance.SetTimeout.assert_called_once_with(10.0)
        mock_client_instance.Init.assert_called_once()

    @patch(
        "actions.move_game_controller.connector.go2_game_controller.UnitreeGo2StateProvider"
    )
    @patch(
        "actions.move_game_controller.connector.go2_game_controller.UnitreeGo2OdomProvider"
    )
    @patch(
        "actions.move_game_controller.connector.go2_game_controller.open_zenoh_session"
    )
    @patch("actions.move_game_controller.connector.go2_game_controller.SportClient")
    @patch("actions.move_game_controller.connector.go2_game_controller.hid")
    def test_init_with_custom_config(
        self,
        mock_hid_module,
        mock_sport_client_class,
        mock_zenoh_session,
        mock_odom_provider,
        mock_state_provider,
        custom_config,
    ):
        """Test initialization with custom configuration."""
        mock_client_instance = Mock()
        mock_sport_client_class.return_value = mock_client_instance
        mock_hid_module.enumerate.return_value = []

        connector = Go2GameControllerConnector(custom_config)

        assert connector.move_speed == 1.2
        assert connector.turn_speed == 0.8
        assert connector.yaw_correction == 0.1
        assert connector.lateral_correction == 0.05

    @patch(
        "actions.move_game_controller.connector.go2_game_controller.UnitreeGo2StateProvider"
    )
    @patch(
        "actions.move_game_controller.connector.go2_game_controller.UnitreeGo2OdomProvider"
    )
    @patch(
        "actions.move_game_controller.connector.go2_game_controller.open_zenoh_session"
    )
    @patch("actions.move_game_controller.connector.go2_game_controller.SportClient")
    @patch("actions.move_game_controller.connector.go2_game_controller.hid")
    def test_init_sport_client_error(
        self,
        mock_hid_module,
        mock_sport_client_class,
        mock_zenoh_session,
        mock_odom_provider,
        mock_state_provider,
        default_config,
    ):
        """Test initialization when SportClient fails."""
        mock_sport_client_class.side_effect = Exception("Connection error")
        mock_hid_module.enumerate.return_value = []

        connector = Go2GameControllerConnector(default_config)

        assert connector.sport_client is None

    @patch(
        "actions.move_game_controller.connector.go2_game_controller.UnitreeGo2StateProvider"
    )
    @patch(
        "actions.move_game_controller.connector.go2_game_controller.UnitreeGo2OdomProvider"
    )
    @patch(
        "actions.move_game_controller.connector.go2_game_controller.open_zenoh_session"
    )
    @patch("actions.move_game_controller.connector.go2_game_controller.SportClient")
    @patch("actions.move_game_controller.connector.go2_game_controller.hid")
    @pytest.mark.asyncio
    async def test_connect(
        self,
        mock_hid_module,
        mock_sport_client_class,
        mock_zenoh_session,
        mock_odom_provider,
        mock_state_provider,
        default_config,
        idle_input,
    ):
        """Test connect method (passes through)."""
        mock_client_instance = Mock()
        mock_sport_client_class.return_value = mock_client_instance
        mock_hid_module.enumerate.return_value = []

        connector = Go2GameControllerConnector(default_config)
        # connect is a pass-through, should not raise
        await connector.connect(idle_input)
