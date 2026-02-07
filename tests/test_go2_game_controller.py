"""
Unit tests for Go2GameControllerConnector.
"""

import sys
from unittest.mock import MagicMock, Mock, patch

import pytest


@pytest.fixture(autouse=True)
def setup_mock_modules():
    """Setup mock modules before each test."""
    module_name = "src.actions.move_game_controller.connector.go2_game_controller"

    # Clean up module if already imported
    if module_name in sys.modules:
        del sys.modules[module_name]

    # Mock all external dependencies
    sys.modules["unitree.unitree_sdk2py.go2.sport.sport_client"] = Mock()
    sys.modules["hid"] = Mock()
    sys.modules["zenoh"] = Mock()
    sys.modules["zenoh_msgs"] = Mock()
    sys.modules["providers.odom_provider"] = Mock()
    sys.modules["providers.unitree_go2_state_provider"] = Mock()

    yield

    # Cleanup after test (optional)


@pytest.fixture
def mock_hid_no_controllers():
    """Mock HID with no controllers found."""
    mock_hid = MagicMock()
    mock_hid.enumerate.return_value = []
    sys.modules["hid"] = mock_hid
    return mock_hid


@pytest.fixture
def mock_patches():
    """Provide common patches for connector tests."""
    with (
        patch(
            "src.actions.move_game_controller.connector.go2_game_controller.open_zenoh_session"
        ) as mock_zenoh,
        patch(
            "src.actions.move_game_controller.connector.go2_game_controller.OdomProvider"
        ) as mock_odom,
        patch(
            "src.actions.move_game_controller.connector.go2_game_controller.UnitreeGo2StateProvider"
        ) as mock_state,
        patch(
            "src.actions.move_game_controller.connector.go2_game_controller.SportClient"
        ) as mock_sport,
    ):

        yield {
            "zenoh_session": mock_zenoh,
            "odom_provider": mock_odom,
            "state_provider": mock_state,
            "sport_client": mock_sport,
        }


@pytest.fixture
def default_config():
    """Create default config for testing."""
    from src.actions.move_game_controller.connector.go2_game_controller import (
        Go2GameControllerConfig,
    )

    return Go2GameControllerConfig()


@pytest.fixture
def custom_config():
    """Create custom config for testing."""
    from src.actions.move_game_controller.connector.go2_game_controller import (
        Go2GameControllerConfig,
    )

    return Go2GameControllerConfig(
        speed_x=1.0,
        speed_yaw=0.5,
        yaw_correction=0.1,
        lateral_correction=0.1,
        unitree_ethernet="eth0",
    )


@pytest.fixture
def mock_config():
    """Create a mock config object."""
    config = Mock()
    config.speed_x = 1.0
    config.speed_yaw = 0.5
    return config


class TestGo2GameControllerConfig:
    """Test configuration class."""

    def test_default_config(self, default_config):
        """Test default configuration values."""
        assert default_config.speed_x == 0.9
        assert default_config.speed_yaw == 0.6
        assert default_config.yaw_correction == 0.0
        assert default_config.lateral_correction == 0.0
        assert default_config.unitree_ethernet is None

    def test_custom_config(self, custom_config):
        """Test custom configuration values."""
        assert custom_config.speed_x == 1.0
        assert custom_config.speed_yaw == 0.5
        assert custom_config.yaw_correction == 0.1
        assert custom_config.lateral_correction == 0.1
        assert custom_config.unitree_ethernet == "eth0"


class TestGo2GameControllerConnector:
    """Test the Go2 Game Controller connector."""

    def test_initialization_with_mocks(
        self, mock_hid_no_controllers, mock_patches, custom_config
    ):
        """Test connector initialization with custom configuration."""
        from src.actions.move_game_controller.connector.go2_game_controller import (
            Go2GameControllerConnector,
        )

        connector = Go2GameControllerConnector(config=custom_config)

        assert connector.config == custom_config
        assert connector.move_speed == 1.0
        assert connector.turn_speed == 0.5
        assert connector.gamepad is None
        assert connector.sony_dualsense is False
        assert connector.xbox is False
        assert connector.sony_edge is False

    def test_init_controller_no_controllers(
        self, mock_hid_no_controllers, mock_patches, mock_config
    ):
        """Test _init_controller when no controllers are found."""
        from src.actions.move_game_controller.connector.go2_game_controller import (
            Go2GameControllerConnector,
        )

        connector = Go2GameControllerConnector(config=mock_config)
        connector._init_controller()

        assert connector.gamepad is None
        assert connector.sony_dualsense is False
        assert connector.xbox is False
        assert connector.sony_edge is False

    @patch("time.sleep", return_value=None)
    def test_tick_method(
        self, mock_sleep, mock_hid_no_controllers, mock_patches, mock_config
    ):
        """Test tick method."""
        from src.actions.move_game_controller.connector.go2_game_controller import (
            Go2GameControllerConnector,
        )

        connector = Go2GameControllerConnector(config=mock_config)
        connector.tick()

    @pytest.mark.asyncio
    async def test_connect_method(
        self, mock_hid_no_controllers, mock_patches, mock_config
    ):
        """Test connect method."""
        from src.actions.move_game_controller.connector.go2_game_controller import (
            Go2GameControllerConnector,
            IDLEInput,
        )

        connector = Go2GameControllerConnector(config=mock_config)
        mock_idle = Mock(spec=IDLEInput)
        await connector.connect(mock_idle)
