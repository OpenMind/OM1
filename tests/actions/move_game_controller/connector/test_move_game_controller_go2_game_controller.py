import importlib
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

from actions.move_game_controller.interface import IDLEInput


@pytest.fixture
def mock_external_modules(monkeypatch):
    """Mock external modules before importing the connector module."""
    mock_zenoh = MagicMock()
    mock_zenoh_msgs = MagicMock()
    mock_hid = MagicMock()
    mock_unitree = MagicMock()
    mock_sport_client = MagicMock()
    mock_unitree.unitree_sdk2py.go2.sport.sport_client.SportClient = mock_sport_client

    monkeypatch.setitem(sys.modules, "zenoh", mock_zenoh)
    monkeypatch.setitem(sys.modules, "zenoh_msgs", mock_zenoh_msgs)
    monkeypatch.setitem(sys.modules, "hid", mock_hid)
    monkeypatch.setitem(sys.modules, "unitree", mock_unitree)
    monkeypatch.setitem(
        sys.modules, "unitree.unitree_sdk2py", mock_unitree.unitree_sdk2py
    )
    monkeypatch.setitem(
        sys.modules,
        "unitree.unitree_sdk2py.go2",
        mock_unitree.unitree_sdk2py.go2,
    )
    monkeypatch.setitem(
        sys.modules,
        "unitree.unitree_sdk2py.go2.sport",
        mock_unitree.unitree_sdk2py.go2.sport,
    )
    monkeypatch.setitem(
        sys.modules,
        "unitree.unitree_sdk2py.go2.sport.sport_client",
        mock_unitree.unitree_sdk2py.go2.sport.sport_client,
    )

    return {
        "zenoh": mock_zenoh,
        "zenoh_msgs": mock_zenoh_msgs,
        "hid": mock_hid,
        "sport_client": mock_sport_client,
    }


@pytest.fixture
def go2_module(mock_external_modules):
    """Import the connector module after mocking external deps."""
    sys.modules.pop(
        "actions.move_game_controller.connector.go2_game_controller",
        None,
    )
    return importlib.import_module(
        "actions.move_game_controller.connector.go2_game_controller"
    )


@pytest.fixture
def default_config(go2_module):
    """Create a default config for testing."""
    return go2_module.Go2GameControllerConfig()


@pytest.fixture
def custom_config(go2_module):
    """Create a custom config for testing."""
    return go2_module.Go2GameControllerConfig(
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


class TestGo2GameControllerConfig:
    """Test the Go2GameController configuration class."""

    def test_default_config(self, go2_module):
        """Test default configuration values."""
        config = go2_module.Go2GameControllerConfig()
        assert config.speed_x == 0.9
        assert config.speed_yaw == 0.6
        assert config.yaw_correction == 0.0
        assert config.lateral_correction == 0.0
        assert config.unitree_ethernet is None

    def test_custom_config(self, go2_module):
        """Test custom configuration values."""
        config = go2_module.Go2GameControllerConfig(
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

    def test_init(
        self,
        go2_module,
        default_config,
    ):
        """Test initialization of Go2GameControllerConnector."""
        with (
            patch(
                "actions.move_game_controller.connector.go2_game_controller.hid"
            ) as mock_hid_module,
            patch(
                "actions.move_game_controller.connector.go2_game_controller.SportClient"
            ) as mock_sport_client_class,
            patch(
                "actions.move_game_controller.connector.go2_game_controller.open_zenoh_session"
            ),
            patch(
                "actions.move_game_controller.connector.go2_game_controller.UnitreeGo2OdomProvider"
            ),
            patch(
                "actions.move_game_controller.connector.go2_game_controller.UnitreeGo2StateProvider"
            ),
        ):
            mock_client_instance = Mock()
            mock_sport_client_class.return_value = mock_client_instance
            mock_hid_module.enumerate.return_value = []

            connector = go2_module.Go2GameControllerConnector(default_config)

            assert connector.move_speed == 0.9
            assert connector.turn_speed == 0.6
            mock_client_instance.SetTimeout.assert_called_once_with(10.0)
            mock_client_instance.Init.assert_called_once()

    def test_init_with_custom_config(
        self,
        go2_module,
        custom_config,
    ):
        """Test initialization with custom configuration."""
        with (
            patch(
                "actions.move_game_controller.connector.go2_game_controller.hid"
            ) as mock_hid_module,
            patch(
                "actions.move_game_controller.connector.go2_game_controller.SportClient"
            ) as mock_sport_client_class,
            patch(
                "actions.move_game_controller.connector.go2_game_controller.open_zenoh_session"
            ),
            patch(
                "actions.move_game_controller.connector.go2_game_controller.UnitreeGo2OdomProvider"
            ),
            patch(
                "actions.move_game_controller.connector.go2_game_controller.UnitreeGo2StateProvider"
            ),
        ):
            mock_client_instance = Mock()
            mock_sport_client_class.return_value = mock_client_instance
            mock_hid_module.enumerate.return_value = []

            connector = go2_module.Go2GameControllerConnector(custom_config)

            assert connector.move_speed == 1.2
            assert connector.turn_speed == 0.8
            assert connector.yaw_correction == 0.1
            assert connector.lateral_correction == 0.05

    def test_init_sport_client_error(
        self,
        go2_module,
        default_config,
    ):
        """Test initialization when SportClient fails."""
        with (
            patch(
                "actions.move_game_controller.connector.go2_game_controller.hid"
            ) as mock_hid_module,
            patch(
                "actions.move_game_controller.connector.go2_game_controller.SportClient"
            ) as mock_sport_client_class,
            patch(
                "actions.move_game_controller.connector.go2_game_controller.open_zenoh_session"
            ),
            patch(
                "actions.move_game_controller.connector.go2_game_controller.UnitreeGo2OdomProvider"
            ),
            patch(
                "actions.move_game_controller.connector.go2_game_controller.UnitreeGo2StateProvider"
            ),
        ):
            mock_sport_client_class.side_effect = Exception("Connection error")
            mock_hid_module.enumerate.return_value = []

            connector = go2_module.Go2GameControllerConnector(default_config)

            assert connector.sport_client is None

    @pytest.mark.asyncio
    async def test_connect(
        self,
        go2_module,
        default_config,
        idle_input,
    ):
        """Test connect method (passes through)."""
        with (
            patch(
                "actions.move_game_controller.connector.go2_game_controller.hid"
            ) as mock_hid_module,
            patch(
                "actions.move_game_controller.connector.go2_game_controller.SportClient"
            ) as mock_sport_client_class,
            patch(
                "actions.move_game_controller.connector.go2_game_controller.open_zenoh_session"
            ),
            patch(
                "actions.move_game_controller.connector.go2_game_controller.UnitreeGo2OdomProvider"
            ),
            patch(
                "actions.move_game_controller.connector.go2_game_controller.UnitreeGo2StateProvider"
            ),
        ):
            mock_client_instance = Mock()
            mock_sport_client_class.return_value = mock_client_instance
            mock_hid_module.enumerate.return_value = []

            connector = go2_module.Go2GameControllerConnector(default_config)
            # connect is a pass-through, should not raise
            await connector.connect(idle_input)
