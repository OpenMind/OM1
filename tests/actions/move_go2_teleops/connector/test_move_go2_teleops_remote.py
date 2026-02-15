import importlib
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

from actions.move_go2_teleops.interface import MoveInput, MovementAction


@pytest.fixture
def mock_external_modules(monkeypatch):
    """Mock external modules before importing the connector module."""
    mock_zenoh = MagicMock()
    mock_zenoh_msgs = MagicMock()
    mock_om1_utils = MagicMock()
    mock_unitree = MagicMock()
    mock_sport_client = MagicMock()
    mock_unitree.unitree_sdk2py.go2.sport.sport_client.SportClient = mock_sport_client

    monkeypatch.setitem(sys.modules, "zenoh", mock_zenoh)
    monkeypatch.setitem(sys.modules, "zenoh_msgs", mock_zenoh_msgs)
    monkeypatch.setitem(sys.modules, "om1_utils", mock_om1_utils)
    monkeypatch.setitem(sys.modules, "om1_utils.ws", mock_om1_utils.ws)
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
        "om1_utils": mock_om1_utils,
        "sport_client": mock_sport_client,
    }


@pytest.fixture
def remote_module(mock_external_modules):
    """Import the connector module after mocking external deps."""
    sys.modules.pop(
        "actions.move_go2_teleops.connector.remote",
        None,
    )
    return importlib.import_module("actions.move_go2_teleops.connector.remote")


@pytest.fixture
def default_config(remote_module):
    """Create a default config for testing."""
    return remote_module.MoveGo2RemoteConfig()


@pytest.fixture
def config_with_api_key(remote_module):
    """Create a config with API key for testing."""
    return remote_module.MoveGo2RemoteConfig(api_key="test_api_key_123")


@pytest.fixture
def move_input_stand():
    """Create a MoveInput instance with stand up action."""
    return MoveInput(action=MovementAction.STAND_UP)


@pytest.fixture
def move_input_sit():
    """Create a MoveInput instance with sit action."""
    return MoveInput(action=MovementAction.SIT)


class TestMoveGo2RemoteConfig:
    """Test the MoveGo2Remote configuration class."""

    def test_default_config(self, remote_module):
        """Test default configuration values."""
        config = remote_module.MoveGo2RemoteConfig()
        assert config.api_key == ""

    def test_custom_config(self, remote_module):
        """Test custom configuration values."""
        config = remote_module.MoveGo2RemoteConfig(api_key="my_api_key")
        assert config.api_key == "my_api_key"


class TestMoveGo2RemoteConnector:
    """Test the MoveGo2Remote connector."""

    def test_init(self, remote_module, default_config):
        """Test initialization of MoveGo2RemoteConnector."""
        with (
            patch("actions.move_go2_teleops.connector.remote.ws") as mock_ws,
            patch(
                "actions.move_go2_teleops.connector.remote.SportClient"
            ) as mock_sport_client_class,
            patch("actions.move_go2_teleops.connector.remote.UnitreeGo2StateProvider"),
        ):
            mock_client_instance = Mock()
            mock_sport_client_class.return_value = mock_client_instance

            mock_ws_client = Mock()
            mock_ws.Client.return_value = mock_ws_client

            connector = remote_module.MoveGo2RemoteConnector(default_config)

            assert connector.sport_client is not None
            mock_client_instance.SetTimeout.assert_called_once_with(10.0)
            mock_client_instance.Init.assert_called_once()

    def test_init_with_api_key(self, remote_module, config_with_api_key):
        """Test initialization with API key."""
        with (
            patch("actions.move_go2_teleops.connector.remote.ws") as mock_ws,
            patch(
                "actions.move_go2_teleops.connector.remote.SportClient"
            ) as mock_sport_client_class,
            patch("actions.move_go2_teleops.connector.remote.UnitreeGo2StateProvider"),
        ):
            mock_client_instance = Mock()
            mock_sport_client_class.return_value = mock_client_instance

            mock_ws_client = Mock()
            mock_ws.Client.return_value = mock_ws_client

            remote_module.MoveGo2RemoteConnector(config_with_api_key)

            mock_ws.Client.assert_called_once()
            call_args = mock_ws.Client.call_args
            assert "test_api_key_123" in call_args[1]["url"]

    def test_init_sport_client_error(self, remote_module, default_config):
        """Test initialization when SportClient fails."""
        with (
            patch("actions.move_go2_teleops.connector.remote.ws") as mock_ws,
            patch(
                "actions.move_go2_teleops.connector.remote.SportClient"
            ) as mock_sport_client_class,
            patch("actions.move_go2_teleops.connector.remote.UnitreeGo2StateProvider"),
        ):
            mock_sport_client_class.side_effect = Exception("Connection error")

            mock_ws_client = Mock()
            mock_ws.Client.return_value = mock_ws_client

            connector = remote_module.MoveGo2RemoteConnector(default_config)

            assert connector.sport_client is None

    @pytest.mark.asyncio
    async def test_connect(
        self,
        remote_module,
        default_config,
        move_input_stand,
    ):
        """Test connect method (passes through)."""
        with (
            patch("actions.move_go2_teleops.connector.remote.ws") as mock_ws,
            patch(
                "actions.move_go2_teleops.connector.remote.SportClient"
            ) as mock_sport_client_class,
            patch("actions.move_go2_teleops.connector.remote.UnitreeGo2StateProvider"),
        ):
            mock_client_instance = Mock()
            mock_sport_client_class.return_value = mock_client_instance

            mock_ws_client = Mock()
            mock_ws.Client.return_value = mock_ws_client

            connector = remote_module.MoveGo2RemoteConnector(default_config)
            await connector.connect(move_input_stand)
