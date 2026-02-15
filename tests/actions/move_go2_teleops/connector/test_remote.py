"""Tests for the Move Go2 Teleops Remote connector."""

import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

# Mock modules at module load time BEFORE any other imports
mock_zenoh = MagicMock()
mock_zenoh_msgs = MagicMock()
mock_om1_utils = MagicMock()
mock_unitree = MagicMock()
mock_sport_client = MagicMock()
mock_unitree.unitree_sdk2py.go2.sport.sport_client.SportClient = mock_sport_client

sys.modules["zenoh"] = mock_zenoh
sys.modules["zenoh_msgs"] = mock_zenoh_msgs
sys.modules["om1_utils"] = mock_om1_utils
sys.modules["om1_utils.ws"] = mock_om1_utils.ws
sys.modules["unitree"] = mock_unitree
sys.modules["unitree.unitree_sdk2py"] = mock_unitree.unitree_sdk2py
sys.modules["unitree.unitree_sdk2py.go2"] = mock_unitree.unitree_sdk2py.go2
sys.modules["unitree.unitree_sdk2py.go2.sport"] = mock_unitree.unitree_sdk2py.go2.sport
sys.modules["unitree.unitree_sdk2py.go2.sport.sport_client"] = (
    mock_unitree.unitree_sdk2py.go2.sport.sport_client
)

from actions.move_go2_teleops.connector.remote import (  # noqa: E402
    MoveGo2RemoteConfig,
    MoveGo2RemoteConnector,
)
from actions.move_go2_teleops.interface import MoveInput, MovementAction  # noqa: E402


@pytest.fixture
def default_config():
    """Create a default config for testing."""
    return MoveGo2RemoteConfig()


@pytest.fixture
def config_with_api_key():
    """Create a config with API key for testing."""
    return MoveGo2RemoteConfig(api_key="test_api_key_123")


@pytest.fixture
def move_input_stand():
    """Create a MoveInput instance with stand up action."""
    return MoveInput(action=MovementAction.STAND_UP)


@pytest.fixture
def move_input_sit():
    """Create a MoveInput instance with sit action."""
    return MoveInput(action=MovementAction.SIT)


@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset all mock objects between tests."""
    mock_zenoh.reset_mock()
    mock_zenoh_msgs.reset_mock()
    mock_om1_utils.reset_mock()
    mock_sport_client.reset_mock()
    yield


class TestMoveGo2RemoteConfig:
    """Test the MoveGo2Remote configuration class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MoveGo2RemoteConfig()
        assert config.api_key == ""

    def test_custom_config(self):
        """Test custom configuration values."""
        config = MoveGo2RemoteConfig(api_key="my_api_key")
        assert config.api_key == "my_api_key"


class TestMoveGo2RemoteConnector:
    """Test the MoveGo2Remote connector."""

    @patch("actions.move_go2_teleops.connector.remote.UnitreeGo2StateProvider")
    @patch("actions.move_go2_teleops.connector.remote.SportClient")
    @patch("actions.move_go2_teleops.connector.remote.ws")
    def test_init(
        self, mock_ws, mock_sport_client_class, mock_state_provider, default_config
    ):
        """Test initialization of MoveGo2RemoteConnector."""
        mock_client_instance = Mock()
        mock_sport_client_class.return_value = mock_client_instance

        mock_ws_client = Mock()
        mock_ws.Client.return_value = mock_ws_client

        connector = MoveGo2RemoteConnector(default_config)

        assert connector.sport_client is not None
        mock_client_instance.SetTimeout.assert_called_once_with(10.0)
        mock_client_instance.Init.assert_called_once()

    @patch("actions.move_go2_teleops.connector.remote.UnitreeGo2StateProvider")
    @patch("actions.move_go2_teleops.connector.remote.SportClient")
    @patch("actions.move_go2_teleops.connector.remote.ws")
    def test_init_with_api_key(
        self, mock_ws, mock_sport_client_class, mock_state_provider, config_with_api_key
    ):
        """Test initialization with API key."""
        mock_client_instance = Mock()
        mock_sport_client_class.return_value = mock_client_instance

        mock_ws_client = Mock()
        mock_ws.Client.return_value = mock_ws_client

        MoveGo2RemoteConnector(config_with_api_key)

        mock_ws.Client.assert_called_once()
        call_args = mock_ws.Client.call_args
        assert "test_api_key_123" in call_args[1]["url"]

    @patch("actions.move_go2_teleops.connector.remote.UnitreeGo2StateProvider")
    @patch("actions.move_go2_teleops.connector.remote.SportClient")
    @patch("actions.move_go2_teleops.connector.remote.ws")
    def test_init_sport_client_error(
        self, mock_ws, mock_sport_client_class, mock_state_provider, default_config
    ):
        """Test initialization when SportClient fails."""
        mock_sport_client_class.side_effect = Exception("Connection error")

        mock_ws_client = Mock()
        mock_ws.Client.return_value = mock_ws_client

        connector = MoveGo2RemoteConnector(default_config)

        assert connector.sport_client is None

    @patch("actions.move_go2_teleops.connector.remote.UnitreeGo2StateProvider")
    @patch("actions.move_go2_teleops.connector.remote.SportClient")
    @patch("actions.move_go2_teleops.connector.remote.ws")
    @pytest.mark.asyncio
    async def test_connect(
        self,
        mock_ws,
        mock_sport_client_class,
        mock_state_provider,
        default_config,
        move_input_stand,
    ):
        """Test connect method (passes through)."""
        mock_client_instance = Mock()
        mock_sport_client_class.return_value = mock_client_instance

        mock_ws_client = Mock()
        mock_ws.Client.return_value = mock_ws_client

        connector = MoveGo2RemoteConnector(default_config)
        # connect is a pass-through, should not raise
        await connector.connect(move_input_stand)
