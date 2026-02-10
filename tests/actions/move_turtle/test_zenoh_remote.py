import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

mock_zenoh_msgs = MagicMock()
sys.modules["zenoh_msgs"] = mock_zenoh_msgs

mock_om1_utils = MagicMock()
sys.modules["om1_utils"] = mock_om1_utils
sys.modules["om1_utils.ws"] = mock_om1_utils.ws

from actions.move_turtle.connector.zenoh_remote import (  # noqa: E402
    MoveZenohRemoteConfig,
    MoveZenohRemoteConnector,
)
from actions.move_turtle.interface import MoveInput, MovementAction  # noqa: E402


@pytest.fixture
def mock_dependencies():
    """Mock all external dependencies."""
    with (
        patch(
            "actions.move_turtle.connector.zenoh_remote.open_zenoh_session"
        ) as mock_open_session,
        patch("actions.move_turtle.connector.zenoh_remote.ws.Client") as mock_ws,
    ):
        mock_session = Mock()
        mock_open_session.return_value = mock_session

        mock_ws_instance = Mock()
        mock_ws.return_value = mock_ws_instance

        yield {
            "session": mock_session,
            "ws": mock_ws_instance,
        }


@pytest.fixture
def connector(mock_dependencies):
    """Create MoveZenohRemoteConnector."""
    config = MoveZenohRemoteConfig(api_key="test_key", URID="test_robot")
    return MoveZenohRemoteConnector(config)


class TestMoveZenohRemoteConfig:
    """Test MoveZenohRemoteConfig configuration."""

    def test_default_config(self):
        config = MoveZenohRemoteConfig()
        assert config.api_key is None
        assert config.URID is None

    def test_custom_config(self):
        config = MoveZenohRemoteConfig(api_key="my_key", URID="my_robot")
        assert config.api_key == "my_key"
        assert config.URID == "my_robot"


class TestMoveZenohRemoteConnectorInit:
    """Test initialization."""

    def test_init(self, connector, mock_dependencies):
        assert connector.cmd_vel == "test_robot/c3/cmd_vel"
        assert connector.session == mock_dependencies["session"]
        mock_dependencies["ws"].start.assert_called_once()
        mock_dependencies["ws"].register_message_callback.assert_called_once()

    def test_init_zenoh_error(self):
        """Test initialization when Zenoh fails."""
        with (
            patch(
                "actions.move_turtle.connector.zenoh_remote.open_zenoh_session"
            ) as mock_session,
            patch("actions.move_turtle.connector.zenoh_remote.ws.Client"),
            patch("actions.move_turtle.connector.zenoh_remote.logging") as mock_logging,
        ):
            mock_session.side_effect = Exception("Failed")
            config = MoveZenohRemoteConfig(URID="test")
            connector = MoveZenohRemoteConnector(config)
            assert connector.session is None
            mock_logging.error.assert_called()


class TestMoveZenohRemoteConnectorOnMessage:
    """Test _on_message callback."""

    def test_on_message_no_session(self, connector, mock_dependencies):
        """Test on_message when session is None."""
        connector.session = None
        with patch(
            "actions.move_turtle.connector.zenoh_remote.logging"
        ) as mock_logging:
            connector._on_message('{"vx": 0.5, "vyaw": 0.0}')
            mock_logging.info.assert_any_call("No open Zenoh session, returning")

    def test_on_message_valid(self, connector, mock_dependencies):
        """Test on_message with valid command."""
        with patch(
            "actions.move_turtle.connector.zenoh_remote.CommandStatus"
        ) as mock_cmd:
            mock_status = Mock()
            mock_status.vx = 0.5
            mock_status.vyaw = 0.0
            mock_status.to_dict.return_value = {"vx": 0.5, "vyaw": 0.0}
            mock_status.timestamp = "0.0"
            mock_cmd.from_dict.return_value = mock_status

            connector._on_message('{"vx": 0.5, "vyaw": 0.0}')
            mock_dependencies["session"].put.assert_called_once()

    def test_on_message_error(self, connector, mock_dependencies):
        """Test on_message with invalid message."""
        with patch(
            "actions.move_turtle.connector.zenoh_remote.logging"
        ) as mock_logging:
            connector._on_message("invalid json{{{")
            mock_logging.error.assert_called()


class TestMoveZenohRemoteConnectorConnect:
    """Test connect method."""

    @pytest.mark.asyncio
    async def test_connect_is_noop(self, connector, mock_dependencies):
        """Test connect is a no-op (pass)."""
        move_input = MoveInput(action=MovementAction.MOVE_FORWARDS)
        await connector.connect(move_input)
