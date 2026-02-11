import json
from unittest.mock import Mock, patch

import pytest

from actions.move_go2_autonomy.interface import MoveInput, MovementAction
from actions.move_tron.connector.tron_sdk import (
    MoveTronSDKConfig,
    MoveTronSDKConnector,
)


class TestMoveTronSDKConfig:
    """Test MoveTronSDKConfig configuration."""

    def test_default_config(self):
        """Test default base_url value."""
        config = MoveTronSDKConfig(accid="robot123")
        assert config.base_url == "ws://10.192.1.2:5000"
        assert config.accid == "robot123"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = MoveTronSDKConfig(
            base_url="ws://custom:8080",
            accid="custom_robot",
        )
        assert config.base_url == "ws://custom:8080"
        assert config.accid == "custom_robot"


class TestMoveTronSDKConnectorInit:
    """Test MoveTronSDKConnector initialization."""

    def test_init_creates_ws_client(self):
        """Test that init creates and starts a WebSocket client."""
        with patch("actions.move_tron.connector.tron_sdk.ws.Client") as mock_ws_client:
            mock_client = Mock()
            mock_ws_client.return_value = mock_client

            config = MoveTronSDKConfig(accid="robot123")
            connector = MoveTronSDKConnector(config)

            mock_ws_client.assert_called_once_with("ws://10.192.1.2:5000")
            mock_client.start.assert_called_once()
            assert connector.client == mock_client


class TestMoveTronSDKConnectorConnect:
    """Test connect method for each movement action."""

    @pytest.fixture
    def connector(self):
        """Create connector with mocked WebSocket client."""
        with patch("actions.move_tron.connector.tron_sdk.ws.Client") as mock_ws_client:
            mock_client = Mock()
            mock_client.connected = True
            mock_ws_client.return_value = mock_client

            config = MoveTronSDKConfig(accid="robot123")
            connector = MoveTronSDKConnector(config)
            return connector

    @pytest.fixture
    def disconnected_connector(self):
        """Create connector with disconnected WebSocket client."""
        with patch("actions.move_tron.connector.tron_sdk.ws.Client") as mock_ws_client:
            mock_client = Mock()
            mock_client.connected = False
            mock_ws_client.return_value = mock_client

            config = MoveTronSDKConfig(accid="robot123")
            connector = MoveTronSDKConnector(config)
            return connector

    @pytest.mark.asyncio
    async def test_connect_move_forwards(self, connector):
        """Test move forwards sends correct x velocity."""
        move_input = MoveInput(action=MovementAction.MOVE_FORWARDS)
        await connector.connect(move_input)

        connector.client.send_message.assert_called_once()
        sent_data = json.loads(connector.client.send_message.call_args[0][0])
        assert sent_data["accid"] == "robot123"
        assert sent_data["title"] == "request_twist"
        assert sent_data["data"]["x"] == 0.5
        assert sent_data["data"]["y"] == 0.0
        assert sent_data["data"]["z"] == 0.0

    @pytest.mark.asyncio
    async def test_connect_move_back(self, connector):
        """Test move back sends negative x velocity."""
        move_input = MoveInput(action=MovementAction.MOVE_BACK)
        await connector.connect(move_input)

        sent_data = json.loads(connector.client.send_message.call_args[0][0])
        assert sent_data["data"]["x"] == -0.5

    @pytest.mark.asyncio
    async def test_connect_turn_left(self, connector):
        """Test turn left sends positive z velocity."""
        move_input = MoveInput(action=MovementAction.TURN_LEFT)
        await connector.connect(move_input)

        sent_data = json.loads(connector.client.send_message.call_args[0][0])
        assert sent_data["data"]["z"] == 0.5
        assert sent_data["data"]["x"] == 0.0

    @pytest.mark.asyncio
    async def test_connect_turn_right(self, connector):
        """Test turn right sends negative z velocity."""
        move_input = MoveInput(action=MovementAction.TURN_RIGHT)
        await connector.connect(move_input)

        sent_data = json.loads(connector.client.send_message.call_args[0][0])
        assert sent_data["data"]["z"] == -0.5
        assert sent_data["data"]["x"] == 0.0

    @pytest.mark.asyncio
    async def test_connect_stand_still(self, connector):
        """Test stand still sends zero velocities."""
        move_input = MoveInput(action=MovementAction.STAND_STILL)
        await connector.connect(move_input)

        sent_data = json.loads(connector.client.send_message.call_args[0][0])
        assert sent_data["data"]["x"] == 0.0
        assert sent_data["data"]["y"] == 0.0
        assert sent_data["data"]["z"] == 0.0

    @pytest.mark.asyncio
    async def test_connect_disconnected_logs_error(self, disconnected_connector):
        """Test that disconnected client logs error instead of sending."""
        move_input = MoveInput(action=MovementAction.MOVE_FORWARDS)
        with patch("actions.move_tron.connector.tron_sdk.logging") as mock_logging:
            await disconnected_connector.connect(move_input)
            mock_logging.error.assert_called_with(
                "Tron webSocket client is not connected."
            )
            disconnected_connector.client.send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_connect_message_has_guid(self, connector):
        """Test that each message has a unique guid."""
        move_input = MoveInput(action=MovementAction.MOVE_FORWARDS)
        await connector.connect(move_input)

        sent_data = json.loads(connector.client.send_message.call_args[0][0])
        assert "guid" in sent_data
        assert len(sent_data["guid"]) > 0
