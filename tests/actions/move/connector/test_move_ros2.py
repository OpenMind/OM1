from unittest.mock import patch

import pytest

from actions.base import ActionConfig
from actions.move.connector.ros2 import MoveUnitreeSDKConnector
from actions.move.interface import MoveInput, MovementAction


@pytest.fixture
def default_config():
    return ActionConfig()


@pytest.fixture
def connector(default_config):
    return MoveUnitreeSDKConnector(default_config)


class TestMoveUnitreeSDKConnectorInit:
    """Test MoveUnitreeSDKConnector initialization."""

    def test_init(self, default_config):
        connector = MoveUnitreeSDKConnector(default_config)
        assert connector.config == default_config


class TestMoveUnitreeSDKConnectorConnect:
    """Test connect method for each movement action."""

    @pytest.mark.asyncio
    async def test_connect_stand_still(self, connector):
        """Test stand still action maps correctly."""
        move_input = MoveInput(action=MovementAction.STAND_STILL)
        with patch("actions.move.connector.ros2.logging") as mock_logging:
            await connector.connect(move_input)
            mock_logging.info.assert_called_with(
                "SendThisToROS2: {'move': 'stand still'}"
            )

    @pytest.mark.asyncio
    async def test_connect_sit(self, connector):
        """Test sit action maps correctly."""
        move_input = MoveInput(action=MovementAction.SIT)
        with patch("actions.move.connector.ros2.logging") as mock_logging:
            await connector.connect(move_input)
            mock_logging.info.assert_called_with("SendThisToROS2: {'move': 'sit'}")

    @pytest.mark.asyncio
    async def test_connect_dance(self, connector):
        """Test dance action maps correctly."""
        move_input = MoveInput(action=MovementAction.DANCE)
        with patch("actions.move.connector.ros2.logging") as mock_logging:
            await connector.connect(move_input)
            mock_logging.info.assert_called_with("SendThisToROS2: {'move': 'dance'}")

    @pytest.mark.asyncio
    async def test_connect_walk(self, connector):
        """Test walk action maps correctly."""
        move_input = MoveInput(action=MovementAction.WALK)
        with patch("actions.move.connector.ros2.logging") as mock_logging:
            await connector.connect(move_input)
            mock_logging.info.assert_called_with("SendThisToROS2: {'move': 'walk'}")

    @pytest.mark.asyncio
    async def test_connect_run(self, connector):
        """Test run action maps correctly."""
        move_input = MoveInput(action=MovementAction.RUN)
        with patch("actions.move.connector.ros2.logging") as mock_logging:
            await connector.connect(move_input)
            mock_logging.info.assert_called_with("SendThisToROS2: {'move': 'run'}")

    @pytest.mark.asyncio
    async def test_connect_jump(self, connector):
        """Test jump action maps correctly."""
        move_input = MoveInput(action=MovementAction.JUMP)
        with patch("actions.move.connector.ros2.logging") as mock_logging:
            await connector.connect(move_input)
            mock_logging.info.assert_called_with("SendThisToROS2: {'move': 'jump'}")

    @pytest.mark.asyncio
    async def test_connect_wag_tail(self, connector):
        """Test wag tail action maps correctly."""
        move_input = MoveInput(action=MovementAction.WAG_TAIL)
        with patch("actions.move.connector.ros2.logging") as mock_logging:
            await connector.connect(move_input)
            mock_logging.info.assert_called_with("SendThisToROS2: {'move': 'wag tail'}")

    @pytest.mark.asyncio
    async def test_connect_shake_paw(self, connector):
        """Test shake paw action maps correctly."""
        move_input = MoveInput(action=MovementAction.SHAKE_PAW)
        with patch("actions.move.connector.ros2.logging") as mock_logging:
            await connector.connect(move_input)
            mock_logging.info.assert_called_with(
                "SendThisToROS2: {'move': 'shake paw'}"
            )

    @pytest.mark.asyncio
    async def test_connect_walk_back(self, connector):
        """Test walk back action maps correctly."""
        move_input = MoveInput(action=MovementAction.WALK_BACK)
        with patch("actions.move.connector.ros2.logging") as mock_logging:
            await connector.connect(move_input)
            mock_logging.info.assert_called_with(
                "SendThisToROS2: {'move': 'walk back'}"
            )

    @pytest.mark.asyncio
    async def test_connect_unknown_action(self, connector):
        """Test unknown action logs info message."""
        move_input = MoveInput(action="unknown_action")  # type: ignore[arg-type]
        with patch("actions.move.connector.ros2.logging") as mock_logging:
            await connector.connect(move_input)
            mock_logging.info.assert_any_call("Other move type: unknown_action")
