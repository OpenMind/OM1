import concurrent.futures
from unittest.mock import AsyncMock, Mock, patch

import pytest

from actions.move_ub.connector.yanshee_motion import (
    Motion,
    MoveYansheeConfig,
    MoveYansheeConnector,
)
from actions.move_ub.interface import MoveInput, MovementAction


class TestMotion:
    """Test Motion dataclass."""

    def test_reset_defaults(self):
        m = Motion("reset")
        assert m.direction == ""
        assert m.speed == "normal"
        assert m.repeat == 1
        assert m.version == "v1"

    def test_wave_defaults(self):
        m = Motion("wave")
        assert m.direction == "both"

    def test_walk_defaults(self):
        m = Motion("walk")
        assert m.direction == "forward"

    def test_custom_override(self):
        m = Motion("walk", direction="backward", repeat=3)
        assert m.direction == "backward"
        assert m.repeat == 3

    def test_unknown_motion_raises(self):
        with pytest.raises(ValueError, match="Unknown motion name"):
            Motion("fly")

    def test_all_known_motions(self):
        """Test all known motion names don't raise."""
        known = [
            "reset",
            "wave",
            "bow",
            "crouch",
            "come on",
            "walk",
            "head",
            "turn around",
            "WakaWaka",
            "Hug",
            "RaiseRightHand",
            "PushUp",
        ]
        for name in known:
            m = Motion(name)
            assert m.name == name


class TestMoveYansheeConfig:
    """Test MoveYansheeConfig configuration."""

    def test_default_config(self):
        config = MoveYansheeConfig()
        assert config.robot_ip == "127.0.0.1"

    def test_custom_ip(self):
        config = MoveYansheeConfig(robot_ip="192.168.1.100")
        assert config.robot_ip == "192.168.1.100"


class TestMoveYansheeConnectorInit:
    """Test MoveYansheeConnector initialization."""

    def test_init(self):
        with patch("actions.move_ub.connector.yanshee_motion.YanAPI") as mock_api:
            config = MoveYansheeConfig()
            connector = MoveYansheeConnector(config)

            mock_api.yan_api_init.assert_called_once_with("127.0.0.1")
            assert connector.move_speed == 0.7
            assert connector.turn_speed == 0.6
            assert connector.timeout == 8.0

    def test_init_api_error(self):
        with (
            patch("actions.move_ub.connector.yanshee_motion.YanAPI") as mock_api,
            patch("actions.move_ub.connector.yanshee_motion.logging") as mock_logging,
        ):
            mock_api.yan_api_init.side_effect = Exception("Robot not found")
            config = MoveYansheeConfig()
            MoveYansheeConnector(config)
            mock_logging.error.assert_called()


class TestMoveYansheeConnectorSendCommand:
    """Test _send_command method."""

    @pytest.fixture
    def connector(self):
        with patch("actions.move_ub.connector.yanshee_motion.YanAPI") as mock_api:
            self.mock_api = mock_api
            config = MoveYansheeConfig()
            connector = MoveYansheeConnector(config)
            return connector

    def test_send_command_success_reset(self, connector):
        with patch("actions.move_ub.connector.yanshee_motion.YanAPI") as mock_api:
            mock_api.sync_play_motion.return_value = "ok"
            result = connector._send_command(Motion("reset"))
            assert result == "ok"
            mock_api.sync_play_motion.assert_called_once()

    def test_send_command_non_reset_sends_reset_after(self, connector):
        with patch("actions.move_ub.connector.yanshee_motion.YanAPI") as mock_api:
            mock_api.sync_play_motion.return_value = "ok"
            connector._send_command(Motion("wave"))
            assert mock_api.sync_play_motion.call_count == 2

    def test_send_command_value_error(self, connector):
        with (
            patch("actions.move_ub.connector.yanshee_motion.YanAPI") as mock_api,
            patch("actions.move_ub.connector.yanshee_motion.logging") as mock_logging,
        ):
            mock_api.sync_play_motion.side_effect = ValueError("bad param")
            result = connector._send_command(Motion("wave"))
            assert result is False
            mock_logging.error.assert_called()

    def test_send_command_timeout(self, connector):
        """Test _send_command with timeout."""
        with patch(
            "actions.move_ub.connector.yanshee_motion.concurrent.futures.ThreadPoolExecutor"
        ) as mock_executor:
            mock_future = Mock()
            mock_future.result.side_effect = concurrent.futures.TimeoutError()
            mock_executor.return_value.__enter__ = Mock(
                return_value=Mock(submit=Mock(return_value=mock_future))
            )
            mock_executor.return_value.__exit__ = Mock(return_value=False)

            result = connector._send_command(Motion("wave"))
            assert result is False


class TestMoveYansheeConnectorConnect:
    """Test connect method."""

    @pytest.fixture
    def connector(self):
        with patch("actions.move_ub.connector.yanshee_motion.YanAPI"):
            config = MoveYansheeConfig()
            connector = MoveYansheeConnector(config)
            connector._execute_sport_command = AsyncMock()
            return connector

    @pytest.mark.asyncio
    async def test_connect_wave(self, connector):
        move_input = MoveInput(action=MovementAction.WAVE)
        with patch("actions.move_ub.connector.yanshee_motion.logging") as mock_logging:
            await connector.connect(move_input)
            mock_logging.info.assert_any_call("UB command: wave")

    @pytest.mark.asyncio
    async def test_connect_walk_forward(self, connector):
        move_input = MoveInput(action=MovementAction.WALK_FORWARD)
        with patch("actions.move_ub.connector.yanshee_motion.logging") as mock_logging:
            await connector.connect(move_input)
            mock_logging.info.assert_any_call("UB command: walk forward")

    @pytest.mark.asyncio
    async def test_connect_walk_backward(self, connector):
        move_input = MoveInput(action=MovementAction.WALK_BACKWARD)
        with patch("actions.move_ub.connector.yanshee_motion.logging") as mock_logging:
            await connector.connect(move_input)
            mock_logging.info.assert_any_call("UB command: walk backward")

    @pytest.mark.asyncio
    async def test_connect_turn_left(self, connector):
        move_input = MoveInput(action=MovementAction.TURN_LEFT)
        with patch("actions.move_ub.connector.yanshee_motion.logging") as mock_logging:
            await connector.connect(move_input)
            mock_logging.info.assert_any_call("UB command: turn left")

    @pytest.mark.asyncio
    async def test_connect_turn_right(self, connector):
        move_input = MoveInput(action=MovementAction.TURN_RIGHT)
        with patch("actions.move_ub.connector.yanshee_motion.logging") as mock_logging:
            await connector.connect(move_input)
            mock_logging.info.assert_any_call("UB command: turn right")

    @pytest.mark.asyncio
    async def test_connect_bow(self, connector):
        move_input = MoveInput(action=MovementAction.BOW)
        with patch("actions.move_ub.connector.yanshee_motion.logging") as mock_logging:
            await connector.connect(move_input)
            mock_logging.info.assert_any_call("UB command: bow")

    @pytest.mark.asyncio
    async def test_connect_reset(self, connector):
        move_input = MoveInput(action=MovementAction.STAND_STILL)
        with patch("actions.move_ub.connector.yanshee_motion.logging") as mock_logging:
            await connector.connect(move_input)
            mock_logging.info.assert_any_call("UB command: stand still")

    @pytest.mark.asyncio
    async def test_connect_hug(self, connector):
        move_input = MoveInput(action=MovementAction.HUG)
        with patch("actions.move_ub.connector.yanshee_motion.logging") as mock_logging:
            await connector.connect(move_input)
            mock_logging.info.assert_any_call("UB command: hug")

    @pytest.mark.asyncio
    async def test_connect_walk_left(self, connector):
        move_input = MoveInput(action=MovementAction.WALK_LEFT)
        with patch("actions.move_ub.connector.yanshee_motion.logging") as mock_logging:
            await connector.connect(move_input)
            mock_logging.info.assert_any_call("UB command: walk left")

    @pytest.mark.asyncio
    async def test_connect_walk_right(self, connector):
        move_input = MoveInput(action=MovementAction.WALK_RIGHT)
        with patch("actions.move_ub.connector.yanshee_motion.logging") as mock_logging:
            await connector.connect(move_input)
            mock_logging.info.assert_any_call("UB command: walk right")

    @pytest.mark.asyncio
    async def test_connect_look_left(self, connector):
        move_input = MoveInput(action=MovementAction.LOOK_LEFT)
        with patch("actions.move_ub.connector.yanshee_motion.logging") as mock_logging:
            await connector.connect(move_input)
            mock_logging.info.assert_any_call("UB command: look left")

    @pytest.mark.asyncio
    async def test_connect_look_right(self, connector):
        move_input = MoveInput(action=MovementAction.LOOK_RIGHT)
        with patch("actions.move_ub.connector.yanshee_motion.logging") as mock_logging:
            await connector.connect(move_input)
            mock_logging.info.assert_any_call("UB command: look right")

    @pytest.mark.asyncio
    async def test_connect_crouch(self, connector):
        move_input = MoveInput(action=MovementAction.CROUCH)
        with patch("actions.move_ub.connector.yanshee_motion.logging") as mock_logging:
            await connector.connect(move_input)
            mock_logging.info.assert_any_call("UB command: crouch")

    @pytest.mark.asyncio
    async def test_connect_come_on(self, connector):
        move_input = MoveInput(action=MovementAction.COME)
        with patch("actions.move_ub.connector.yanshee_motion.logging") as mock_logging:
            await connector.connect(move_input)
            mock_logging.info.assert_any_call("UB command: come on")

    @pytest.mark.asyncio
    async def test_connect_waka_waka(self, connector):
        move_input = MoveInput(action=MovementAction.WAKAWAKA)
        with patch("actions.move_ub.connector.yanshee_motion.logging") as mock_logging:
            await connector.connect(move_input)
            mock_logging.info.assert_any_call("UB command: waka waka")

    @pytest.mark.asyncio
    async def test_connect_raise_right_hand(self, connector):
        move_input = MoveInput(action=MovementAction.RAISE_RIGHT_HAND)
        with patch("actions.move_ub.connector.yanshee_motion.logging") as mock_logging:
            await connector.connect(move_input)
            mock_logging.info.assert_any_call("UB command: raise right hand")

    @pytest.mark.asyncio
    async def test_connect_push_up(self, connector):
        move_input = MoveInput(action=MovementAction.PUSH_UP)
        with patch("actions.move_ub.connector.yanshee_motion.logging") as mock_logging:
            await connector.connect(move_input)
            mock_logging.info.assert_any_call("UB command: push up")

    @pytest.mark.asyncio
    async def test_connect_unknown_action(self, connector):
        move_input = MoveInput(action="fly")  # type: ignore[arg-type]
        with patch("actions.move_ub.connector.yanshee_motion.logging") as mock_logging:
            await connector.connect(move_input)
            mock_logging.info.assert_any_call("Unknown move type: fly")


class TestMoveYansheeConnectorTick:
    """Test tick method."""

    def test_tick_calls_sleep(self):
        with patch("actions.move_ub.connector.yanshee_motion.YanAPI"):
            config = MoveYansheeConfig()
            connector = MoveYansheeConnector(config)
            with patch.object(connector, "sleep") as mock_sleep:
                connector.tick()
                mock_sleep.assert_called_once_with(0.1)
