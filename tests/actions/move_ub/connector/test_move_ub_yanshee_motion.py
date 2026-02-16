import concurrent.futures
import sys
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Mock ubtech modules
sys.modules["ubtech"] = MagicMock()
sys.modules["ubtech.ubtechapi"] = MagicMock()

from actions.move_ub.connector.yanshee_motion import (  # noqa: E402
    Motion,
    MoveYansheeConfig,
    MoveYansheeConnector,
)
from actions.move_ub.interface import MoveInput, MovementAction  # noqa: E402


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

    def test_init_startup_exception(self):
        """Cover lines 140-141: exception during startup."""
        with patch("actions.move_ub.connector.yanshee_motion.YanAPI") as mock_api:
            mock_api.yan_api_init.return_value = None
            config = MoveYansheeConfig()
            with patch.object(
                MoveYansheeConnector, "_send_command", side_effect=Exception("Forced")
            ):
                connector = MoveYansheeConnector(config)
                assert connector is not None


class TestSendCommand:
    """Test _send_command method."""

    @pytest.fixture
    def connector(self):
        with patch("actions.move_ub.connector.yanshee_motion.YanAPI"):
            config = MoveYansheeConfig()
            return MoveYansheeConnector(config)

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

    def test_send_command_unexpected_error(self, connector):
        with (
            patch("actions.move_ub.connector.yanshee_motion.YanAPI") as mock_api,
            patch("actions.move_ub.connector.yanshee_motion.logging") as mock_log,
        ):
            mock_api.sync_play_motion.side_effect = RuntimeError("Unexpected error")
            result = connector._send_command(Motion("wave"))
            assert result is False
            mock_log.error.assert_called()


class TestConnect:
    """Test connect method."""

    @pytest.fixture
    def connector(self):
        with patch("actions.move_ub.connector.yanshee_motion.YanAPI"):
            config = MoveYansheeConfig()
            connector = MoveYansheeConnector(config)
            connector._execute_sport_command = AsyncMock()
            return connector

    @pytest.mark.asyncio
    async def test_connect_all_actions(self, connector):
        """Test all movement actions."""
        actions_map = {
            MovementAction.WAVE: "wave",
            MovementAction.WALK_FORWARD: "walk forward",
            MovementAction.WALK_BACKWARD: "walk backward",
            MovementAction.WALK_LEFT: "walk left",
            MovementAction.WALK_RIGHT: "walk right",
            MovementAction.TURN_LEFT: "turn left",
            MovementAction.TURN_RIGHT: "turn right",
            MovementAction.LOOK_LEFT: "look left",
            MovementAction.LOOK_RIGHT: "look right",
            MovementAction.BOW: "bow",
            MovementAction.CROUCH: "crouch",
            MovementAction.COME: "come on",
            MovementAction.WAKAWAKA: "waka waka",
            MovementAction.HUG: "hug",
            MovementAction.RAISE_RIGHT_HAND: "raise right hand",
            MovementAction.PUSH_UP: "push up",
            MovementAction.STAND_STILL: "stand still",
        }

        for action, expected_log in actions_map.items():
            with patch(
                "actions.move_ub.connector.yanshee_motion.logging"
            ) as mock_logging:
                await connector.connect(MoveInput(action=action))
                mock_logging.info.assert_any_call(f"UB command: {expected_log}")

    @pytest.mark.asyncio
    async def test_connect_unknown_action(self, connector):
        move_input = MoveInput(action="fly")  # type: ignore[arg-type]
        with patch("actions.move_ub.connector.yanshee_motion.logging") as mock_logging:
            await connector.connect(move_input)
            mock_logging.info.assert_any_call("Unknown move type: fly")


class TestThreading:
    """Test threading methods."""

    @pytest.fixture
    def connector(self):
        with patch("actions.move_ub.connector.yanshee_motion.YanAPI"):
            config = MoveYansheeConfig()
            return MoveYansheeConnector(config)

    def test_execute_command_thread_success(self, connector):
        """Test successful command execution in thread."""
        connector.thread_lock.acquire()
        with patch.object(connector, "_send_command", return_value=True):
            connector._execute_command_thread(Motion("wave"))
            assert not connector.thread_lock.locked()

    def test_execute_command_thread_error(self, connector):
        """Test error handling in command thread."""
        connector.thread_lock.acquire()
        with patch.object(
            connector, "_send_command", side_effect=Exception("Thread error")
        ):
            connector._execute_command_thread(Motion("wave"))
            assert not connector.thread_lock.locked()

    def test_execute_sport_command_sync_success(self, connector):
        """Test sync sport command execution."""
        with patch(
            "actions.move_ub.connector.yanshee_motion.threading.Thread"
        ) as mock_thread:
            mock_thread.return_value.start = Mock()
            connector._execute_sport_command_sync(Motion("wave"))
            mock_thread.assert_called_once()

    def test_execute_sport_command_sync_lock_busy(self, connector):
        """Test sync command when lock is busy."""
        connector.thread_lock.acquire()
        connector._execute_sport_command_sync(Motion("wave"))
        connector.thread_lock.release()

    def test_execute_sport_command_sync_thread_error(self, connector):
        """Test sync command with thread creation error."""
        with patch(
            "actions.move_ub.connector.yanshee_motion.threading.Thread",
            side_effect=Exception("Thread error"),
        ):
            connector._execute_sport_command_sync(Motion("wave"))
            assert not connector.thread_lock.locked()

    @pytest.mark.asyncio
    async def test_execute_sport_command_async_success(self, connector):
        """Test async sport command execution."""
        with patch(
            "actions.move_ub.connector.yanshee_motion.threading.Thread"
        ) as mock_thread:
            mock_thread.return_value.start = Mock()
            await connector._execute_sport_command(Motion("bow"))
            mock_thread.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_sport_command_async_lock_busy(self, connector):
        """Test async command when lock is busy."""
        connector.thread_lock.acquire()
        await connector._execute_sport_command(Motion("bow"))
        connector.thread_lock.release()

    @pytest.mark.asyncio
    async def test_execute_sport_command_async_thread_error(self, connector):
        """Test async command with thread creation error."""
        with patch(
            "actions.move_ub.connector.yanshee_motion.threading.Thread",
            side_effect=Exception("Thread error"),
        ):
            await connector._execute_sport_command(Motion("bow"))
            assert not connector.thread_lock.locked()


class TestTick:
    """Test tick method."""

    def test_tick_calls_sleep(self):
        with patch("actions.move_ub.connector.yanshee_motion.YanAPI"):
            config = MoveYansheeConfig()
            connector = MoveYansheeConnector(config)
            with patch.object(connector, "sleep") as mock_sleep:
                connector.tick()
                mock_sleep.assert_called_once_with(0.1)
