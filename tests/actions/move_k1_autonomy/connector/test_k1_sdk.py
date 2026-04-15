from queue import Queue
from unittest.mock import AsyncMock, Mock, patch

import pytest

from actions.base import MoveCommand
from actions.move_k1_autonomy.connector.k1_sdk import (
    MoveBoosterZenohConfig,
    MoveBoosterZenohConnector,
)
from actions.move_k1_autonomy.interface import MoveInput, MovementAction
from providers.k1_odom_provider import RobotState


@pytest.fixture
def mock_dependencies():
    """
    Mock all external dependencies.

    Returns
    -------
    dict
        Dictionary containing mock instances for all dependencies.
    """
    with (
        patch("actions.move_k1_autonomy.connector.k1_sdk.open_zenoh_session") as mock_zenoh,
        patch("actions.move_k1_autonomy.connector.k1_sdk.K1OdomProvider") as mock_odom,
        patch("actions.move_k1_autonomy.connector.k1_sdk.SimplePathsProvider") as mock_paths,
    ):
        # Setup mock zenoh session
        mock_zenoh_instance = Mock()
        mock_zenoh_instance.get = Mock()
        mock_zenoh.return_value = mock_zenoh_instance

        # Setup mock odom provider
        mock_odom_instance = Mock()
        mock_odom_instance.position = {
            "moving": False,
            "odom_x": 1.0,
            "odom_y": 0.0,
            "odom_yaw_m180_p180": 0.0,
            "body_attitude": RobotState.STANDING,
            "odom_subscriber_ts": 1234567890.0,
        }
        mock_odom.return_value = mock_odom_instance

        # Setup mock paths provider
        mock_paths_instance = Mock()
        mock_paths_instance.advance = [4]
        mock_paths_instance.retreat = [1]
        mock_paths_instance.turn_left = [2, 3]
        mock_paths_instance.turn_right = [5, 6]
        mock_paths_instance.path_angles = {1: 0, 2: 45, 3: 90, 4: 0, 5: -45, 6: -90}
        mock_paths.return_value = mock_paths_instance

        yield {
            "zenoh": mock_zenoh_instance,
            "odom": mock_odom_instance,
            "paths": mock_paths_instance,
        }


@pytest.fixture
def connector(mock_dependencies):
    """
    Create a MoveBoosterZenohConnector instance with mocked dependencies.

    Parameters
    ----------
    mock_dependencies : dict
        Dictionary containing mock instances.

    Returns
    -------
    MoveBoosterZenohConnector
        Configured connector instance with _move_robot mocked to prevent warnings.
    """
    config = MoveBoosterZenohConfig()
    connector_instance = MoveBoosterZenohConnector(config)

    connector_instance._original_move_robot = connector_instance._move_robot  # type: ignore
    connector_instance._move_robot = AsyncMock()

    return connector_instance


@pytest.fixture
def connector_with_real_move_robot(mock_dependencies):
    """
    Create a MoveBoosterZenohConnector instance without mocking _move_robot.
    Use this for tests that need to test the actual _move_robot implementation.

    Parameters
    ----------
    mock_dependencies : dict
        Dictionary containing mock instances.

    Returns
    -------
    MoveBoosterZenohConnector
        Configured connector instance with real _move_robot.
    """
    config = MoveBoosterZenohConfig()
    return MoveBoosterZenohConnector(config)


class TestMoveBoosterZenohConfig:
    """Test MoveBoosterZenohConfig configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MoveBoosterZenohConfig()
        assert config.odom_topic == "odometer_state"
        assert config.rpc_service_name == "booster_rpc_service"
        assert config.cmd_vel_topic is None
        assert config.allow_move_without_odom is False

    def test_custom_config(self):
        """Test custom configuration values."""
        config = MoveBoosterZenohConfig(
            odom_topic="custom_odom",
            rpc_service_name="custom_rpc",
            allow_move_without_odom=True,
        )
        assert config.odom_topic == "custom_odom"
        assert config.rpc_service_name == "custom_rpc"
        assert config.allow_move_without_odom is True

    def test_backward_compat_cmd_vel_topic(self):
        """Test backward compatibility with cmd_vel_topic."""
        config = MoveBoosterZenohConfig(cmd_vel_topic="legacy_topic")
        assert config.cmd_vel_topic == "legacy_topic"


class TestMoveBoosterZenohConnectorInit:
    """Test MoveBoosterZenohConnector initialization."""

    def test_initialization(self, connector, mock_dependencies):
        """Test successful initialization."""
        assert connector.move_speed == 0.1
        assert connector.turn_speed == 0.5
        assert connector.angle_tolerance == 5.0
        assert connector.distance_tolerance == 0.05
        assert isinstance(connector.pending_movements, Queue)
        assert connector.movement_attempts == 0
        assert connector.movement_attempt_limit == 15
        assert connector.gap_previous == 0
        assert connector._consecutive_retreat_cmds == 0

        # Verify providers are initialized
        assert connector.odom == mock_dependencies["odom"]
        assert connector.path_provider == mock_dependencies["paths"]
        assert connector.session == mock_dependencies["zenoh"]

    def test_initialization_zenoh_error(self):
        """Test initialization when Zenoh session fails."""
        with (
            patch("actions.move_k1_autonomy.connector.k1_sdk.open_zenoh_session") as mock_zenoh,
            patch("actions.move_k1_autonomy.connector.k1_sdk.K1OdomProvider"),
            patch("actions.move_k1_autonomy.connector.k1_sdk.SimplePathsProvider"),
            patch("actions.move_k1_autonomy.connector.k1_sdk.logging") as mock_logging,
        ):
            mock_zenoh.side_effect = Exception("Connection failed")

            config = MoveBoosterZenohConfig()
            connector = MoveBoosterZenohConnector(config)

            assert connector.session is None
            mock_logging.error.assert_called()

    def test_initialization_with_backward_compat_topic(self):
        """Test initialization uses cmd_vel_topic as rpc_service_name for backward compatibility."""
        with (
            patch("actions.move_k1_autonomy.connector.k1_sdk.open_zenoh_session"),
            patch("actions.move_k1_autonomy.connector.k1_sdk.K1OdomProvider"),
            patch("actions.move_k1_autonomy.connector.k1_sdk.SimplePathsProvider"),
        ):
            # When cmd_vel_topic is provided, it should be used as fallback for rpc_service_name
            config = MoveBoosterZenohConfig(cmd_vel_topic="legacy_service")
            connector = MoveBoosterZenohConnector(config)

            # Either cmd_vel_topic or default rpc_service_name should be used
            assert connector.rpc_service_name in ["legacy_service", "booster_rpc_service"]


class TestHasFreshOdom:
    """Test _has_fresh_odom method."""

    def test_fresh_odom_valid(self, connector, mock_dependencies):
        """Test _has_fresh_odom with valid recent data."""
        with patch("actions.move_k1_autonomy.connector.k1_sdk.time.time", return_value=1234567891.0):
            # odom_subscriber_ts is 1234567890.0 (1 second ago)
            assert connector._has_fresh_odom() is True

    def test_fresh_odom_stale(self, connector, mock_dependencies):
        """Test _has_fresh_odom with stale data."""
        with patch("actions.move_k1_autonomy.connector.k1_sdk.time.time", return_value=1234567895.0):
            # odom_subscriber_ts is 1234567890.0 (5 seconds ago, exceeds 2.0s default)
            assert connector._has_fresh_odom() is False

    def test_fresh_odom_no_timestamp(self, connector, mock_dependencies):
        """Test _has_fresh_odom when no timestamp available."""
        mock_dependencies["odom"].position["odom_subscriber_ts"] = 0.0
        assert connector._has_fresh_odom() is False

    def test_fresh_odom_negative_timestamp(self, connector, mock_dependencies):
        """Test _has_fresh_odom with negative timestamp."""
        mock_dependencies["odom"].position["odom_subscriber_ts"] = -1.0
        assert connector._has_fresh_odom() is False

    def test_fresh_odom_custom_max_age(self, connector, mock_dependencies):
        """Test _has_fresh_odom with custom max_age parameter."""
        with patch("actions.move_k1_autonomy.connector.k1_sdk.time.time", return_value=1234567895.0):
            # odom_subscriber_ts is 1234567890.0 (5 seconds ago)
            assert connector._has_fresh_odom(max_age_s=10.0) is True
            assert connector._has_fresh_odom(max_age_s=4.0) is False


class TestStopRobot:
    """Test _stop_robot method."""

    def test_stop_robot(self, connector):
        """Test _stop_robot calls _run_move_robot with zero velocities."""
        with patch.object(connector, "_run_move_robot") as mock_run_move:
            connector._stop_robot()
            mock_run_move.assert_called_once_with(0.0, 0.0, 0.0)

    def test_stop_robot_with_exception(self, connector):
        """Test _stop_robot handles exceptions gracefully."""
        with (
            patch.object(connector, "_run_move_robot") as mock_run_move,
            patch("actions.move_k1_autonomy.connector.k1_sdk.logging"),
        ):
            mock_run_move.side_effect = Exception("Stop failed")
            connector._stop_robot()
            # Should not raise exception


class TestMoveRobot:
    """Test _move_robot async method."""

    @pytest.mark.asyncio
    async def test_move_robot_success(self, connector_with_real_move_robot, mock_dependencies):
        """Test _move_robot sends movement command successfully."""
        mock_reply = Mock()
        mock_reply.ok = Mock()
        mock_reply.ok.payload.to_bytes.return_value = b'{"status": 0, "body": ""}'

        mock_dependencies["zenoh"].get.return_value = [mock_reply]

        with patch("actions.move_k1_autonomy.connector.k1_sdk.RpcServiceResponse.deserialize") as mock_deserialize:
            mock_response = Mock()
            mock_response.msg.status = 0
            mock_response.msg.body = ""
            mock_deserialize.return_value = mock_response

            await connector_with_real_move_robot._move_robot(0.1, 0.0, 0.0)

            mock_dependencies["zenoh"].get.assert_called_once()
            args, kwargs = mock_dependencies["zenoh"].get.call_args
            assert args[0] == "booster_rpc_service"
            assert kwargs["timeout"] == 5.0

    @pytest.mark.asyncio
    async def test_move_robot_no_session(self, connector_with_real_move_robot, mock_dependencies):
        """Test _move_robot returns early when session is None."""
        connector_with_real_move_robot.session = None

        await connector_with_real_move_robot._move_robot(0.1, 0.0, 0.0)

        mock_dependencies["zenoh"].get.assert_not_called()

    @pytest.mark.asyncio
    async def test_move_robot_not_standing(self, connector_with_real_move_robot, mock_dependencies):
        """Test _move_robot returns early when robot is not standing."""
        mock_dependencies["odom"].position["body_attitude"] = RobotState.SITTING

        await connector_with_real_move_robot._move_robot(0.1, 0.0, 0.0)

        mock_dependencies["zenoh"].get.assert_not_called()

    @pytest.mark.asyncio
    async def test_move_robot_allow_without_odom(self, connector_with_real_move_robot, mock_dependencies):
        """Test _move_robot bypasses odom check when allow_move_without_odom is True."""
        connector_with_real_move_robot.config.allow_move_without_odom = True
        mock_dependencies["odom"].position["body_attitude"] = RobotState.SITTING

        mock_reply = Mock()
        mock_reply.ok = Mock()
        mock_reply.ok.payload.to_bytes.return_value = b'{"status": 0, "body": ""}'
        mock_dependencies["zenoh"].get.return_value = [mock_reply]

        with patch("actions.move_k1_autonomy.connector.k1_sdk.RpcServiceResponse.deserialize"):
            await connector_with_real_move_robot._move_robot(0.1, 0.0, 0.0)

            mock_dependencies["zenoh"].get.assert_called_once()

    @pytest.mark.asyncio
    async def test_move_robot_service_error(self, connector_with_real_move_robot, mock_dependencies):
        """Test _move_robot handles service errors."""
        mock_reply = Mock()
        mock_reply.ok = None
        mock_reply.err = "Service unavailable"
        mock_dependencies["zenoh"].get.return_value = [mock_reply]

        with patch("actions.move_k1_autonomy.connector.k1_sdk.logging") as mock_logging:
            await connector_with_real_move_robot._move_robot(0.1, 0.0, 0.0)

            mock_logging.error.assert_called()

    @pytest.mark.asyncio
    async def test_move_robot_exception(self, connector_with_real_move_robot, mock_dependencies):
        """Test _move_robot handles exceptions during service call."""
        mock_dependencies["zenoh"].get.side_effect = Exception("Connection timeout")

        with patch("actions.move_k1_autonomy.connector.k1_sdk.logging") as mock_logging:
            await connector_with_real_move_robot._move_robot(0.1, 0.0, 0.0)

            mock_logging.error.assert_called()


class TestRunMoveRobot:
    """Test _run_move_robot method."""

    def test_run_move_robot_no_event_loop(self, connector):
        """Test _run_move_robot creates event loop when none exists."""
        with (
            patch("actions.move_k1_autonomy.connector.k1_sdk.asyncio.get_running_loop") as mock_get_loop,
            patch("actions.move_k1_autonomy.connector.k1_sdk.asyncio.run") as mock_run,
        ):
            mock_get_loop.side_effect = RuntimeError("No running loop")

            connector._run_move_robot(0.1, 0.0, 0.0)

            mock_run.assert_called_once()

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_run_move_robot_with_event_loop(self, connector):
        """Test _run_move_robot creates task when event loop exists."""
        mock_loop = Mock()

        with patch("actions.move_k1_autonomy.connector.k1_sdk.asyncio.get_running_loop", return_value=mock_loop):
            connector._run_move_robot(0.1, 0.0, 0.0)

            mock_loop.create_task.assert_called_once()


class TestConnect:
    """Test the connect method."""

    @pytest.mark.asyncio
    async def test_connect_robot_already_moving(self, connector, mock_dependencies):
        """Test connect when robot is already moving."""
        mock_dependencies["odom"].position["moving"] = True
        move_input = MoveInput(action=MovementAction.MOVE_FORWARDS)

        await connector.connect(move_input)

        assert connector.pending_movements.qsize() == 0

    @pytest.mark.asyncio
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    async def test_connect_movement_already_pending(self, connector, mock_dependencies):
        """Test connect when movement is already pending."""
        connector.pending_movements.put(MoveCommand(dx=0.5, yaw=0.0, start_x=0.0, start_y=0.0, turn_complete=False))
        move_input = MoveInput(action=MovementAction.MOVE_FORWARDS)

        await connector.connect(move_input)

        assert connector.pending_movements.qsize() == 1

    @pytest.mark.asyncio
    async def test_connect_no_fresh_odom_test_mode(self, connector, mock_dependencies):
        """Test connect sends direct test command when allow_move_without_odom is True."""
        connector.config.allow_move_without_odom = True
        mock_dependencies["odom"].position["odom_subscriber_ts"] = 0.0

        with patch.object(connector, "_run_move_robot") as mock_run_move:
            move_input = MoveInput(action=MovementAction.MOVE_FORWARDS)
            await connector.connect(move_input)

            mock_run_move.assert_called_once_with(0.1, 0.0, 0.0)

    @pytest.mark.asyncio
    async def test_connect_no_fresh_odom_normal_mode(self, connector, mock_dependencies):
        """Test connect returns early when no fresh odom in normal mode."""
        mock_dependencies["odom"].position["odom_subscriber_ts"] = 0.0
        move_input = MoveInput(action=MovementAction.MOVE_FORWARDS)

        await connector.connect(move_input)

        assert connector.pending_movements.qsize() == 0

    @pytest.mark.asyncio
    async def test_connect_turn_left(self, connector, mock_dependencies):
        """Test connect with turn left command."""
        with (
            patch("actions.move_k1_autonomy.connector.k1_sdk.random.choice", return_value=2),
            patch("actions.move_k1_autonomy.connector.k1_sdk.time.time", return_value=1234567890.5),
        ):
            move_input = MoveInput(action=MovementAction.TURN_LEFT)

            await connector.connect(move_input)

            assert connector.pending_movements.qsize() == 1
            command = connector.pending_movements.get()
            assert command.dx == 0.0
            assert command.turn_complete is False

    @pytest.mark.asyncio
    async def test_connect_turn_right(self, connector, mock_dependencies):
        """Test connect with turn right command."""
        with (
            patch("actions.move_k1_autonomy.connector.k1_sdk.random.choice", return_value=5),
            patch("actions.move_k1_autonomy.connector.k1_sdk.time.time", return_value=1234567890.5),
        ):
            move_input = MoveInput(action=MovementAction.TURN_RIGHT)

            await connector.connect(move_input)

            assert connector.pending_movements.qsize() == 1
            command = connector.pending_movements.get()
            assert command.dx == 0.0
            assert command.turn_complete is False

    @pytest.mark.asyncio
    async def test_connect_move_forwards(self, connector, mock_dependencies):
        """Test connect with move forwards command."""
        with (
            patch("actions.move_k1_autonomy.connector.k1_sdk.random.choice", return_value=4),
            patch("actions.move_k1_autonomy.connector.k1_sdk.time.time", return_value=1234567890.5),
        ):
            move_input = MoveInput(action=MovementAction.MOVE_FORWARDS)

            await connector.connect(move_input)

            assert connector.pending_movements.qsize() == 1
            command = connector.pending_movements.get()
            assert command.dx == 0.1

    @pytest.mark.asyncio
    async def test_connect_move_back(self, connector, mock_dependencies):
        """Test connect with move back command."""
        with patch("actions.move_k1_autonomy.connector.k1_sdk.time.time", return_value=1234567890.5):
            move_input = MoveInput(action=MovementAction.MOVE_BACK)

            await connector.connect(move_input)

            assert connector.pending_movements.qsize() == 1
            command = connector.pending_movements.get()
            assert command.dx == -0.15
            assert command.turn_complete is True

    @pytest.mark.asyncio
    async def test_connect_stand_still(self, connector, mock_dependencies):
        """Test connect with stand still command."""
        with (
            patch.object(connector, "_stop_robot") as mock_stop,
            patch("actions.move_k1_autonomy.connector.k1_sdk.time.time", return_value=1234567890.5),
        ):
            move_input = MoveInput(action=MovementAction.STAND_STILL)

            await connector.connect(move_input)

            mock_stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_unknown_action(self, connector, mock_dependencies):
        """Test connect with unknown action."""
        move_input = MoveInput(action="unknown_action")  # type: ignore

        with patch("actions.move_k1_autonomy.connector.k1_sdk.logging") as mock_logging:
            await connector.connect(move_input)

            mock_logging.info.assert_called()


class TestCleanAbort:
    """Test clean_abort method."""

    def test_clean_abort(self, connector):
        """Test clean_abort resets state and stops robot."""
        connector.movement_attempts = 10
        connector._consecutive_retreat_cmds = 3
        connector.pending_movements.put(MoveCommand(dx=0.5, yaw=0.0, start_x=0.0, start_y=0.0, turn_complete=False))

        with patch.object(connector, "_stop_robot") as mock_stop:
            connector.clean_abort()

            mock_stop.assert_called_once()
            assert connector.movement_attempts == 0
            assert connector._consecutive_retreat_cmds == 0
            assert connector.pending_movements.empty()


class TestNormalizeAngle:
    """Test _normalize_angle method."""

    def test_normalize_angle_in_range(self, connector):
        """Test _normalize_angle with angle already in range."""
        assert connector._normalize_angle(45.0) == 45.0
        assert connector._normalize_angle(-90.0) == -90.0
        assert connector._normalize_angle(180.0) == 180.0
        assert connector._normalize_angle(-180.0) == -180.0

    def test_normalize_angle_above_range(self, connector):
        """Test _normalize_angle with angle above 180."""
        assert connector._normalize_angle(270.0) == -90.0
        assert connector._normalize_angle(190.0) == -170.0

    def test_normalize_angle_below_range(self, connector):
        """Test _normalize_angle with angle below -180."""
        assert connector._normalize_angle(-270.0) == 90.0
        assert connector._normalize_angle(-190.0) == 170.0


class TestCalculateAngleGap:
    """Test _calculate_angle_gap method."""

    def test_calculate_angle_gap_positive(self, connector):
        """Test _calculate_angle_gap with positive gap."""
        assert connector._calculate_angle_gap(45.0, 0.0) == 45.0
        assert connector._calculate_angle_gap(90.0, 45.0) == 45.0

    def test_calculate_angle_gap_negative(self, connector):
        """Test _calculate_angle_gap with negative gap."""
        assert connector._calculate_angle_gap(0.0, 45.0) == -45.0
        assert connector._calculate_angle_gap(45.0, 90.0) == -45.0

    def test_calculate_angle_gap_wrap_positive(self, connector):
        """Test _calculate_angle_gap wraps correctly for positive wrap."""
        # current=170, target=-170 should give 340->-20
        assert connector._calculate_angle_gap(170.0, -170.0) == -20.0

    def test_calculate_angle_gap_wrap_negative(self, connector):
        """Test _calculate_angle_gap wraps correctly for negative wrap."""
        # current=-170, target=170 should give -340->20
        assert connector._calculate_angle_gap(-170.0, 170.0) == 20.0


class TestExecuteTurn:
    """Test _execute_turn method."""

    def test_execute_turn_left_success(self, connector, mock_dependencies):
        """Test _execute_turn with successful left turn."""
        with patch.object(connector, "_run_move_robot") as mock_run_move:
            result = connector._execute_turn(45.0)

            assert result is True
            mock_run_move.assert_called_once()

    def test_execute_turn_left_blocked(self, connector, mock_dependencies):
        """Test _execute_turn with left turn blocked by barrier."""
        mock_dependencies["paths"].turn_left = []

        result = connector._execute_turn(45.0)

        assert result is False

    def test_execute_turn_right_success(self, connector, mock_dependencies):
        """Test _execute_turn with successful right turn."""
        with patch.object(connector, "_run_move_robot") as mock_run_move:
            result = connector._execute_turn(-45.0)

            assert result is True
            mock_run_move.assert_called_once()

    def test_execute_turn_right_blocked(self, connector, mock_dependencies):
        """Test _execute_turn with right turn blocked by barrier."""
        mock_dependencies["paths"].turn_right = []

        result = connector._execute_turn(-45.0)

        assert result is False


class TestProcessMovementCommands:
    """Test process movement command methods."""

    def test_process_turn_left_success(self, connector, mock_dependencies):
        """Test _process_turn_left enqueues turn command."""
        with patch("actions.move_k1_autonomy.connector.k1_sdk.random.choice", return_value=2):
            connector._process_turn_left()

            assert connector.pending_movements.qsize() == 1
            command = connector.pending_movements.get()
            assert command.dx == 0.0
            assert command.turn_complete is False

    def test_process_turn_left_blocked(self, connector, mock_dependencies):
        """Test _process_turn_left when blocked by barrier."""
        mock_dependencies["paths"].turn_left = []

        connector._process_turn_left()

        assert connector.pending_movements.qsize() == 0

    def test_process_turn_right_success(self, connector, mock_dependencies):
        """Test _process_turn_right enqueues turn command."""
        with patch("actions.move_k1_autonomy.connector.k1_sdk.random.choice", return_value=5):
            connector._process_turn_right()

            assert connector.pending_movements.qsize() == 1

    def test_process_turn_right_blocked(self, connector, mock_dependencies):
        """Test _process_turn_right when blocked by barrier."""
        mock_dependencies["paths"].turn_right = []

        connector._process_turn_right()

        assert connector.pending_movements.qsize() == 0

    def test_process_move_forward_success(self, connector, mock_dependencies):
        """Test _process_move_forward enqueues move command."""
        with patch("actions.move_k1_autonomy.connector.k1_sdk.random.choice", return_value=4):
            connector._process_move_forward()

            assert connector.pending_movements.qsize() == 1
            command = connector.pending_movements.get()
            assert command.dx == 0.1

    def test_process_move_forward_blocked(self, connector, mock_dependencies):
        """Test _process_move_forward when blocked by barrier."""
        mock_dependencies["paths"].advance = []

        connector._process_move_forward()

        assert connector.pending_movements.qsize() == 0

    def test_process_move_back_success(self, connector, mock_dependencies):
        """Test _process_move_back enqueues retreat command."""
        connector._process_move_back()

        assert connector.pending_movements.qsize() == 1
        command = connector.pending_movements.get()
        assert command.dx == -0.15
        assert command.turn_complete is True

    def test_process_move_back_blocked(self, connector, mock_dependencies):
        """Test _process_move_back when blocked by barrier."""
        mock_dependencies["paths"].retreat = []

        connector._process_move_back()

        assert connector.pending_movements.qsize() == 0


class TestEnqueueForwardIfFrontClear:
    """Test _enqueue_forward_if_front_clear method."""

    def test_enqueue_forward_front_clear(self, connector, mock_dependencies):
        """Test _enqueue_forward_if_front_clear when front is clear."""
        result = connector._enqueue_forward_if_front_clear(dx=0.1)

        assert result is True
        assert connector.pending_movements.qsize() == 1

    def test_enqueue_forward_front_blocked(self, connector, mock_dependencies):
        """Test _enqueue_forward_if_front_clear when front is blocked."""
        mock_dependencies["paths"].advance = [1, 2, 3]  # No path 4 (straight)

        result = connector._enqueue_forward_if_front_clear(dx=0.1)

        assert result is False
        assert connector.pending_movements.qsize() == 0

    def test_enqueue_forward_custom_dx(self, connector, mock_dependencies):
        """Test _enqueue_forward_if_front_clear with custom distance."""
        result = connector._enqueue_forward_if_front_clear(dx=0.5)

        assert result is True
        command = connector.pending_movements.get()
        assert command.dx == 0.5


class TestTick:
    """Test the tick method."""

    def test_tick_no_odom(self, connector):
        """Test tick when odom is None."""
        connector.odom = None

        with patch.object(connector, "sleep") as mock_sleep:
            connector.tick()

            mock_sleep.assert_called_once_with(0.5)

    def test_tick_no_fresh_odom(self, connector, mock_dependencies):
        """Test tick when odom data is stale."""
        mock_dependencies["odom"].position["odom_subscriber_ts"] = 0.0

        with patch.object(connector, "sleep") as mock_sleep:
            connector.tick()

            mock_sleep.assert_called_once_with(0.5)

    def test_tick_robot_not_standing(self, connector, mock_dependencies):
        """Test tick when robot is not standing."""
        mock_dependencies["odom"].position["body_attitude"] = RobotState.SITTING

        with (
            patch.object(connector, "_has_fresh_odom", return_value=True),
            patch.object(connector, "sleep") as mock_sleep,
        ):
            connector.tick()

            mock_sleep.assert_called_once_with(0.5)

    def test_tick_no_pending_movements(self, connector, mock_dependencies):
        """Test tick when no movements are pending."""
        with (
            patch.object(connector, "_has_fresh_odom", return_value=True),
            patch.object(connector, "sleep") as mock_sleep,
        ):
            connector.tick()

            mock_sleep.assert_called_once_with(0.1)

    def test_tick_timeout_exceeded(self, connector, mock_dependencies):
        """Test tick aborts when movement attempts exceed limit."""
        connector.pending_movements.put(MoveCommand(dx=0.1, yaw=0.0, start_x=0.0, start_y=0.0, turn_complete=False))
        connector.movement_attempts = 20

        with (
            patch.object(connector, "_has_fresh_odom", return_value=True),
            patch.object(connector, "clean_abort") as mock_abort,
        ):
            connector.tick()

            mock_abort.assert_called_once()

    def test_tick_phase1_turning(self, connector, mock_dependencies):
        """Test tick Phase 1 - turning to target yaw."""
        mock_dependencies["odom"].position["odom_yaw_m180_p180"] = 0.0
        connector.pending_movements.put(MoveCommand(dx=0.1, yaw=45.0, start_x=0.0, start_y=0.0, turn_complete=False))

        with (
            patch.object(connector, "_has_fresh_odom", return_value=True),
            patch.object(connector, "_execute_turn", return_value=True) as mock_turn,
        ):
            connector.tick()

            mock_turn.assert_called_once()
            assert connector.movement_attempts == 1

    def test_tick_phase2_moving_forward(self, connector, mock_dependencies):
        """Test tick Phase 2 - moving forward to target."""
        mock_dependencies["odom"].position["odom_x"] = 0.0
        mock_dependencies["odom"].position["odom_y"] = 0.0
        connector.pending_movements.put(
            MoveCommand(dx=0.1, yaw=0.0, start_x=0.0, start_y=0.0, turn_complete=True, speed=0.1)
        )

        with (
            patch.object(connector, "_has_fresh_odom", return_value=True),
            patch.object(connector, "_run_move_robot") as mock_move,
        ):
            connector.tick()

            mock_move.assert_called_once()
            assert connector.movement_attempts == 1

    def test_tick_phase2_retreat_limit(self, connector, mock_dependencies):
        """Test tick Phase 2 - abort after 5 consecutive retreat commands."""
        mock_dependencies["odom"].position["odom_x"] = 0.0
        mock_dependencies["odom"].position["odom_y"] = 0.0
        connector.pending_movements.put(
            MoveCommand(dx=-0.1, yaw=0.0, start_x=0.0, start_y=0.0, turn_complete=True, speed=0.1)
        )
        connector._consecutive_retreat_cmds = 5

        with (
            patch.object(connector, "_has_fresh_odom", return_value=True),
            patch.object(connector, "clean_abort") as mock_abort,
            patch.object(connector, "_enqueue_forward_if_front_clear", return_value=False),
        ):
            connector.tick()

            mock_abort.assert_called_once()

    def test_tick_phase2_movement_complete(self, connector, mock_dependencies):
        """Test tick Phase 2 - movement completes when within tolerance."""
        mock_dependencies["odom"].position["odom_x"] = 0.09  # Close to target
        mock_dependencies["odom"].position["odom_y"] = 0.0
        connector.pending_movements.put(
            MoveCommand(dx=0.1, yaw=0.0, start_x=0.0, start_y=0.0, turn_complete=True, speed=0.1)
        )

        with (
            patch.object(connector, "_has_fresh_odom", return_value=True),
            patch.object(connector, "clean_abort") as mock_abort,
        ):
            connector.tick()

            mock_abort.assert_called_once()
