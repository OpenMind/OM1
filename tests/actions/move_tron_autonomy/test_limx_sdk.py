from queue import Queue
from unittest.mock import Mock, patch

import pytest

from actions.base import MoveCommand
from actions.move_tron_autonomy.connector.limx_sdk import (
    MoveTronZenohConfig,
    MoveTronZenohConnector,
)
from actions.move_tron_autonomy.interface import MoveInput, MovementAction
from providers.tron_odom_provider import RobotState


@pytest.fixture
def mock_dependencies():
    """Mock all external dependencies."""
    with (
        patch(
            "actions.move_tron_autonomy.connector.limx_sdk.SimplePathsProvider"
        ) as mock_paths,
        patch(
            "actions.move_tron_autonomy.connector.limx_sdk.TronOdomProvider"
        ) as mock_odom,
        patch(
            "actions.move_tron_autonomy.connector.limx_sdk.open_zenoh_session"
        ) as mock_zenoh,
    ):
        # Setup mock instances
        mock_paths_instance = Mock()
        mock_paths_instance.advance = [4]
        mock_paths_instance.retreat = [1]
        mock_paths_instance.turn_left = [2]
        mock_paths_instance.turn_right = [6]
        mock_paths_instance.path_angles = {1: 0, 2: 45, 4: 0, 6: -45}
        mock_paths.return_value = mock_paths_instance

        mock_odom_instance = Mock()
        mock_odom_instance.position = {
            "moving": False,
            "odom_x": 1.0,
            "odom_y": 0.0,
            "odom_yaw_m180_p180": 0.0,
            "body_attitude": RobotState.STANDING,
        }
        mock_odom.return_value = mock_odom_instance

        mock_session = Mock()
        mock_session.put = Mock()
        mock_zenoh.return_value = mock_session

        yield {
            "paths": mock_paths_instance,
            "odom": mock_odom_instance,
            "session": mock_session,
        }


@pytest.fixture
def connector(mock_dependencies):
    """Create a MoveTronZenohConnector instance with mocked dependencies."""
    config = MoveTronZenohConfig()
    connector = MoveTronZenohConnector(config)
    return connector


class TestMoveTronZenohConfig:
    """Test MoveTronZenohConfig configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MoveTronZenohConfig()
        assert config.odom_topic == "odom"
        assert config.cmd_vel_topic == "cmd_vel"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = MoveTronZenohConfig(
            odom_topic="custom_odom", cmd_vel_topic="custom_cmd_vel"
        )
        assert config.odom_topic == "custom_odom"
        assert config.cmd_vel_topic == "custom_cmd_vel"


class TestMoveTronZenohConnectorInit:
    """Test MoveTronZenohConnector initialization."""

    def test_initialization(self, connector, mock_dependencies):
        """Test successful initialization."""
        assert connector.move_speed == 0.25
        assert connector.turn_speed == 0.35
        assert connector.angle_tolerance == 5.0
        assert connector.distance_tolerance == 0.05
        assert isinstance(connector.pending_movements, Queue)
        assert connector.movement_attempts == 0
        assert connector.movement_attempt_limit == 15
        assert connector.gap_previous == 0

        # Verify providers are initialized
        assert connector.path_provider == mock_dependencies["paths"]
        assert connector.odom == mock_dependencies["odom"]
        assert connector.session == mock_dependencies["session"]

    def test_initialization_zenoh_error(self):
        """Test initialization when Zenoh session fails."""
        with (
            patch("actions.move_tron_autonomy.connector.limx_sdk.SimplePathsProvider"),
            patch("actions.move_tron_autonomy.connector.limx_sdk.TronOdomProvider"),
            patch(
                "actions.move_tron_autonomy.connector.limx_sdk.open_zenoh_session"
            ) as mock_zenoh,
            patch(
                "actions.move_tron_autonomy.connector.limx_sdk.logging"
            ) as mock_logging,
        ):
            mock_zenoh.side_effect = Exception("Connection failed")

            config = MoveTronZenohConfig()
            connector = MoveTronZenohConnector(config)

            assert connector.session is None
            mock_logging.error.assert_called()


class TestConnect:
    """Test the connect method."""

    @pytest.mark.asyncio
    async def test_connect_robot_already_moving(self, connector, mock_dependencies):
        """Test connect when robot is already moving."""
        mock_dependencies["odom"].position["moving"] = True
        move_input = MoveInput(action=MovementAction.MOVE_FORWARDS)

        await connector.connect(move_input)

        # Should return early without processing
        assert connector.pending_movements.qsize() == 0

    @pytest.mark.asyncio
    async def test_connect_movement_already_pending(self, connector, mock_dependencies):
        """Test connect when movement is already pending."""
        connector.pending_movements.put(
            MoveCommand(dx=0.5, yaw=0.0, start_x=0.0, start_y=0.0, turn_complete=False)
        )
        move_input = MoveInput(action=MovementAction.MOVE_FORWARDS)

        await connector.connect(move_input)

        # Should still have only one movement
        assert connector.pending_movements.qsize() == 1

    @pytest.mark.asyncio
    async def test_connect_waiting_for_location_data(
        self, connector, mock_dependencies
    ):
        """Test connect when waiting for location data."""
        mock_dependencies["odom"].position["odom_x"] = 0.0
        move_input = MoveInput(action=MovementAction.MOVE_FORWARDS)

        await connector.connect(move_input)

        assert connector.pending_movements.qsize() == 0

    @pytest.mark.asyncio
    async def test_connect_turn_left(self, connector, mock_dependencies):
        """Test connect with turn left command."""
        move_input = MoveInput(action=MovementAction.TURN_LEFT)

        await connector.connect(move_input)

        assert connector.pending_movements.qsize() == 1
        command = connector.pending_movements.get()
        assert command.dx == 0.5
        assert command.turn_complete is False

    @pytest.mark.asyncio
    async def test_connect_turn_right(self, connector, mock_dependencies):
        """Test connect with turn right command."""
        move_input = MoveInput(action=MovementAction.TURN_RIGHT)

        await connector.connect(move_input)

        assert connector.pending_movements.qsize() == 1
        command = connector.pending_movements.get()
        assert command.dx == 0.5
        assert command.turn_complete is False

    @pytest.mark.asyncio
    async def test_connect_move_forwards(self, connector, mock_dependencies):
        """Test connect with move forwards command."""
        move_input = MoveInput(action=MovementAction.MOVE_FORWARDS)

        await connector.connect(move_input)

        assert connector.pending_movements.qsize() == 1
        command = connector.pending_movements.get()
        assert command.dx == 0.5

    @pytest.mark.asyncio
    async def test_connect_move_back(self, connector, mock_dependencies):
        """Test connect with move back command."""
        move_input = MoveInput(action=MovementAction.MOVE_BACK)

        await connector.connect(move_input)

        assert connector.pending_movements.qsize() == 1
        command = connector.pending_movements.get()
        assert command.dx == -0.5
        assert command.turn_complete is True
        assert command.speed == 0.25

    @pytest.mark.asyncio
    async def test_connect_stand_still(self, connector, mock_dependencies):
        """Test connect with stand still command."""
        move_input = MoveInput(action=MovementAction.STAND_STILL)

        await connector.connect(move_input)

        assert connector.pending_movements.qsize() == 0

    @pytest.mark.asyncio
    async def test_connect_unknown_action(self, connector, mock_dependencies):
        """Test connect with unknown action."""
        move_input = MoveInput(action="unknown action")  # type: ignore[arg-type]

        await connector.connect(move_input)

        assert connector.pending_movements.qsize() == 0


class TestMovementProcessing:
    """Test movement processing methods."""

    def test_process_turn_left_with_barrier(self, connector, mock_dependencies):
        """Test turn left when blocked by barrier."""
        mock_dependencies["paths"].turn_left = []

        connector._process_turn_left()

        assert connector.pending_movements.qsize() == 0

    def test_process_turn_left_success(self, connector, mock_dependencies):
        """Test successful turn left."""
        connector._process_turn_left()

        assert connector.pending_movements.qsize() == 1

    def test_process_turn_right_with_barrier(self, connector, mock_dependencies):
        """Test turn right when blocked by barrier."""
        mock_dependencies["paths"].turn_right = []

        connector._process_turn_right()

        assert connector.pending_movements.qsize() == 0

    def test_process_turn_right_success(self, connector, mock_dependencies):
        """Test successful turn right."""
        connector._process_turn_right()

        assert connector.pending_movements.qsize() == 1

    def test_process_move_forward_with_barrier(self, connector, mock_dependencies):
        """Test move forward when blocked by barrier."""
        mock_dependencies["paths"].advance = []

        connector._process_move_forward()

        assert connector.pending_movements.qsize() == 0

    def test_process_move_forward_success(self, connector, mock_dependencies):
        """Test successful move forward."""
        connector._process_move_forward()

        assert connector.pending_movements.qsize() == 1

    def test_process_move_back_with_barrier(self, connector, mock_dependencies):
        """Test move back when blocked by barrier."""
        mock_dependencies["paths"].retreat = []

        connector._process_move_back()

        assert connector.pending_movements.qsize() == 0

    def test_process_move_back_success(self, connector, mock_dependencies):
        """Test successful move back."""
        connector._process_move_back()

        assert connector.pending_movements.qsize() == 1
        command = connector.pending_movements.get()
        assert command.dx == -0.5


class TestMoveRobot:
    """Test _move_robot method."""

    def test_move_robot_no_session(self, connector, mock_dependencies):
        """Test move robot when session is None."""
        connector.session = None

        connector._move_robot(0.5, 0.0, 0.0)

        # Should not crash

    def test_move_robot_not_standing(self, connector, mock_dependencies):
        """Test move robot when robot is not standing."""
        mock_dependencies["odom"].position["body_attitude"] = RobotState.SITTING

        connector._move_robot(0.5, 0.0, 0.0)

        mock_dependencies["session"].put.assert_not_called()

    def test_move_robot_success(self, connector, mock_dependencies):
        """Test successful robot movement."""
        connector._move_robot(0.5, 0.0, 0.3)

        mock_dependencies["session"].put.assert_called_once()
        call_args = mock_dependencies["session"].put.call_args
        assert call_args[0][0] == "cmd_vel"


class TestCleanAbort:
    """Test clean_abort method."""

    def test_clean_abort_with_pending_movement(self, connector, mock_dependencies):
        """Test clean abort with pending movement."""
        connector.movement_attempts = 5
        connector.pending_movements.put(
            MoveCommand(dx=0.5, yaw=0.0, start_x=0.0, start_y=0.0, turn_complete=False)
        )

        connector.clean_abort()

        assert connector.movement_attempts == 0
        assert connector.pending_movements.qsize() == 0

    def test_clean_abort_empty_queue(self, connector, mock_dependencies):
        """Test clean abort with empty queue."""
        connector.movement_attempts = 5

        connector.clean_abort()

        assert connector.movement_attempts == 0
        assert connector.pending_movements.qsize() == 0


class TestAngleCalculations:
    """Test angle calculation methods."""

    def test_normalize_angle_positive_overflow(self, connector, mock_dependencies):
        """Test normalize angle with positive overflow."""
        result = connector._normalize_angle(270.0)
        assert result == -90.0

    def test_normalize_angle_negative_overflow(self, connector, mock_dependencies):
        """Test normalize angle with negative overflow."""
        result = connector._normalize_angle(-270.0)
        assert result == 90.0

    def test_normalize_angle_within_range(self, connector, mock_dependencies):
        """Test normalize angle within range."""
        result = connector._normalize_angle(45.0)
        assert result == 45.0

    def test_calculate_angle_gap_simple(self, connector, mock_dependencies):
        """Test calculate angle gap with simple case."""
        result = connector._calculate_angle_gap(10.0, 5.0)
        assert result == 5.0

    def test_calculate_angle_gap_wrap_positive(self, connector, mock_dependencies):
        """Test calculate angle gap with positive wrap."""
        result = connector._calculate_angle_gap(170.0, -170.0)
        assert result == -20.0

    def test_calculate_angle_gap_wrap_negative(self, connector, mock_dependencies):
        """Test calculate angle gap with negative wrap."""
        result = connector._calculate_angle_gap(-170.0, 170.0)
        assert result == 20.0


class TestExecuteTurn:
    """Test _execute_turn method."""

    def test_execute_turn_left_blocked(self, connector, mock_dependencies):
        """Test execute turn left when blocked."""
        mock_dependencies["paths"].turn_left = []

        result = connector._execute_turn(10.0)

        assert result is False

    def test_execute_turn_left_success(self, connector, mock_dependencies):
        """Test successful turn left."""
        mock_dependencies["paths"].turn_left = [2, 3]

        result = connector._execute_turn(10.0)

        assert result is True
        mock_dependencies["session"].put.assert_called()

    def test_execute_turn_right_blocked(self, connector, mock_dependencies):
        """Test execute turn right when blocked."""
        mock_dependencies["paths"].turn_right = []

        result = connector._execute_turn(-10.0)

        assert result is False

    def test_execute_turn_right_success(self, connector, mock_dependencies):
        """Test successful turn right."""
        mock_dependencies["paths"].turn_right = [5, 6]

        result = connector._execute_turn(-10.0)

        assert result is True
        mock_dependencies["session"].put.assert_called()


class TestTick:
    """Test tick method."""

    def test_tick_odom_none(self, connector, mock_dependencies):
        """Test tick when odom is None."""
        connector.odom = None

        with patch.object(connector, "sleep") as mock_sleep:
            connector.tick()
            mock_sleep.assert_called_once_with(0.5)

    def test_tick_waiting_for_odom_data(self, connector, mock_dependencies):
        """Test tick when waiting for odom data."""
        mock_dependencies["odom"].position["odom_x"] = 0.0

        with patch.object(connector, "sleep") as mock_sleep:
            connector.tick()
            mock_sleep.assert_called_once_with(0.5)

    def test_tick_robot_sitting(self, connector, mock_dependencies):
        """Test tick when robot is sitting."""
        mock_dependencies["odom"].position["body_attitude"] = RobotState.SITTING

        with patch.object(connector, "sleep") as mock_sleep:
            connector.tick()
            mock_sleep.assert_called_once_with(0.5)

    def test_tick_no_pending_movements(self, connector, mock_dependencies):
        """Test tick with no pending movements."""
        with patch.object(connector, "sleep") as mock_sleep:
            connector.tick()
            mock_sleep.assert_called_once_with(0.1)

    def test_tick_movement_timeout(self, connector, mock_dependencies):
        """Test tick when movement times out."""
        connector.movement_attempts = 20
        connector.pending_movements.put(
            MoveCommand(dx=0.5, yaw=0.0, start_x=0.0, start_y=0.0, turn_complete=False)
        )

        with patch.object(connector, "sleep"):
            connector.tick()

        assert connector.movement_attempts == 0
        assert connector.pending_movements.qsize() == 0

    def test_tick_turn_phase_large_gap(self, connector, mock_dependencies):
        """Test tick during turn phase with large gap."""
        mock_dependencies["odom"].position["odom_yaw_m180_p180"] = 0.0
        connector.pending_movements.put(
            MoveCommand(dx=0.5, yaw=45.0, start_x=0.0, start_y=0.0, turn_complete=False)
        )

        with patch.object(connector, "_execute_turn", return_value=True):
            with patch.object(connector, "sleep"):
                connector.tick()

        assert connector.movement_attempts == 1

    def test_tick_turn_phase_small_gap(self, connector, mock_dependencies):
        """Test tick during turn phase with small gap (between 5 and 10 degrees)."""
        # Gap = current - target = -38 - (-45) = 7 degrees (small gap)
        mock_dependencies["odom"].position["odom_yaw_m180_p180"] = -38.0
        connector.pending_movements.put(
            MoveCommand(
                dx=0.5, yaw=-45.0, start_x=1.0, start_y=0.0, turn_complete=False
            )
        )

        with patch.object(connector, "sleep"):
            connector.tick()

        assert connector.movement_attempts == 1
        mock_dependencies["session"].put.assert_called()

    def test_tick_turn_complete(self, connector, mock_dependencies):
        """Test tick when turn is complete."""
        mock_dependencies["odom"].position["odom_yaw_m180_p180"] = -43.0
        connector.pending_movements.put(
            MoveCommand(dx=0.5, yaw=45.0, start_x=0.0, start_y=0.0, turn_complete=False)
        )

        with patch.object(connector, "sleep"):
            connector.tick()

        command = list(connector.pending_movements.queue)[0]
        assert command.turn_complete is True

    def test_tick_movement_phase_no_distance(self, connector, mock_dependencies):
        """Test tick during movement phase with no distance required."""
        connector.pending_movements.put(
            MoveCommand(dx=0, yaw=0.0, start_x=0.0, start_y=0.0, turn_complete=True)
        )

        with patch.object(connector, "sleep"):
            connector.tick()

        assert connector.pending_movements.qsize() == 0

    def test_tick_movement_phase_forward_blocked(self, connector, mock_dependencies):
        """Test tick during movement phase when forward is blocked."""
        mock_dependencies["paths"].advance = []
        mock_dependencies["odom"].position["odom_x"] = 1.0
        mock_dependencies["odom"].position["odom_y"] = 0.0
        connector.pending_movements.put(
            MoveCommand(dx=0.5, yaw=0.0, start_x=1.0, start_y=0.0, turn_complete=True)
        )

        with patch.object(connector, "sleep"):
            connector.tick()

        assert connector.pending_movements.qsize() == 0

    def test_tick_movement_phase_retreat_blocked(self, connector, mock_dependencies):
        """Test tick during movement phase when retreat is blocked."""
        mock_dependencies["paths"].retreat = []
        mock_dependencies["odom"].position["odom_x"] = 1.0
        mock_dependencies["odom"].position["odom_y"] = 0.0
        connector.pending_movements.put(
            MoveCommand(dx=-0.5, yaw=0.0, start_x=1.0, start_y=0.0, turn_complete=True)
        )

        with patch.object(connector, "sleep"):
            connector.tick()

        assert connector.pending_movements.qsize() == 0

    def test_tick_movement_phase_continue_moving(self, connector, mock_dependencies):
        """Test tick during movement phase continuing to move."""
        mock_dependencies["odom"].position["odom_x"] = 0.2
        mock_dependencies["odom"].position["odom_y"] = 0.0
        connector.pending_movements.put(
            MoveCommand(dx=0.5, yaw=0.0, start_x=0.0, start_y=0.0, turn_complete=True)
        )

        with patch.object(connector, "sleep"):
            connector.tick()

        assert connector.movement_attempts == 1
        mock_dependencies["session"].put.assert_called()

    def test_tick_movement_phase_complete(self, connector, mock_dependencies):
        """Test tick when movement is complete."""
        mock_dependencies["odom"].position["odom_x"] = 0.5
        mock_dependencies["odom"].position["odom_y"] = 0.0
        connector.pending_movements.put(
            MoveCommand(
                dx=0.5, yaw=0.0, start_x=0.0, start_y=0.0, turn_complete=True, speed=0.5
            )
        )

        with patch.object(connector, "sleep"):
            connector.tick()

        assert connector.pending_movements.qsize() == 0
        assert connector.movement_attempts == 0
