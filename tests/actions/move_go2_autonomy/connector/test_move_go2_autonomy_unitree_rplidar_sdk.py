from queue import Queue
from unittest.mock import Mock, patch

import pytest

from actions.base import MoveCommand
from actions.move_go2_autonomy.connector.unitree_rplidar_sdk import (
    MoveUnitreeRPLidarSDKConfig,
    MoveUnitreeRPLidarSDKConnector,
)
from actions.move_go2_autonomy.interface import MoveInput, MovementAction
from providers.unitree_go2_odom_provider import RobotState


@pytest.fixture
def mock_dependencies():
    """Mock all external dependencies."""
    with (
        patch(
            "actions.move_go2_autonomy.connector.unitree_rplidar_sdk.UnitreeGo2RPLidarProvider"
        ) as mock_lidar,
        patch(
            "actions.move_go2_autonomy.connector.unitree_rplidar_sdk.UnitreeGo2StateProvider"
        ) as mock_state,
        patch(
            "actions.move_go2_autonomy.connector.unitree_rplidar_sdk.SportClient"
        ) as mock_sport,
        patch(
            "actions.move_go2_autonomy.connector.unitree_rplidar_sdk.UnitreeGo2OdomProvider"
        ) as mock_odom,
    ):
        # Setup mock instances
        mock_lidar_instance = Mock()
        mock_lidar_instance.advance = [4]
        mock_lidar_instance.retreat = [1]
        mock_lidar_instance.turn_left = [2]
        mock_lidar_instance.turn_right = [6]
        mock_lidar_instance.path_angles = {1: 0, 2: 45, 4: 0, 6: -45}
        mock_lidar.return_value = mock_lidar_instance

        mock_state_instance = Mock()
        mock_state_instance.state_code = None
        mock_state_instance.state = "standing"
        mock_state_instance.action_progress = 0
        mock_state.return_value = mock_state_instance

        mock_sport_instance = Mock()
        mock_sport_instance.SetTimeout = Mock()
        mock_sport_instance.Init = Mock()
        mock_sport_instance.StopMove = Mock()
        mock_sport_instance.Move = Mock()
        mock_sport_instance.BalanceStand = Mock()
        mock_sport.return_value = mock_sport_instance

        mock_odom_instance = Mock()
        mock_odom_instance.position = {
            "moving": False,
            "odom_x": 1.0,
            "odom_y": 0.0,
            "odom_yaw_m180_p180": 0.0,
            "body_attitude": RobotState.STANDING,
        }
        mock_odom.return_value = mock_odom_instance

        yield {
            "lidar": mock_lidar_instance,
            "state": mock_state_instance,
            "sport": mock_sport_instance,
            "odom": mock_odom_instance,
        }


@pytest.fixture
def connector(mock_dependencies):
    """Create a MoveUnitreeRPLidarSDKConnector instance with mocked dependencies."""
    config = MoveUnitreeRPLidarSDKConfig(unitree_ethernet="eth0")
    connector = MoveUnitreeRPLidarSDKConnector(config)
    return connector


class TestMoveUnitreeRPLidarSDKConfig:
    """Test MoveUnitreeRPLidarSDKConfig configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MoveUnitreeRPLidarSDKConfig()
        assert config.unitree_ethernet == "eth0"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = MoveUnitreeRPLidarSDKConfig(unitree_ethernet="eth1")
        assert config.unitree_ethernet == "eth1"


class TestMoveUnitreeRPLidarSDKConnectorInit:
    """Test MoveUnitreeRPLidarSDKConnector initialization."""

    def test_initialization(self, connector, mock_dependencies):
        """Test successful initialization."""
        assert connector.dog_attitude is None
        assert connector.turn_speed == 0.8
        assert connector.angle_tolerance == 5.0
        assert connector.distance_tolerance == 0.05
        assert isinstance(connector.pending_movements, Queue)
        assert connector.movement_attempts == 0
        assert connector.movement_attempt_limit == 15
        assert connector.gap_previous == 0

        # Verify providers are initialized
        assert connector.lidar == mock_dependencies["lidar"]
        assert connector.unitree_go2_state == mock_dependencies["state"]
        assert connector.sport_client == mock_dependencies["sport"]
        assert connector.odom == mock_dependencies["odom"]

        # Verify sport client initialization
        mock_dependencies["sport"].SetTimeout.assert_called_once_with(10.0)
        mock_dependencies["sport"].Init.assert_called_once()
        mock_dependencies["sport"].StopMove.assert_called_once()
        mock_dependencies["sport"].Move.assert_called_once_with(0.05, 0, 0)

    def test_initialization_sport_client_error(self):
        """Test initialization when sport client fails."""
        with (
            patch(
                "actions.move_go2_autonomy.connector.unitree_rplidar_sdk.UnitreeGo2RPLidarProvider"
            ),
            patch(
                "actions.move_go2_autonomy.connector.unitree_rplidar_sdk.UnitreeGo2StateProvider"
            ),
            patch(
                "actions.move_go2_autonomy.connector.unitree_rplidar_sdk.SportClient"
            ) as mock_sport,
            patch(
                "actions.move_go2_autonomy.connector.unitree_rplidar_sdk.UnitreeGo2OdomProvider"
            ),
            patch(
                "actions.move_go2_autonomy.connector.unitree_rplidar_sdk.logging"
            ) as mock_logging,
        ):
            mock_sport.side_effect = Exception("Connection failed")

            config = MoveUnitreeRPLidarSDKConfig()
            connector = MoveUnitreeRPLidarSDKConnector(config)

            assert connector.sport_client is None
            mock_logging.error.assert_called()


class TestConnect:
    """Test the connect method."""

    @pytest.mark.asyncio
    async def test_connect_with_joint_lock_state(self, connector, mock_dependencies):
        """Test connect when robot is in jointLock state."""
        mock_dependencies["state"].state_code = 1002
        move_input = MoveInput(action=MovementAction.MOVE_FORWARDS)

        await connector.connect(move_input)

        mock_dependencies["sport"].BalanceStand.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_with_action_in_progress(self, connector, mock_dependencies):
        """Test connect when action is in progress."""
        mock_dependencies["state"].action_progress = 50
        move_input = MoveInput(action=MovementAction.MOVE_FORWARDS)

        await connector.connect(move_input)

        # Should return early without processing
        assert connector.pending_movements.qsize() == 0

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
        assert command.speed == 0.3

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
        mock_dependencies["lidar"].turn_left = []

        connector._process_turn_left()

        assert connector.pending_movements.qsize() == 0

    def test_process_turn_left_success(self, connector, mock_dependencies):
        """Test successful turn left."""
        connector._process_turn_left()

        assert connector.pending_movements.qsize() == 1

    def test_process_turn_right_with_barrier(self, connector, mock_dependencies):
        """Test turn right when blocked by barrier."""
        mock_dependencies["lidar"].turn_right = []

        connector._process_turn_right()

        assert connector.pending_movements.qsize() == 0

    def test_process_turn_right_success(self, connector, mock_dependencies):
        """Test successful turn right."""
        connector._process_turn_right()

        assert connector.pending_movements.qsize() == 1

    def test_process_move_forward_with_barrier(self, connector, mock_dependencies):
        """Test move forward when blocked by barrier."""
        mock_dependencies["lidar"].advance = []

        connector._process_move_forward()

        assert connector.pending_movements.qsize() == 0

    def test_process_move_forward_success(self, connector, mock_dependencies):
        """Test successful move forward."""
        connector._process_move_forward()

        assert connector.pending_movements.qsize() == 1

    def test_process_move_back_with_barrier(self, connector, mock_dependencies):
        """Test move back when blocked by barrier."""
        mock_dependencies["lidar"].retreat = []

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

    def test_move_robot_no_sport_client(self, connector, mock_dependencies):
        """Test move robot when sport client is None."""
        connector.sport_client = None

        connector._move_robot(0.5, 0.0, 0.0)

        # Should not crash

    def test_move_robot_not_standing(self, connector, mock_dependencies):
        """Test move robot when robot is not standing."""
        mock_dependencies["odom"].position["body_attitude"] = RobotState.SITTING
        mock_dependencies["sport"].Move.reset_mock()

        connector._move_robot(0.5, 0.0, 0.0)
        mock_dependencies["sport"].Move.assert_not_called()

    def test_move_robot_joint_lock_state(self, connector, mock_dependencies):
        """Test move robot when in joint lock state."""
        mock_dependencies["state"].state = "jointLock"

        connector._move_robot(0.5, 0.0, 0.0)

        mock_dependencies["sport"].BalanceStand.assert_called_once()
        mock_dependencies["sport"].Move.assert_called_with(0.5, 0.0, 0.0)

    def test_move_robot_success(self, connector, mock_dependencies):
        """Test successful robot movement."""
        connector._move_robot(0.5, 0.0, 0.3)

        mock_dependencies["sport"].Move.assert_called_with(0.5, 0.0, 0.3)

    def test_move_robot_error(self, connector, mock_dependencies):
        """Test move robot when exception occurs."""
        mock_dependencies["sport"].Move.side_effect = Exception("Movement failed")

        with patch(
            "actions.move_go2_autonomy.connector.unitree_rplidar_sdk.logging"
        ) as mock_logging:
            connector._move_robot(0.5, 0.0, 0.0)

            mock_logging.error.assert_called()


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
        mock_dependencies["lidar"].turn_left = []

        result = connector._execute_turn(10.0)

        assert result is False

    def test_execute_turn_left_success(self, connector, mock_dependencies):
        """Test successful turn left."""
        mock_dependencies["lidar"].turn_left = [2, 3]

        result = connector._execute_turn(10.0)

        assert result is True
        mock_dependencies["sport"].Move.assert_called()

    def test_execute_turn_right_blocked(self, connector, mock_dependencies):
        """Test execute turn right when blocked."""
        mock_dependencies["lidar"].turn_right = []

        result = connector._execute_turn(-10.0)

        assert result is False

    def test_execute_turn_right_success(self, connector, mock_dependencies):
        """Test successful turn right."""
        mock_dependencies["lidar"].turn_right = [5, 6]

        result = connector._execute_turn(-10.0)

        assert result is True
        mock_dependencies["sport"].Move.assert_called()


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

    def test_tick_dog_sitting(self, connector, mock_dependencies):
        """Test tick when dog is sitting."""
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
        mock_dependencies["sport"].Move.assert_called()

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
        mock_dependencies["lidar"].advance = []
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
        mock_dependencies["lidar"].retreat = []
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
        mock_dependencies["sport"].Move.assert_called()

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
