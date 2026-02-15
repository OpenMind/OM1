from queue import Queue
from unittest.mock import Mock, patch

import pytest

from actions.base import MoveCommand
from actions.move_turtle.connector.zenoh import MoveZenohConfig, MoveZenohConnector
from actions.move_turtle.interface import MoveInput, MovementAction


@pytest.fixture
def mock_dependencies():
    """Mock all external dependencies."""

    with (
        patch(
            "actions.move_turtle.connector.zenoh.open_zenoh_session"
        ) as mock_open_session,
        patch(
            "actions.move_turtle.connector.zenoh.TurtleBot4RPLidarProvider"
        ) as mock_lidar,
        patch(
            "actions.move_turtle.connector.zenoh.TurtleBot4OdomProvider"
        ) as mock_odom,
    ):
        mock_session = Mock()
        mock_open_session.return_value = mock_session

        mock_lidar_instance = Mock()
        mock_lidar_instance.valid_paths = [4, 9]
        mock_lidar_instance.turn_left = [2, 3]
        mock_lidar_instance.turn_right = [5, 6]
        mock_lidar.return_value = mock_lidar_instance

        mock_odom_instance = Mock()
        mock_odom_instance.x = 1.0
        mock_odom_instance.y = 0.0
        mock_odom_instance.odom_yaw_m180_p180 = 0.0
        mock_odom_instance.position = {"odom_yaw_m180_p180": 0.0}
        mock_odom.return_value = mock_odom_instance

        yield {
            "session": mock_session,
            "lidar": mock_lidar_instance,
            "odom": mock_odom_instance,
        }


@pytest.fixture
def connector(mock_dependencies):
    """Create MoveZenohConnector with mocked dependencies."""
    config = MoveZenohConfig(URID="test_robot")
    return MoveZenohConnector(config)


class TestMoveZenohConfig:
    """Test MoveZenohConfig configuration."""

    def test_default_config(self):
        config = MoveZenohConfig()
        assert config.URID is None

    def test_custom_config(self):
        config = MoveZenohConfig(URID="my_robot")
        assert config.URID == "my_robot"


class TestMoveZenohConnectorInit:
    """Test MoveZenohConnector initialization."""

    def test_init_with_urid(self, connector, mock_dependencies):
        assert connector.turn_speed == 0.8
        assert connector.angle_tolerance == 5.0
        assert connector.distance_tolerance == 0.05
        assert isinstance(connector.pending_movements, Queue)
        assert connector.session == mock_dependencies["session"]
        assert connector.cmd_vel == "test_robot/c3/cmd_vel"

    def test_init_without_urid(self):
        """Test initialization without URID aborts."""
        with (
            patch("actions.move_turtle.connector.zenoh.open_zenoh_session"),
            patch("actions.move_turtle.connector.zenoh.TurtleBot4RPLidarProvider"),
            patch("actions.move_turtle.connector.zenoh.TurtleBot4OdomProvider"),
            patch("actions.move_turtle.connector.zenoh.logging") as mock_logging,
        ):
            config = MoveZenohConfig(URID=None)
            connector = MoveZenohConnector(config)
            mock_logging.warning.assert_called_with(
                "Aborting TurtleBot4 Move system, no URID provided"
            )
            assert connector.session is None

    def test_init_zenoh_error(self):
        """Test initialization when Zenoh fails."""
        with (
            patch(
                "actions.move_turtle.connector.zenoh.open_zenoh_session"
            ) as mock_session,
            patch("actions.move_turtle.connector.zenoh.TurtleBot4RPLidarProvider"),
            patch("actions.move_turtle.connector.zenoh.TurtleBot4OdomProvider"),
            patch("actions.move_turtle.connector.zenoh.logging") as mock_logging,
        ):
            mock_session.side_effect = Exception("Connection failed")
            config = MoveZenohConfig(URID="test")
            connector = MoveZenohConnector(config)
            assert connector.session is None
            mock_logging.error.assert_called()


class TestMoveZenohConnectorConnect:
    """Test connect method."""

    @pytest.mark.asyncio
    async def test_connect_stand_still(self, connector, mock_dependencies):
        """Test stand still does nothing."""
        move_input = MoveInput(action=MovementAction.STAND_STILL)
        await connector.connect(move_input)
        assert connector.pending_movements.qsize() == 0

    @pytest.mark.asyncio
    async def test_connect_turn_left(self, connector, mock_dependencies):
        """Test turn left adds pending movement."""
        move_input = MoveInput(action=MovementAction.TURN_LEFT)
        await connector.connect(move_input)
        assert connector.pending_movements.qsize() == 1
        cmd = connector.pending_movements.get()
        assert cmd.dx == 0.0

    @pytest.mark.asyncio
    async def test_connect_turn_right(self, connector, mock_dependencies):
        """Test turn right adds pending movement."""
        move_input = MoveInput(action=MovementAction.TURN_RIGHT)
        await connector.connect(move_input)
        assert connector.pending_movements.qsize() == 1

    @pytest.mark.asyncio
    async def test_connect_move_forwards(self, connector, mock_dependencies):
        """Test move forwards adds pending movement."""
        move_input = MoveInput(action=MovementAction.MOVE_FORWARDS)
        await connector.connect(move_input)
        assert connector.pending_movements.qsize() == 1
        cmd = connector.pending_movements.get()
        assert cmd.dx == 0.5

    @pytest.mark.asyncio
    async def test_connect_move_forwards_blocked(self, connector, mock_dependencies):
        """Test move forwards when path is blocked."""
        mock_dependencies["lidar"].valid_paths = []
        move_input = MoveInput(action=MovementAction.MOVE_FORWARDS)
        await connector.connect(move_input)
        assert connector.pending_movements.qsize() == 0

    @pytest.mark.asyncio
    async def test_connect_pending_movement_queued(self, connector, mock_dependencies):
        """Test connect when movement is already pending."""
        connector.pending_movements.put(
            MoveCommand(dx=0.5, yaw=0.0, start_x=0.0, start_y=0.0)
        )
        move_input = MoveInput(action=MovementAction.MOVE_FORWARDS)
        await connector.connect(move_input)
        assert connector.pending_movements.qsize() == 1

    @pytest.mark.asyncio
    async def test_connect_emergency_active(self, connector, mock_dependencies):
        """Test connect when emergency is active."""
        connector.emergency = 90.0
        move_input = MoveInput(action=MovementAction.MOVE_FORWARDS)
        await connector.connect(move_input)
        assert connector.pending_movements.qsize() == 0

    @pytest.mark.asyncio
    async def test_connect_waiting_odom(self, connector, mock_dependencies):
        """Test connect when waiting for odom data."""
        mock_dependencies["odom"].x = 0.0
        move_input = MoveInput(action=MovementAction.MOVE_FORWARDS)
        await connector.connect(move_input)
        assert connector.pending_movements.qsize() == 0


class TestMoveZenohConnectorMove:
    """Test move method."""

    def test_move_no_session(self, connector, mock_dependencies):
        """Test move when session is None."""
        connector.session = None
        connector.move(0.5, 0.0)

    def test_move_success(self, connector, mock_dependencies):
        """Test successful move publishes Zenoh message."""
        connector.move(0.5, 0.3)
        mock_dependencies["session"].put.assert_called_once()


class TestMoveZenohConnectorAngle:
    """Test angle calculation methods."""

    def test_calculate_angle_gap_simple(self, connector, mock_dependencies):
        result = connector._calculate_angle_gap(10.0, 5.0)
        assert result == 5.0

    def test_calculate_angle_gap_wrap_positive(self, connector, mock_dependencies):
        result = connector._calculate_angle_gap(170.0, -170.0)
        assert result == -20.0

    def test_calculate_angle_gap_wrap_negative(self, connector, mock_dependencies):
        result = connector._calculate_angle_gap(-170.0, 170.0)
        assert result == 20.0


class TestMoveZenohConnectorTick:
    """Test tick method."""

    def test_tick_waiting_for_odom(self, connector, mock_dependencies):
        """Test tick when waiting for odom data."""
        mock_dependencies["odom"].x = 0.0
        with patch.object(connector, "sleep"):
            connector.tick()

    def test_tick_no_pending(self, connector, mock_dependencies):
        """Test tick with no pending movements."""
        with patch.object(connector, "sleep"):
            connector.tick()


class TestMoveZenohConnectorCleanAbort:
    """Test clean_abort method."""

    def test_clean_abort(self, connector, mock_dependencies):
        connector.movement_attempts = 5
        connector.pending_movements.put(
            MoveCommand(dx=0.5, yaw=0.0, start_x=0.0, start_y=0.0)
        )
        connector.clean_abort()
        assert connector.movement_attempts == 0
        assert connector.pending_movements.qsize() == 0

    def test_clean_abort_empty_queue(self, connector, mock_dependencies):
        connector.movement_attempts = 3
        connector.clean_abort()
        assert connector.movement_attempts == 0


class TestExecuteTurn:
    """Test _execute_turn method."""

    def test_execute_turn_left_blocked(self, connector, mock_dependencies):
        mock_dependencies["lidar"].turn_left = []
        result = connector._execute_turn(10.0)
        assert result is False

    def test_execute_turn_left_success(self, connector, mock_dependencies):
        result = connector._execute_turn(10.0)
        assert result is True

    def test_execute_turn_right_blocked(self, connector, mock_dependencies):
        mock_dependencies["lidar"].turn_right = []
        result = connector._execute_turn(-10.0)
        assert result is False

    def test_execute_turn_right_success(self, connector, mock_dependencies):
        result = connector._execute_turn(-10.0)
        assert result is True
