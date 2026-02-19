import sys
from queue import Queue
from unittest.mock import MagicMock, Mock, patch

import pytest

# Mock unitree SDK modules before any imports to allow CI to run without hardware
sys.modules["unitree"] = MagicMock()
sys.modules["unitree.unitree_sdk2py"] = MagicMock()
sys.modules["unitree.unitree_sdk2py.go2"] = MagicMock()
sys.modules["unitree.unitree_sdk2py.go2.sport"] = MagicMock()
sys.modules["unitree.unitree_sdk2py.go2.sport.sport_client"] = MagicMock()

from actions.base import MoveCommand  # noqa: E402
from actions.move_go2_autonomy.connector.unitree_om_path_sdk import (  # noqa: E402
    MoveUnitreeOMPathSDKConfig,
    MoveUnitreeOMPathSDKConnector,
)
from actions.move_go2_autonomy.interface import MoveInput, MovementAction  # noqa: E402
from providers.unitree_go2_odom_provider import RobotState  # noqa: E402


def _make_zenoh_sample(payload: bytes = b"") -> Mock:
    """Create a minimal mock Zenoh sample."""
    sample = Mock()
    sample.payload.to_bytes.return_value = payload
    return sample


def _make_ai_status_mock(code: int, request_id: str = "req-1", frame_id: str = "map"):
    """Return (mock_request_cls, mock_response_cls) pre-configured for an AI status call."""
    mock_header = Mock()
    mock_header.frame_id = frame_id

    mock_req_instance = Mock()
    mock_req_instance.code = code
    mock_req_instance.request_id = request_id
    mock_req_instance.header = mock_header

    mock_req_cls = Mock()
    mock_req_cls.deserialize.return_value = mock_req_instance

    mock_resp_instance = Mock()
    mock_resp_instance.serialize.return_value = b"response"

    mock_resp_cls = Mock()
    mock_resp_cls.return_value = mock_resp_instance

    return mock_req_cls, mock_resp_cls


@pytest.fixture
def mock_dependencies():
    """Mock all external dependencies for the connector."""
    with (
        patch(
            "actions.move_go2_autonomy.connector.unitree_om_path_sdk.SimplePathsProvider"
        ) as mock_paths,
        patch(
            "actions.move_go2_autonomy.connector.unitree_om_path_sdk.UnitreeGo2StateProvider"
        ) as mock_state,
        patch(
            "actions.move_go2_autonomy.connector.unitree_om_path_sdk.SportClient"
        ) as mock_sport,
        patch(
            "actions.move_go2_autonomy.connector.unitree_om_path_sdk.UnitreeGo2OdomProvider"
        ) as mock_odom,
        patch(
            "actions.move_go2_autonomy.connector.unitree_om_path_sdk.FacePresenceProvider"
        ) as mock_face,
        patch(
            "actions.move_go2_autonomy.connector.unitree_om_path_sdk.open_zenoh_session"
        ) as mock_zenoh,
    ):
        mock_paths_instance = Mock()
        mock_paths_instance.advance = [4]
        mock_paths_instance.retreat = [1]
        mock_paths_instance.turn_left = [2]
        mock_paths_instance.turn_right = [6]
        mock_paths_instance.path_angles = {1: 0, 2: 45, 4: 0, 6: -45}
        mock_paths.return_value = mock_paths_instance

        mock_state_instance = Mock()
        mock_state_instance.state_code = None
        mock_state_instance.state = "standing"
        mock_state_instance.action_progress = 0
        mock_state.return_value = mock_state_instance

        mock_sport_instance = Mock()
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

        mock_face_instance = Mock()
        mock_face_instance.unknown_faces = 0
        mock_face.return_value = mock_face_instance

        mock_session = Mock()
        mock_session.declare_publisher.return_value = Mock()
        mock_zenoh.return_value = mock_session

        yield {
            "paths": mock_paths_instance,
            "state": mock_state_instance,
            "sport": mock_sport_instance,
            "odom": mock_odom_instance,
            "face": mock_face_instance,
            "session": mock_session,
        }


@pytest.fixture
def connector(mock_dependencies):
    """Connector instance with all dependencies mocked."""
    config = MoveUnitreeOMPathSDKConfig(unitree_ethernet="eth0")
    return MoveUnitreeOMPathSDKConnector(config)


class TestConfig:
    """Test MoveUnitreeOMPathSDKConfig."""

    def test_default_values(self):
        config = MoveUnitreeOMPathSDKConfig()
        assert config.unitree_ethernet == "eth0"
        assert config.mode is None

    def test_custom_values(self):
        config = MoveUnitreeOMPathSDKConfig(unitree_ethernet="eth1", mode="guard")
        assert config.unitree_ethernet == "eth1"
        assert config.mode == "guard"


class TestInit:
    """Test connector initialization."""

    def test_default_state(self, connector, mock_dependencies):
        assert connector.dog_attitude is None
        assert connector.move_speed == 0.5
        assert connector.turn_speed == 0.8
        assert connector.angle_tolerance == 5.0
        assert connector.distance_tolerance == 0.05
        assert isinstance(connector.pending_movements, Queue)
        assert connector.movement_attempts == 0
        assert connector.movement_attempt_limit == 15
        assert connector.gap_previous == 0
        assert connector.ai_control_enabled is True
        assert connector.session is not None

    def test_providers_assigned(self, connector, mock_dependencies):
        assert connector.path_provider == mock_dependencies["paths"]
        assert connector.unitree_go2_state == mock_dependencies["state"]
        assert connector.sport_client == mock_dependencies["sport"]
        assert connector.odom == mock_dependencies["odom"]
        assert connector.face_presence_provider == mock_dependencies["face"]

    def test_sport_client_initialized(self, connector, mock_dependencies):
        sport = mock_dependencies["sport"]
        sport.SetTimeout.assert_called_once_with(10.0)
        sport.Init.assert_called_once()
        sport.StopMove.assert_called_once()
        sport.Move.assert_called_once_with(0.05, 0, 0)

    def test_zenoh_session_initialized(self, connector, mock_dependencies):
        session = mock_dependencies["session"]
        session.declare_subscriber.assert_called_once()
        session.declare_publisher.assert_called_once_with("om/ai/response")

    def test_sport_client_error_sets_none(self, mock_dependencies):
        with patch(
            "actions.move_go2_autonomy.connector.unitree_om_path_sdk.SportClient",
            side_effect=Exception("no hardware"),
        ):
            config = MoveUnitreeOMPathSDKConfig()
            c = MoveUnitreeOMPathSDKConnector(config)
            assert c.sport_client is None

    def test_zenoh_error_sets_session_none(self, mock_dependencies):
        with patch(
            "actions.move_go2_autonomy.connector.unitree_om_path_sdk.open_zenoh_session",
            side_effect=Exception("no zenoh"),
        ):
            config = MoveUnitreeOMPathSDKConfig()
            c = MoveUnitreeOMPathSDKConnector(config)
            assert c.session is None

    def test_missing_ethernet_raises(self, mock_dependencies):
        config = MoveUnitreeOMPathSDKConfig()
        config.unitree_ethernet = None
        with pytest.raises(ValueError):
            MoveUnitreeOMPathSDKConnector(config)


class TestConnect:
    """Test the connect() method."""

    @pytest.mark.asyncio
    async def test_guard_mode_blocks_unknown_face(self, connector, mock_dependencies):
        connector.mode = "guard"
        mock_dependencies["face"].unknown_faces = 1
        await connector.connect(MoveInput(action=MovementAction.MOVE_FORWARDS))
        assert connector.pending_movements.qsize() == 0

    @pytest.mark.asyncio
    async def test_ai_control_disabled_blocks(self, connector, mock_dependencies):
        connector.ai_control_enabled = False
        await connector.connect(MoveInput(action=MovementAction.MOVE_FORWARDS))
        assert connector.pending_movements.qsize() == 0

    @pytest.mark.asyncio
    async def test_joint_lock_triggers_balance_stand(
        self, connector, mock_dependencies
    ):
        mock_dependencies["state"].state_code = 1002
        await connector.connect(MoveInput(action=MovementAction.MOVE_FORWARDS))
        mock_dependencies["sport"].BalanceStand.assert_called_once()

    @pytest.mark.asyncio
    async def test_action_in_progress_blocks(self, connector, mock_dependencies):
        mock_dependencies["state"].action_progress = 50
        await connector.connect(MoveInput(action=MovementAction.MOVE_FORWARDS))
        assert connector.pending_movements.qsize() == 0

    @pytest.mark.asyncio
    async def test_robot_already_moving_blocks(self, connector, mock_dependencies):
        mock_dependencies["state"].state_code = None
        mock_dependencies["odom"].position["moving"] = True
        await connector.connect(MoveInput(action=MovementAction.MOVE_FORWARDS))
        assert connector.pending_movements.qsize() == 0

    @pytest.mark.asyncio
    async def test_pending_movement_blocks(self, connector, mock_dependencies):
        connector.pending_movements.put(
            MoveCommand(dx=0.5, yaw=0.0, start_x=0.0, start_y=0.0, turn_complete=False)
        )
        await connector.connect(MoveInput(action=MovementAction.MOVE_FORWARDS))
        assert connector.pending_movements.qsize() == 1

    @pytest.mark.asyncio
    async def test_zero_odom_x_blocks(self, connector, mock_dependencies):
        mock_dependencies["odom"].position["odom_x"] = 0.0
        await connector.connect(MoveInput(action=MovementAction.MOVE_FORWARDS))
        assert connector.pending_movements.qsize() == 0

    @pytest.mark.asyncio
    async def test_turn_left(self, connector, mock_dependencies):
        await connector.connect(MoveInput(action=MovementAction.TURN_LEFT))
        assert connector.pending_movements.qsize() == 1
        cmd = connector.pending_movements.get()
        assert cmd.dx == 0.5
        assert cmd.turn_complete is False

    @pytest.mark.asyncio
    async def test_turn_right(self, connector, mock_dependencies):
        await connector.connect(MoveInput(action=MovementAction.TURN_RIGHT))
        assert connector.pending_movements.qsize() == 1
        cmd = connector.pending_movements.get()
        assert cmd.dx == 0.5
        assert cmd.turn_complete is False

    @pytest.mark.asyncio
    async def test_move_forwards(self, connector, mock_dependencies):
        await connector.connect(MoveInput(action=MovementAction.MOVE_FORWARDS))
        assert connector.pending_movements.qsize() == 1
        assert connector.pending_movements.get().dx == 0.5

    @pytest.mark.asyncio
    async def test_move_back(self, connector, mock_dependencies):
        await connector.connect(MoveInput(action=MovementAction.MOVE_BACK))
        assert connector.pending_movements.qsize() == 1
        cmd = connector.pending_movements.get()
        assert cmd.dx == -0.5
        assert cmd.turn_complete is True
        assert cmd.speed == 0.2

    @pytest.mark.asyncio
    async def test_stand_still(self, connector, mock_dependencies):
        await connector.connect(MoveInput(action=MovementAction.STAND_STILL))
        assert connector.pending_movements.qsize() == 0

    @pytest.mark.asyncio
    async def test_unknown_action(self, connector, mock_dependencies):
        await connector.connect(MoveInput(action="fly"))  # type: ignore[arg-type]
        assert connector.pending_movements.qsize() == 0


class TestMovementProcessing:
    """Test _process_* methods."""

    def test_turn_left_blocked(self, connector, mock_dependencies):
        mock_dependencies["paths"].turn_left = []
        connector._process_turn_left()
        assert connector.pending_movements.qsize() == 0

    def test_turn_left_success(self, connector, mock_dependencies):
        connector._process_turn_left()
        assert connector.pending_movements.qsize() == 1

    def test_turn_right_blocked(self, connector, mock_dependencies):
        mock_dependencies["paths"].turn_right = []
        connector._process_turn_right()
        assert connector.pending_movements.qsize() == 0

    def test_turn_right_success(self, connector, mock_dependencies):
        connector._process_turn_right()
        assert connector.pending_movements.qsize() == 1

    def test_move_forward_blocked(self, connector, mock_dependencies):
        mock_dependencies["paths"].advance = []
        connector._process_move_forward()
        assert connector.pending_movements.qsize() == 0

    def test_move_forward_zero_angle(self, connector, mock_dependencies):
        mock_dependencies["paths"].advance = [4]
        mock_dependencies["paths"].path_angles = {4: 0}
        connector._process_move_forward()
        assert connector.pending_movements.get().turn_complete is True

    def test_move_forward_nonzero_angle(self, connector, mock_dependencies):
        mock_dependencies["paths"].advance = [2]
        mock_dependencies["paths"].path_angles = {2: 30}
        connector._process_move_forward()
        assert connector.pending_movements.get().turn_complete is False

    def test_move_back_blocked(self, connector, mock_dependencies):
        mock_dependencies["paths"].retreat = []
        connector._process_move_back()
        assert connector.pending_movements.qsize() == 0

    def test_move_back_success(self, connector, mock_dependencies):
        connector._process_move_back()
        cmd = connector.pending_movements.get()
        assert cmd.dx == -0.5
        assert cmd.turn_complete is True
        assert cmd.speed == 0.2


class TestMoveRobot:
    """Test _move_robot()."""

    def test_no_sport_client(self, connector, mock_dependencies):
        connector.sport_client = None
        connector._move_robot(0.5, 0.0, 0.0)  # should not raise

    def test_not_standing(self, connector, mock_dependencies):
        mock_dependencies["odom"].position["body_attitude"] = RobotState.SITTING
        mock_dependencies["sport"].Move.reset_mock()
        connector._move_robot(0.5, 0.0, 0.0)
        mock_dependencies["sport"].Move.assert_not_called()

    def test_joint_lock_calls_balance_stand(self, connector, mock_dependencies):
        mock_dependencies["state"].state = "jointLock"
        connector._move_robot(0.5, 0.0, 0.0)
        mock_dependencies["sport"].BalanceStand.assert_called_once()
        mock_dependencies["sport"].Move.assert_called_with(0.5, 0.0, 0.0)

    def test_success(self, connector, mock_dependencies):
        connector._move_robot(0.5, 0.0, 0.3)
        mock_dependencies["sport"].Move.assert_called_with(0.5, 0.0, 0.3)

    def test_move_exception_logged(self, connector, mock_dependencies):
        mock_dependencies["sport"].Move.side_effect = Exception("fail")
        with patch(
            "actions.move_go2_autonomy.connector.unitree_om_path_sdk.logging"
        ) as mock_log:
            connector._move_robot(0.5, 0.0, 0.0)
            mock_log.error.assert_called()


class TestCleanAbort:
    """Test clean_abort()."""

    def test_clears_attempts_and_queue(self, connector, mock_dependencies):
        connector.movement_attempts = 5
        connector.pending_movements.put(
            MoveCommand(dx=0.5, yaw=0.0, start_x=0.0, start_y=0.0, turn_complete=False)
        )
        connector.clean_abort()
        assert connector.movement_attempts == 0
        assert connector.pending_movements.qsize() == 0

    def test_empty_queue_no_error(self, connector, mock_dependencies):
        connector.movement_attempts = 3
        connector.clean_abort()
        assert connector.movement_attempts == 0


class TestAngleCalculations:
    """Test _normalize_angle() and _calculate_angle_gap()."""

    def test_normalize_positive_overflow(self, connector, mock_dependencies):
        assert connector._normalize_angle(270.0) == -90.0

    def test_normalize_negative_overflow(self, connector, mock_dependencies):
        assert connector._normalize_angle(-270.0) == 90.0

    def test_normalize_within_range(self, connector, mock_dependencies):
        assert connector._normalize_angle(45.0) == 45.0

    def test_angle_gap_simple(self, connector, mock_dependencies):
        assert connector._calculate_angle_gap(10.0, 5.0) == 5.0

    def test_angle_gap_wrap_positive(self, connector, mock_dependencies):
        assert connector._calculate_angle_gap(170.0, -170.0) == -20.0

    def test_angle_gap_wrap_negative(self, connector, mock_dependencies):
        assert connector._calculate_angle_gap(-170.0, 170.0) == 20.0


class TestExecuteTurn:
    """Test _execute_turn()."""

    def test_left_blocked(self, connector, mock_dependencies):
        mock_dependencies["paths"].turn_left = []
        assert connector._execute_turn(10.0) is False

    def test_left_success(self, connector, mock_dependencies):
        mock_dependencies["paths"].turn_left = [2, 3]
        assert connector._execute_turn(10.0) is True
        mock_dependencies["sport"].Move.assert_called()

    def test_right_blocked(self, connector, mock_dependencies):
        mock_dependencies["paths"].turn_right = []
        assert connector._execute_turn(-10.0) is False

    def test_right_success(self, connector, mock_dependencies):
        mock_dependencies["paths"].turn_right = [5, 6]
        assert connector._execute_turn(-10.0) is True
        mock_dependencies["sport"].Move.assert_called()


class TestTick:
    """Test tick()."""

    def test_odom_none_sleeps(self, connector, mock_dependencies):
        connector.odom = None
        with patch.object(connector, "sleep") as mock_sleep:
            connector.tick()
        mock_sleep.assert_called_once_with(0.5)

    def test_zero_odom_x_sleeps(self, connector, mock_dependencies):
        mock_dependencies["odom"].position["odom_x"] = 0.0
        with patch.object(connector, "sleep") as mock_sleep:
            connector.tick()
        mock_sleep.assert_called_once_with(0.5)

    def test_not_standing_sleeps(self, connector, mock_dependencies):
        mock_dependencies["odom"].position["body_attitude"] = RobotState.SITTING
        with patch.object(connector, "sleep") as mock_sleep:
            connector.tick()
        mock_sleep.assert_called_once_with(0.5)

    def test_no_pending_movements_sleeps(self, connector, mock_dependencies):
        with patch.object(connector, "sleep") as mock_sleep:
            connector.tick()
        mock_sleep.assert_called_once_with(0.1)

    def test_timeout_aborts(self, connector, mock_dependencies):
        connector.movement_attempts = 20
        connector.pending_movements.put(
            MoveCommand(dx=0.5, yaw=0.0, start_x=0.0, start_y=0.0, turn_complete=False)
        )
        with patch.object(connector, "sleep"):
            connector.tick()
        assert connector.movement_attempts == 0
        assert connector.pending_movements.qsize() == 0

    def test_turn_phase_large_gap_success(self, connector, mock_dependencies):
        mock_dependencies["odom"].position["odom_yaw_m180_p180"] = 0.0
        connector.pending_movements.put(
            MoveCommand(dx=0.5, yaw=45.0, start_x=0.0, start_y=0.0, turn_complete=False)
        )
        with (
            patch.object(connector, "_execute_turn", return_value=True),
            patch.object(connector, "sleep"),
        ):
            connector.tick()
        assert connector.movement_attempts == 1

    def test_turn_phase_large_gap_blocked_aborts(self, connector, mock_dependencies):
        mock_dependencies["odom"].position["odom_yaw_m180_p180"] = 0.0
        connector.pending_movements.put(
            MoveCommand(dx=0.5, yaw=45.0, start_x=0.0, start_y=0.0, turn_complete=False)
        )
        with (
            patch.object(connector, "_execute_turn", return_value=False),
            patch.object(connector, "sleep"),
        ):
            connector.tick()
        assert connector.pending_movements.qsize() == 0

    def test_turn_phase_small_gap_positive(self, connector, mock_dependencies):
        # gap = (-1 * -7) - 0 = 7 degrees → small positive gap → rotate left slowly
        mock_dependencies["odom"].position["odom_yaw_m180_p180"] = -7.0
        connector.pending_movements.put(
            MoveCommand(dx=0.5, yaw=0.0, start_x=1.0, start_y=0.0, turn_complete=False)
        )
        with patch.object(connector, "sleep"):
            connector.tick()
        assert connector.movement_attempts == 1
        mock_dependencies["sport"].Move.assert_called()

    def test_turn_phase_small_gap_negative(self, connector, mock_dependencies):
        # gap = (-1 * 7) - 0 = -7 degrees → small negative gap → rotate right slowly
        mock_dependencies["odom"].position["odom_yaw_m180_p180"] = 7.0
        connector.pending_movements.put(
            MoveCommand(dx=0.5, yaw=0.0, start_x=1.0, start_y=0.0, turn_complete=False)
        )
        with patch.object(connector, "sleep"):
            connector.tick()
        assert connector.movement_attempts == 1
        mock_dependencies["sport"].Move.assert_called()

    def test_turn_phase_within_tolerance_completes(self, connector, mock_dependencies):
        # gap ~-2 degrees → within tolerance, mark turn complete
        mock_dependencies["odom"].position["odom_yaw_m180_p180"] = -43.0
        connector.pending_movements.put(
            MoveCommand(dx=0.5, yaw=45.0, start_x=0.0, start_y=0.0, turn_complete=False)
        )
        with patch.object(connector, "sleep"):
            connector.tick()
        cmd = list(connector.pending_movements.queue)[0]
        assert cmd.turn_complete is True

    def test_turn_phase_logs_progress_on_subsequent_attempts(
        self, connector, mock_dependencies
    ):
        mock_dependencies["odom"].position["odom_yaw_m180_p180"] = -38.0
        connector.movement_attempts = (
            1  # simulate subsequent attempt to trigger progress log
        )
        connector.pending_movements.put(
            MoveCommand(
                dx=0.5, yaw=-45.0, start_x=1.0, start_y=0.0, turn_complete=False
            )
        )
        with patch.object(connector, "sleep"):
            connector.tick()
        assert connector.movement_attempts == 2

    def test_movement_phase_no_distance_aborts(self, connector, mock_dependencies):
        connector.pending_movements.put(
            MoveCommand(dx=0, yaw=0.0, start_x=0.0, start_y=0.0, turn_complete=True)
        )
        with patch.object(connector, "sleep"):
            connector.tick()
        assert connector.pending_movements.qsize() == 0

    def test_movement_phase_forward_blocked(self, connector, mock_dependencies):
        mock_dependencies["paths"].advance = []
        mock_dependencies["odom"].position["odom_x"] = 1.0
        connector.pending_movements.put(
            MoveCommand(dx=0.5, yaw=0.0, start_x=1.0, start_y=0.0, turn_complete=True)
        )
        with patch.object(connector, "sleep"):
            connector.tick()
        assert connector.pending_movements.qsize() == 0

    def test_movement_phase_retreat_blocked(self, connector, mock_dependencies):
        mock_dependencies["paths"].retreat = []
        mock_dependencies["odom"].position["odom_x"] = 1.0
        connector.pending_movements.put(
            MoveCommand(dx=-0.5, yaw=0.0, start_x=1.0, start_y=0.0, turn_complete=True)
        )
        with patch.object(connector, "sleep"):
            connector.tick()
        assert connector.pending_movements.qsize() == 0

    def test_movement_phase_continue_forward(self, connector, mock_dependencies):
        mock_dependencies["odom"].position["odom_x"] = 0.2
        connector.pending_movements.put(
            MoveCommand(dx=0.5, yaw=0.0, start_x=0.0, start_y=0.0, turn_complete=True)
        )
        with patch.object(connector, "sleep"):
            connector.tick()
        assert connector.movement_attempts == 1
        mock_dependencies["sport"].Move.assert_called()

    def test_movement_phase_overshoot(self, connector, mock_dependencies):
        mock_dependencies["odom"].position["odom_x"] = 0.7
        connector.pending_movements.put(
            MoveCommand(dx=0.5, yaw=0.0, start_x=0.0, start_y=0.0, turn_complete=True)
        )
        with patch.object(connector, "sleep"):
            connector.tick()
        assert connector.movement_attempts == 1
        mock_dependencies["sport"].Move.assert_called()

    def test_movement_phase_complete(self, connector, mock_dependencies):
        mock_dependencies["odom"].position["odom_x"] = 0.5
        connector.pending_movements.put(
            MoveCommand(
                dx=0.5, yaw=0.0, start_x=0.0, start_y=0.0, turn_complete=True, speed=0.5
            )
        )
        with patch.object(connector, "sleep"):
            connector.tick()
        assert connector.pending_movements.qsize() == 0
        assert connector.movement_attempts == 0

    def test_movement_phase_continue_retreat(self, connector, mock_dependencies):
        mock_dependencies["odom"].position["odom_x"] = 0.2
        connector.pending_movements.put(
            MoveCommand(
                dx=-0.5,
                yaw=0.0,
                start_x=0.0,
                start_y=0.0,
                turn_complete=True,
                speed=0.2,
            )
        )
        with patch.object(connector, "sleep"):
            connector.tick()
        assert connector.movement_attempts == 1
        mock_dependencies["sport"].Move.assert_called()

    def test_movement_phase_logs_progress_on_subsequent_attempts(
        self, connector, mock_dependencies
    ):
        mock_dependencies["odom"].position["odom_x"] = 0.2
        connector.movement_attempts = (
            1  # simulate subsequent attempt to trigger progress log
        )
        connector.pending_movements.put(
            MoveCommand(dx=0.5, yaw=0.0, start_x=0.0, start_y=0.0, turn_complete=True)
        )
        with patch.object(connector, "sleep"):
            connector.tick()
        assert connector.movement_attempts == 2


class TestZenohAIStatus:
    """Test _zenoh_ai_status_request()."""

    def _call(self, connector, code: int):
        """Helper to call _zenoh_ai_status_request with a given code."""
        mock_req_cls, mock_resp_cls = _make_ai_status_mock(code)
        sample = _make_zenoh_sample()
        with (
            patch(
                "actions.move_go2_autonomy.connector.unitree_om_path_sdk.AIStatusRequest",
                mock_req_cls,
            ),
            patch(
                "actions.move_go2_autonomy.connector.unitree_om_path_sdk.AIStatusResponse",
                mock_resp_cls,
            ),
        ):
            connector._zenoh_ai_status_request(sample)

    def test_enable_ai_control(self, connector, mock_dependencies):
        connector.ai_control_enabled = False
        self._call(connector, code=1)
        assert connector.ai_control_enabled is True
        connector._zenoh_ai_status_response_pub.put.assert_called_once()

    def test_disable_ai_control(self, connector, mock_dependencies):
        connector.ai_control_enabled = True
        self._call(connector, code=0)
        assert connector.ai_control_enabled is False
        connector._zenoh_ai_status_response_pub.put.assert_called_once()

    def test_read_status(self, connector, mock_dependencies):
        connector.ai_control_enabled = True
        self._call(connector, code=2)
        connector._zenoh_ai_status_response_pub.put.assert_called_once()
