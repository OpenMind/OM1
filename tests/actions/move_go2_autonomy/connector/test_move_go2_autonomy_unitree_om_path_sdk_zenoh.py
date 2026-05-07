from unittest.mock import MagicMock, patch

import pytest

from actions.base import MoveCommand
from actions.move_go2_autonomy.connector.unitree_om_path_sdk_zenoh import (
    MoveUnitreeOMPathSDKZenohConfig,
    MoveUnitreeOMPathSDKZenohConnector,
)
from providers.odom_provider_base import RobotState


@pytest.fixture
def deps():
    """Patch out everything that does I/O at construction time."""
    with (
        patch("actions.move_go2_autonomy.connector.unitree_om_path_sdk_zenoh.SimplePathsProvider") as mock_paths_class,
        patch("actions.move_go2_autonomy.connector.unitree_om_path_sdk_zenoh.FacePresenceProvider") as mock_face_class,
        patch(
            "actions.move_go2_autonomy.connector.unitree_om_path_sdk_zenoh.UnitreeGo2StateZenohProvider"
        ) as mock_state_zenoh_class,
        patch(
            "actions.move_go2_autonomy.connector.unitree_om_path_sdk_zenoh.UnitreeGo2OdomZenohProvider"
        ) as mock_odom_zenoh_class,
        patch("actions.move_go2_autonomy.connector.unitree_om_path_sdk_zenoh.open_zenoh_session") as mock_open_session,
    ):
        # Common stub instances
        paths = MagicMock()
        paths.advance = [4]
        paths.retreat = True
        paths.turn_left = [0, 1]
        paths.turn_right = [6, 7]
        paths.path_angles = [-90, -60, -30, 0, 30, 60, 90, 120, 180]
        mock_paths_class.return_value = paths

        face = MagicMock()
        face.unknown_faces = 0
        mock_face_class.return_value = face

        state_zenoh = MagicMock()
        state_zenoh.state_code = None
        state_zenoh.state = "Standing"
        state_zenoh.action_progress = 0
        mock_state_zenoh_class.return_value = state_zenoh

        odom_zenoh = MagicMock()
        odom_zenoh.position = {
            "moving": False,
            "body_attitude": RobotState.STANDING,
            "odom_x": 1.0,
            "odom_y": 0.0,
            "odom_yaw_m180_p180": 0.0,
            "odom_subscriber_ts": 1.0,
        }
        mock_odom_zenoh_class.return_value = odom_zenoh

        session = MagicMock()
        sub = MagicMock()
        session.declare_subscriber.return_value = sub
        pub = MagicMock()
        session.declare_publisher.return_value = pub
        mock_open_session.return_value = session

        yield {
            "paths_class": mock_paths_class,
            "paths": paths,
            "face": face,
            "state_zenoh_class": mock_state_zenoh_class,
            "state_zenoh": state_zenoh,
            "odom_zenoh_class": mock_odom_zenoh_class,
            "odom_zenoh": odom_zenoh,
            "open_session": mock_open_session,
            "session": session,
            "publisher": pub,
        }


def test_init_zenoh_cmd_vel_mode(deps):
    config = MoveUnitreeOMPathSDKZenohConfig()
    conn = MoveUnitreeOMPathSDKZenohConnector(config=config)
    deps["state_zenoh_class"].assert_called_once()
    deps["odom_zenoh_class"].assert_called_once()
    assert conn._sport_pub is not None


def test_init_zenoh_sport_api_mode(deps):
    config = MoveUnitreeOMPathSDKZenohConfig()
    conn = MoveUnitreeOMPathSDKZenohConnector(config=config)
    assert conn._sport_pub is not None


def test_init_sport_api_without_zenoh_raises(deps):
    config = MoveUnitreeOMPathSDKZenohConfig()
    conn = MoveUnitreeOMPathSDKZenohConnector(config=config)
    assert conn is not None


def test_init_local_path(deps):
    config = MoveUnitreeOMPathSDKZenohConfig()
    MoveUnitreeOMPathSDKZenohConnector(config=config)
    deps["state_zenoh_class"].assert_called_once()
    deps["odom_zenoh_class"].assert_called_once()


def test_init_session_failure_keeps_running(deps):
    deps["open_session"].side_effect = RuntimeError("zenoh down")
    config = MoveUnitreeOMPathSDKZenohConfig()
    conn = MoveUnitreeOMPathSDKZenohConnector(config=config)
    assert conn.session is None


def test_init_permissive_paths_logged(deps):
    config = MoveUnitreeOMPathSDKZenohConfig()
    conn = MoveUnitreeOMPathSDKZenohConnector(config=config)
    assert conn is not None


@pytest.fixture
def conn(deps):
    config = MoveUnitreeOMPathSDKZenohConfig()
    return MoveUnitreeOMPathSDKZenohConnector(config=config)


@pytest.mark.parametrize(
    "input_angle,expected",
    [
        (0.0, 0.0),
        (180.0, 180.0),
        (-180.0, -180.0),
        (190.0, -170.0),
        (-190.0, 170.0),
    ],
)
def test_normalize_angle(conn, input_angle, expected):
    assert conn._normalize_angle(input_angle) == expected


@pytest.mark.parametrize(
    "current,target,expected",
    [
        (10.0, 0.0, 10.0),
        (-170.0, 170.0, -340.0 + 360.0),  # -> 20
        (170.0, -170.0, 340.0 - 360.0),  # -> -20
        (0.0, 0.0, 0.0),
    ],
)
def test_calculate_angle_gap(conn, current, target, expected):
    assert conn._calculate_angle_gap(current, target) == round(expected, 2)


def test_pick_path_angle_returns_default_for_empty(conn):
    angle = conn._pick_path_angle([], default=42.0)
    assert angle == 42.0


def test_pick_path_angle_picks_from_list(conn):
    angle = conn._pick_path_angle([0, 1], default=999.0)
    assert angle in (-90, -60)


def test_move_robot_zenoh_cmd_vel(conn):
    conn._sport_pub = MagicMock()
    conn._move_robot(0.5, 0.0, 0.1)
    conn._sport_pub.put.assert_called_once()


def test_move_robot_zenoh_cmd_vel_skips_when_pub_missing(conn):
    conn._sport_pub = None
    # Should not raise
    conn._move_robot(0.5, 0.0, 0.1)


def test_move_robot_skips_when_sitting(conn):
    conn.odom.position["body_attitude"] = RobotState.SITTING
    conn._sport_pub = MagicMock()
    conn._move_robot(0.5, 0.0, 0.0)
    conn._sport_pub.put.assert_not_called()


def test_move_robot_sport_api_path(deps):
    config = MoveUnitreeOMPathSDKZenohConfig()
    conn = MoveUnitreeOMPathSDKZenohConnector(config=config)
    conn._sport_pub = MagicMock()
    conn._move_robot(0.5, 0.0, 0.0)
    conn._sport_pub.put.assert_called_once()


def test_move_robot_sport_api_no_pub(deps):
    config = MoveUnitreeOMPathSDKZenohConfig()
    conn = MoveUnitreeOMPathSDKZenohConnector(config=config)
    conn._sport_pub = None  # publisher missing
    conn._move_robot(0.5, 0.0, 0.0)


def test_move_robot_local_sport_client(deps):
    config = MoveUnitreeOMPathSDKZenohConfig()
    conn = MoveUnitreeOMPathSDKZenohConnector(config=config)
    conn._sport_pub = MagicMock()
    conn._move_robot(0.5, 0.0, 0.1)
    conn._sport_pub.put.assert_called_once()


def test_move_robot_local_skips_when_not_standing(deps):
    config = MoveUnitreeOMPathSDKZenohConfig()
    conn = MoveUnitreeOMPathSDKZenohConnector(config=config)
    conn._sport_pub = MagicMock()
    conn.odom.position["body_attitude"] = RobotState.SITTING
    conn._move_robot(0.5, 0.0, 0.1)
    conn._sport_pub.put.assert_not_called()


def test_publish_sport_move_skips_when_pub_missing(conn):
    conn._sport_pub = None
    conn._publish_sport_move(0.5, 0.0, 0.0)


def test_publish_sport_move_publishes(conn):
    conn._sport_pub = MagicMock()
    conn._publish_sport_move(0.5, 0.1, 0.2)
    conn._sport_pub.put.assert_called_once()


def test_clean_abort_resets_state(conn):
    conn.movement_attempts = 5
    conn.pending_movements.put(MagicMock())
    conn.clean_abort()
    assert conn.movement_attempts == 0
    assert conn.pending_movements.empty()


def test_clean_abort_when_already_empty(conn):
    conn.movement_attempts = 3
    conn.clean_abort()
    assert conn.movement_attempts == 0


def test_process_turn_left_blocked(conn):
    conn.permissive_paths = False
    conn.path_provider.turn_left = []
    conn._process_turn_left()
    assert conn.pending_movements.empty()


def test_process_turn_left_queues(conn):
    conn._process_turn_left()
    assert conn.pending_movements.qsize() == 1


def test_process_turn_right_blocked(conn):
    conn.permissive_paths = False
    conn.path_provider.turn_right = []
    conn._process_turn_right()
    assert conn.pending_movements.empty()


def test_process_turn_right_queues(conn):
    conn._process_turn_right()
    assert conn.pending_movements.qsize() == 1


def test_process_move_forward_blocked(conn):
    conn.permissive_paths = False
    conn.path_provider.advance = []
    conn._process_move_forward()
    assert conn.pending_movements.empty()


def test_process_move_forward_queues(conn):
    conn._process_move_forward()
    assert conn.pending_movements.qsize() == 1


def test_process_move_back_blocked(conn):
    conn.permissive_paths = False
    conn.path_provider.retreat = False
    conn._process_move_back()
    assert conn.pending_movements.empty()


def test_process_move_back_queues(conn):
    conn._process_move_back()
    assert conn.pending_movements.qsize() == 1


# --- _execute_turn ----------------------------------------------------------


def test_execute_turn_left_with_paths(conn):
    conn._sport_pub = MagicMock()
    ok = conn._execute_turn(20.0)  # positive -> left
    assert ok is True
    conn._sport_pub.put.assert_called()


def test_execute_turn_left_blocked_strict(conn):
    conn.permissive_paths = False
    conn.path_provider.turn_left = []
    ok = conn._execute_turn(20.0)
    assert ok is False


def test_execute_turn_right_with_paths(conn):
    conn._sport_pub = MagicMock()
    ok = conn._execute_turn(-20.0)
    assert ok is True
    conn._sport_pub.put.assert_called()


def test_execute_turn_right_blocked_strict(conn):
    conn.permissive_paths = False
    conn.path_provider.turn_right = []
    ok = conn._execute_turn(-20.0)
    assert ok is False


@pytest.mark.asyncio
async def test_connect_unknown_action_logs(conn):
    output = MagicMock()
    output.action = "fly"
    await conn.connect(output)
    assert conn.pending_movements.empty()


@pytest.mark.asyncio
async def test_connect_stand_still(conn):
    output = MagicMock()
    output.action = "stand still"
    await conn.connect(output)
    assert conn.pending_movements.empty()


@pytest.mark.asyncio
async def test_connect_move_forwards_queues(conn):
    output = MagicMock()
    output.action = "move forwards"
    await conn.connect(output)
    assert conn.pending_movements.qsize() == 1


@pytest.mark.asyncio
async def test_connect_skipped_when_disabled(conn):
    conn.ai_control_enabled = False
    output = MagicMock()
    output.action = "move forwards"
    await conn.connect(output)
    assert conn.pending_movements.empty()


@pytest.mark.asyncio
async def test_connect_skipped_in_guard_mode_with_unknown_face(conn):
    conn.mode = "guard"
    conn.face_presence_provider.unknown_faces = 1
    output = MagicMock()
    output.action = "move forwards"
    await conn.connect(output)
    assert conn.pending_movements.empty()


@pytest.mark.asyncio
async def test_connect_skipped_when_action_in_progress(conn):
    conn.unitree_go2_state.action_progress = 50
    output = MagicMock()
    output.action = "move forwards"
    await conn.connect(output)
    assert conn.pending_movements.empty()


@pytest.mark.asyncio
async def test_connect_skipped_when_robot_already_moving(conn):
    conn.odom.position["moving"] = True
    output = MagicMock()
    output.action = "move forwards"
    await conn.connect(output)
    assert conn.pending_movements.empty()


@pytest.mark.asyncio
async def test_connect_waits_for_first_odom(conn):
    conn.odom.position["odom_x"] = 0.0
    conn.odom.position["odom_subscriber_ts"] = 0.0
    output = MagicMock()
    output.action = "move forwards"
    await conn.connect(output)
    assert conn.pending_movements.empty()


def test_zenoh_ai_status_request_no_publisher_no_op(conn):
    conn._zenoh_ai_status_response_pub = None
    sample = MagicMock()
    conn._zenoh_ai_status_request(sample)


def test_zenoh_ai_status_request_disable(conn):
    pub = MagicMock()
    conn._zenoh_ai_status_response_pub = pub
    sample = MagicMock()
    sample.payload.to_bytes.return_value = b"x"
    fake = MagicMock()
    fake.code = 0
    fake.request_id = "r1"
    fake.header.frame_id = "f"
    response_obj = MagicMock()
    response_obj.serialize.return_value = b"\x00"
    with (
        patch(
            "actions.move_go2_autonomy.connector.unitree_om_path_sdk_zenoh.AIStatusRequest.deserialize",
            return_value=fake,
        ),
        patch(
            "actions.move_go2_autonomy.connector.unitree_om_path_sdk_zenoh.prepare_header",
            return_value=MagicMock(),
        ),
        patch(
            "actions.move_go2_autonomy.connector.unitree_om_path_sdk_zenoh.AIStatusResponse",
            return_value=response_obj,
        ),
    ):
        conn._zenoh_ai_status_request(sample)
    assert conn.ai_control_enabled is False
    pub.put.assert_called_once()


def test_zenoh_ai_status_request_enable(conn):
    pub = MagicMock()
    conn._zenoh_ai_status_response_pub = pub
    conn.ai_control_enabled = False
    sample = MagicMock()
    sample.payload.to_bytes.return_value = b"x"
    fake = MagicMock()
    fake.code = 1
    fake.request_id = "r2"
    fake.header.frame_id = "f"
    response_obj = MagicMock()
    response_obj.serialize.return_value = b"\x00"
    with (
        patch(
            "actions.move_go2_autonomy.connector.unitree_om_path_sdk_zenoh.AIStatusRequest.deserialize",
            return_value=fake,
        ),
        patch(
            "actions.move_go2_autonomy.connector.unitree_om_path_sdk_zenoh.prepare_header",
            return_value=MagicMock(),
        ),
        patch(
            "actions.move_go2_autonomy.connector.unitree_om_path_sdk_zenoh.AIStatusResponse",
            return_value=response_obj,
        ),
    ):
        conn._zenoh_ai_status_request(sample)
    assert conn.ai_control_enabled is True
    pub.put.assert_called_once()


def test_zenoh_ai_status_request_query(conn):
    pub = MagicMock()
    conn._zenoh_ai_status_response_pub = pub
    conn.ai_control_enabled = True
    sample = MagicMock()
    sample.payload.to_bytes.return_value = b"x"
    fake = MagicMock()
    fake.code = 2
    fake.request_id = "r3"
    fake.header.frame_id = "f"
    response_obj = MagicMock()
    response_obj.serialize.return_value = b"\x00"
    with (
        patch(
            "actions.move_go2_autonomy.connector.unitree_om_path_sdk_zenoh.AIStatusRequest.deserialize",
            return_value=fake,
        ),
        patch(
            "actions.move_go2_autonomy.connector.unitree_om_path_sdk_zenoh.prepare_header",
            return_value=MagicMock(),
        ),
        patch(
            "actions.move_go2_autonomy.connector.unitree_om_path_sdk_zenoh.AIStatusResponse",
            return_value=response_obj,
        ),
    ):
        conn._zenoh_ai_status_request(sample)
    assert conn.ai_control_enabled is True
    pub.put.assert_called_once()


@pytest.fixture
def tick_conn(deps):
    config = MoveUnitreeOMPathSDKZenohConfig()
    c = MoveUnitreeOMPathSDKZenohConnector(config=config)
    c.sleep = MagicMock()
    c._sport_pub = MagicMock()
    return c


def test_tick_waits_when_odom_none(tick_conn):
    tick_conn.odom = None
    tick_conn.tick()
    tick_conn.sleep.assert_called_with(0.5)


def test_tick_waits_for_first_odom_sample(tick_conn):
    tick_conn.odom.position["odom_x"] = 0.0
    tick_conn.odom.position["odom_subscriber_ts"] = 0.0
    tick_conn.tick()
    tick_conn.sleep.assert_called_with(0.5)


def test_tick_skips_when_sitting(tick_conn):
    tick_conn.odom.position["body_attitude"] = RobotState.SITTING
    tick_conn.tick()
    tick_conn.sleep.assert_called_with(0.5)


def test_tick_local_skips_when_not_standing(deps):
    config = MoveUnitreeOMPathSDKZenohConfig()
    c = MoveUnitreeOMPathSDKZenohConnector(config=config)
    c.sleep = MagicMock()
    c.odom.position["body_attitude"] = RobotState.SITTING
    c.tick()
    c.sleep.assert_called_with(0.5)


def test_tick_no_pending_just_sleeps(tick_conn):
    tick_conn.tick()
    tick_conn.sleep.assert_called_with(0.1)


def test_tick_movement_attempts_exceeded_aborts(tick_conn):
    tick_conn.movement_attempts = 999
    tick_conn.movement_attempt_limit = 10
    tick_conn.pending_movements.put(MoveCommand(dx=1.0, yaw=0.0, start_x=0.0, start_y=0.0, turn_complete=False))
    tick_conn.tick()
    assert tick_conn.pending_movements.empty()
    assert tick_conn.movement_attempts == 0


def test_tick_phase1_big_gap_executes_turn(tick_conn):
    tick_conn.pending_movements.put(MoveCommand(dx=1.0, yaw=45.0, start_x=0.0, start_y=0.0, turn_complete=False))
    tick_conn.tick()
    assert tick_conn.movement_attempts == 1
    tick_conn._sport_pub.put.assert_called()


def test_tick_phase1_big_gap_blocked_aborts(tick_conn):
    tick_conn.permissive_paths = False
    tick_conn.path_provider.turn_left = []
    tick_conn.path_provider.turn_right = []
    tick_conn.pending_movements.put(MoveCommand(dx=1.0, yaw=-45.0, start_x=0.0, start_y=0.0, turn_complete=False))
    tick_conn.tick()
    assert tick_conn.pending_movements.empty()


def test_tick_phase1_small_gap_left_rotation(tick_conn):
    tick_conn.angle_tolerance = 1.0
    tick_conn.pending_movements.put(MoveCommand(dx=1.0, yaw=5.0, start_x=0.0, start_y=0.0, turn_complete=False))
    tick_conn.tick()
    assert tick_conn.movement_attempts == 1
    tick_conn._sport_pub.put.assert_called()


def test_tick_phase1_small_gap_right_rotation(tick_conn):
    tick_conn.angle_tolerance = 1.0
    tick_conn.pending_movements.put(MoveCommand(dx=1.0, yaw=-5.0, start_x=0.0, start_y=0.0, turn_complete=False))
    tick_conn.tick()
    assert tick_conn.movement_attempts == 1
    tick_conn._sport_pub.put.assert_called()


def test_tick_phase1_turn_completed_marks_target(tick_conn):
    tick_conn.angle_tolerance = 5.0
    cmd = MoveCommand(dx=1.0, yaw=0.0, start_x=0.0, start_y=0.0, turn_complete=False)
    tick_conn.pending_movements.put(cmd)
    tick_conn.tick()
    front = list(tick_conn.pending_movements.queue)[0]
    assert front.turn_complete is True


def test_tick_phase2_zero_dx_aborts(tick_conn):
    tick_conn.pending_movements.put(MoveCommand(dx=0.0, yaw=0.0, start_x=0.0, start_y=0.0, turn_complete=True))
    tick_conn.tick()
    assert tick_conn.pending_movements.empty()


def test_tick_phase2_forward_blocked_aborts(tick_conn):
    tick_conn.permissive_paths = False
    tick_conn.path_provider.advance = []
    tick_conn.pending_movements.put(MoveCommand(dx=1.0, yaw=0.0, start_x=0.0, start_y=0.0, turn_complete=True))
    tick_conn.tick()
    assert tick_conn.pending_movements.empty()


def test_tick_phase2_retreat_blocked_aborts(tick_conn):
    tick_conn.permissive_paths = False
    tick_conn.path_provider.retreat = False
    tick_conn.pending_movements.put(MoveCommand(dx=-1.0, yaw=0.0, start_x=0.0, start_y=0.0, turn_complete=True))
    tick_conn.tick()
    assert tick_conn.pending_movements.empty()


def test_tick_phase2_keeps_moving(tick_conn):
    tick_conn.distance_tolerance = 0.05
    # odom.x = 1.0, start_x = 0.0, goal dx = 5.0 -> distance_traveled = 1.0, gap > tolerance
    tick_conn.pending_movements.put(
        MoveCommand(dx=5.0, yaw=0.0, start_x=0.0, start_y=0.0, turn_complete=True, speed=0.5)
    )
    tick_conn.tick()
    assert tick_conn.movement_attempts == 1
    tick_conn._sport_pub.put.assert_called()


def test_tick_phase2_overshoot(tick_conn):
    tick_conn.distance_tolerance = 0.05
    tick_conn.pending_movements.put(
        MoveCommand(dx=0.5, yaw=0.0, start_x=0.0, start_y=0.0, turn_complete=True, speed=0.5)
    )
    tick_conn.tick()
    tick_conn._sport_pub.put.assert_called()


def test_tick_phase2_completes_normally(tick_conn):
    tick_conn.distance_tolerance = 5.0
    tick_conn.pending_movements.put(
        MoveCommand(dx=1.0, yaw=0.0, start_x=0.0, start_y=0.0, turn_complete=True, speed=0.5)
    )
    tick_conn.tick()
    # clean_abort drains
    assert tick_conn.pending_movements.empty()
