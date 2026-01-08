from unittest.mock import Mock, patch

import pytest

from actions.move_web_sim.connector.websocket import (
    MoveWebSimConfig,
    MoveWebSimConnector,
)
from actions.move_web_sim.interface import MoveInput, MovementAction


@pytest.fixture
def connector_config():
    return MoveWebSimConfig(simulator_url="http://localhost:8001")


@pytest.fixture
def connector(connector_config):
    return MoveWebSimConnector(connector_config)


def test_connector_init(connector):
    assert connector.simulator_url == "http://localhost:8001"
    assert connector.robot_state["x"] == 0.0
    assert connector.robot_state["yaw"] == 0.0
    assert connector.robot_state["moving"] is False


def test_connector_normalize_angle(connector):
    assert connector._normalize_angle(0) == 0
    assert connector._normalize_angle(90) == 90
    assert connector._normalize_angle(180) == 180
    assert connector._normalize_angle(181) == -179
    assert connector._normalize_angle(-181) == 179
    assert connector._normalize_angle(360) == 0


def test_connector_calculate_angle_gap(connector):
    gap = connector._calculate_angle_gap(0, 90)
    assert gap == -90

    gap = connector._calculate_angle_gap(90, 0)
    assert gap == 90

    gap = connector._calculate_angle_gap(170, -170)
    assert abs(gap) < 180


@patch("actions.move_web_sim.connector.websocket.requests.post")
@pytest.mark.asyncio
async def test_connector_send_command(mock_post, connector):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "ok", "robot_state": {"x": 1.0}}
    mock_post.return_value = mock_response

    connector._send_command({"type": "move_forward"})

    mock_post.assert_called_once()
    assert mock_post.call_args[0][0] == "http://localhost:8001/api/command"


@patch("actions.move_web_sim.connector.websocket.requests.post")
@pytest.mark.asyncio
async def test_connector_connect_move_forward(mock_post, connector):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "ok"}
    mock_post.return_value = mock_response

    move_input = MoveInput(action=MovementAction.MOVE_FORWARDS)
    await connector.connect(move_input)

    assert mock_post.called


@patch("actions.move_web_sim.connector.websocket.requests.post")
@pytest.mark.asyncio
async def test_connector_connect_turn_left(mock_post, connector):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "ok"}
    mock_post.return_value = mock_response

    move_input = MoveInput(action=MovementAction.TURN_LEFT)
    await connector.connect(move_input)

    assert mock_post.called


@patch("actions.move_web_sim.connector.websocket.requests.post")
@pytest.mark.asyncio
async def test_connector_connect_turn_right(mock_post, connector):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "ok"}
    mock_post.return_value = mock_response

    move_input = MoveInput(action=MovementAction.TURN_RIGHT)
    await connector.connect(move_input)

    assert mock_post.called


@patch("actions.move_web_sim.connector.websocket.requests.post")
@pytest.mark.asyncio
async def test_connector_connect_move_back(mock_post, connector):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "ok"}
    mock_post.return_value = mock_response

    move_input = MoveInput(action=MovementAction.MOVE_BACK)
    await connector.connect(move_input)

    assert mock_post.called


@patch("actions.move_web_sim.connector.websocket.requests.post")
@pytest.mark.asyncio
async def test_connector_connect_stand_still(mock_post, connector):
    move_input = MoveInput(action=MovementAction.STAND_STILL)
    await connector.connect(move_input)

    assert not mock_post.called


def test_connector_clean_abort(connector):
    connector.robot_state["moving"] = True
    connector.movement_attempts = 5

    connector.clean_abort()

    assert connector.robot_state["moving"] is False
    assert connector.movement_attempts == 0


@patch("actions.move_web_sim.connector.websocket.requests.post")
def test_connector_tick(mock_post, connector):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "ok"}
    mock_post.return_value = mock_response

    connector.tick()

    assert connector.robot_state["moving"] is False
