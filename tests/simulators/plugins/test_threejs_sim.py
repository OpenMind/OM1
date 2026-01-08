from unittest.mock import patch

from simulators.base import SimulatorConfig
from simulators.plugins.ThreeJSSim import ThreeJSSim


@patch("simulators.plugins.ThreeJSSim.threading.Thread")
@patch("simulators.plugins.ThreeJSSim.time.sleep")
def test_threejs_sim_init(mock_sleep, mock_thread):
    config = SimulatorConfig(name="Three.js Simulator", host="0.0.0.0", port=8001)
    simulator = ThreeJSSim(config)
    assert simulator.name == "Three.js Simulator"
    assert simulator.config == config


@patch("simulators.plugins.ThreeJSSim.threading.Thread")
@patch("simulators.plugins.ThreeJSSim.time.sleep")
def test_threejs_sim_init_default(mock_sleep, mock_thread):
    config = SimulatorConfig()
    simulator = ThreeJSSim(config)
    assert simulator.config == config
    assert hasattr(simulator, "robot_state")
    assert simulator.robot_state["x"] == 0.0
    assert simulator.robot_state["yaw"] == 0.0


@patch("simulators.plugins.ThreeJSSim.threading.Thread")
@patch("simulators.plugins.ThreeJSSim.time.sleep")
def test_threejs_sim_robot_state(mock_sleep, mock_thread):
    config = SimulatorConfig()
    simulator = ThreeJSSim(config)
    
    assert "x" in simulator.robot_state
    assert "y" in simulator.robot_state
    assert "z" in simulator.robot_state
    assert "yaw" in simulator.robot_state
    assert "moving" in simulator.robot_state
    assert "current_action" in simulator.robot_state


@patch("simulators.plugins.ThreeJSSim.threading.Thread")
@patch("simulators.plugins.ThreeJSSim.time.sleep")
def test_threejs_sim_normalize_angle(mock_sleep, mock_thread):
    config = SimulatorConfig()
    simulator = ThreeJSSim(config)
    
    assert simulator._normalize_angle(0) == 0
    assert simulator._normalize_angle(90) == 90
    assert simulator._normalize_angle(180) == 180
    assert simulator._normalize_angle(181) == -179
    assert simulator._normalize_angle(-181) == 179
    assert simulator._normalize_angle(360) == 0
    assert simulator._normalize_angle(-360) == 0


@patch("simulators.plugins.ThreeJSSim.threading.Thread")
@patch("simulators.plugins.ThreeJSSim.time.sleep")
def test_threejs_sim_tick(mock_sleep, mock_thread):
    config = SimulatorConfig()
    simulator = ThreeJSSim(config)
    
    initial_x = simulator.robot_state["x"]
    simulator.tick()
    
    assert simulator.robot_state["x"] == initial_x


@patch("simulators.plugins.ThreeJSSim.threading.Thread")
@patch("simulators.plugins.ThreeJSSim.time.sleep")
def test_threejs_sim_api_command(mock_sleep, mock_thread):
    config = SimulatorConfig()
    simulator = ThreeJSSim(config)
    
    from fastapi.testclient import TestClient
    client = TestClient(simulator.app)
    
    response = client.post("/api/command", json={"action": "move forwards"})
    assert response.status_code == 200
    assert "robot_state" in response.json()


@patch("simulators.plugins.ThreeJSSim.threading.Thread")
@patch("simulators.plugins.ThreeJSSim.time.sleep")
def test_threejs_sim_broadcast_state(mock_sleep, mock_thread):
    config = SimulatorConfig()
    simulator = ThreeJSSim(config)
    
    import asyncio
    asyncio.run(simulator.broadcast_state())

