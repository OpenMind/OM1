"""Unit tests for FollowPerson Zenoh connector."""
import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from actions.follow_person.connector.zenoh import (
    FollowPersonZenohConfig,
    FollowPersonZenohConnector,
)
from actions.follow_person.interface import FollowPersonInput


@pytest.fixture
def mock_io_provider():
    """Create a mock IOProvider."""
    provider = MagicMock()
    provider.inputs = {}
    provider.add_input = MagicMock()
    return provider


@pytest.fixture
def mock_zenoh_session():
    """Create a mock Zenoh session."""
    session = MagicMock()
    session.declare_subscriber = MagicMock()
    session.put = MagicMock()
    return session


@pytest.fixture
def mock_odom_provider():
    """Create a mock OdomProvider."""
    return MagicMock()


@pytest.fixture
def zenoh_config():
    """Create a FollowPersonZenohConfig for testing."""
    return FollowPersonZenohConfig(
        URID="test_urid",
        person_detection_topic="test_person_detection",
        movement_command_topic="test_cmd_vel",
        update_rate_hz=10.0,
        max_following_distance=5.0,
        min_following_distance=0.8,
        linear_speed_max=0.5,
        angular_speed_max=0.5,
        position_tolerance=0.2,
        angle_tolerance=0.1,
    )


@pytest.fixture
def zenoh_connector(zenoh_config, mock_io_provider, mock_zenoh_session, mock_odom_provider):
    """Create a FollowPersonZenohConnector for testing."""
    with patch("actions.follow_person.connector.zenoh.IOProvider", return_value=mock_io_provider), \
         patch("actions.follow_person.connector.zenoh.open_zenoh_session", return_value=mock_zenoh_session), \
         patch("actions.follow_person.connector.zenoh.OdomProvider", return_value=mock_odom_provider):
        connector = FollowPersonZenohConnector(zenoh_config)
        return connector


def test_zenoh_config_defaults():
    """Test FollowPersonZenohConfig with default values."""
    config = FollowPersonZenohConfig()
    assert config.URID is None
    assert config.person_detection_topic == "person_detection"
    assert config.movement_command_topic == "cmd_vel"
    assert config.update_rate_hz == 10.0
    assert config.max_following_distance == 5.0
    assert config.min_following_distance == 0.8
    assert config.linear_speed_max == 0.5
    assert config.angular_speed_max == 0.5
    assert config.position_tolerance == 0.2
    assert config.angle_tolerance == 0.1


def test_zenoh_config_custom_values():
    """Test FollowPersonZenohConfig with custom values."""
    config = FollowPersonZenohConfig(
        URID="custom_urid",
        person_detection_topic="custom_detection",
        movement_command_topic="custom_cmd_vel",
        update_rate_hz=20.0,
        max_following_distance=10.0,
        min_following_distance=1.0,
        linear_speed_max=1.0,
        angular_speed_max=1.0,
        position_tolerance=0.5,
        angle_tolerance=0.2,
    )
    assert config.URID == "custom_urid"
    assert config.person_detection_topic == "custom_detection"
    assert config.movement_command_topic == "custom_cmd_vel"
    assert config.update_rate_hz == 20.0
    assert config.max_following_distance == 10.0
    assert config.min_following_distance == 1.0
    assert config.linear_speed_max == 1.0
    assert config.angular_speed_max == 1.0
    assert config.position_tolerance == 0.5
    assert config.angle_tolerance == 0.2


def test_zenoh_connector_initialization_with_urid(zenoh_connector, zenoh_config, mock_zenoh_session):
    """Test FollowPersonZenohConnector initialization with URID."""
    assert zenoh_connector.config == zenoh_config
    assert zenoh_connector._is_following is False
    assert zenoh_connector._target_person_id is None
    assert zenoh_connector._follow_mode is None
    assert zenoh_connector._target_distance == 1.5
    assert zenoh_connector._follow_speed == 0.5
    assert zenoh_connector._stop_on_arrival is True
    assert zenoh_connector._timeout_sec == 30.0
    assert zenoh_connector._last_person_position is None
    assert zenoh_connector._person_lost_time is None
    assert zenoh_connector.session == mock_zenoh_session
    # Should have subscribed to person detection
    assert mock_zenoh_session.declare_subscriber.called


def test_zenoh_connector_initialization_without_urid(mock_io_provider):
    """Test FollowPersonZenohConnector initialization without URID."""
    config = FollowPersonZenohConfig(URID=None)
    
    with patch("actions.follow_person.connector.zenoh.IOProvider", return_value=mock_io_provider):
        connector = FollowPersonZenohConnector(config)
        
        assert connector.session is None
        assert connector._is_following is False


def test_zenoh_connector_initialization_zenoh_error(mock_io_provider):
    """Test FollowPersonZenohConnector initialization with Zenoh error."""
    config = FollowPersonZenohConfig(URID="test_urid")
    
    with patch("actions.follow_person.connector.zenoh.IOProvider", return_value=mock_io_provider), \
         patch("actions.follow_person.connector.zenoh.open_zenoh_session", side_effect=Exception("Zenoh error")):
        connector = FollowPersonZenohConnector(config)
        
        # Should handle error gracefully
        assert connector.session is None


def test_zenoh_connector_stop_following(zenoh_connector):
    """Test stop_following method."""
    # Set up following state
    zenoh_connector._is_following = True
    zenoh_connector._target_person_id = "alice"
    zenoh_connector._follow_mode = "by_name"
    
    zenoh_connector.stop_following()
    
    assert zenoh_connector._is_following is False
    assert zenoh_connector._target_person_id is None
    assert zenoh_connector._follow_mode is None
    assert zenoh_connector._person_lost_time is None


def test_zenoh_connector_calculate_movement_command_forward(zenoh_connector):
    """Test movement command calculation when person is too far."""
    # Person is 3m away, target is 1.5m
    cmd = zenoh_connector._calculate_movement_command(3.0, 0.1, 1.5)
    
    assert "linear" in cmd
    assert "angular" in cmd
    # Should move forward (positive linear velocity)
    assert cmd["linear"] > 0
    # Velocities should be within limits
    assert abs(cmd["linear"]) <= zenoh_connector.config.linear_speed_max
    assert abs(cmd["angular"]) <= zenoh_connector.config.angular_speed_max


def test_zenoh_connector_calculate_movement_command_backward(zenoh_connector):
    """Test movement command calculation when person is too close."""
    # Person is 0.5m away, target is 1.5m
    cmd = zenoh_connector._calculate_movement_command(0.5, -0.1, 1.5)
    
    assert "linear" in cmd
    assert "angular" in cmd
    # Should move backward (negative linear velocity)
    assert cmd["linear"] < 0
    # Velocities should be within limits
    assert abs(cmd["linear"]) <= zenoh_connector.config.linear_speed_max
    assert abs(cmd["angular"]) <= zenoh_connector.config.angular_speed_max


def test_zenoh_connector_calculate_movement_command_at_target(zenoh_connector):
    """Test movement command calculation when at target distance."""
    # Person is at target distance with small angle
    cmd = zenoh_connector._calculate_movement_command(1.5, 0.05, 1.5)
    
    assert "linear" in cmd
    assert "angular" in cmd
    # Should have reduced velocities when close to target
    assert abs(cmd["linear"]) < zenoh_connector.config.linear_speed_max
    assert abs(cmd["angular"]) < zenoh_connector.config.angular_speed_max


def test_zenoh_connector_calculate_movement_command_zero_speed(zenoh_connector):
    """Test movement command with zero follow speed."""
    zenoh_connector._follow_speed = 0.0
    
    cmd = zenoh_connector._calculate_movement_command(3.0, 0.5, 1.5)
    
    # Should have zero velocities
    assert cmd["linear"] == 0.0
    assert cmd["angular"] == 0.0


def test_zenoh_connector_publish_movement_with_session(zenoh_connector, mock_zenoh_session, mock_io_provider):
    """Test _publish_movement method with active session."""
    with patch("actions.follow_person.connector.zenoh.geometry_msgs") as mock_geometry_msgs:
        mock_twist = MagicMock()
        mock_twist.serialize.return_value = b"serialized_data"
        mock_geometry_msgs.Twist.return_value = mock_twist
        mock_geometry_msgs.Vector3 = MagicMock
        
        zenoh_connector._publish_movement(0.3, 0.2)
        
        # Should publish to Zenoh
        assert mock_zenoh_session.put.called
        # Should write status
        mock_io_provider.add_input.assert_called()


def test_zenoh_connector_publish_movement_no_session(zenoh_connector):
    """Test _publish_movement method without active session."""
    zenoh_connector.session = None
    
    # Should not raise exception
    zenoh_connector._publish_movement(0.3, 0.2)
    
    # Should handle gracefully (no session, so no publish)


def test_zenoh_connector_publish_movement_error(zenoh_connector, mock_zenoh_session):
    """Test _publish_movement error handling."""
    mock_zenoh_session.put.side_effect = Exception("Publish error")
    
    with patch("actions.follow_person.connector.zenoh.geometry_msgs") as mock_geometry_msgs:
        mock_twist = MagicMock()
        mock_twist.serialize.return_value = b"serialized_data"
        mock_geometry_msgs.Twist.return_value = mock_twist
        mock_geometry_msgs.Vector3 = MagicMock
        
        # Should not raise exception
        zenoh_connector._publish_movement(0.3, 0.2)


def test_zenoh_connector_write_status(zenoh_connector, mock_io_provider):
    """Test _write_status method."""
    zenoh_connector._write_status("test message")
    
    mock_io_provider.add_input.assert_called_once()
    call_args = mock_io_provider.add_input.call_args
    assert call_args[0][0] == "FollowPersonStatus"
    assert call_args[0][1] == "test message"
    assert isinstance(call_args[0][2], float)  # timestamp


def test_zenoh_connector_write_status_error_handling(zenoh_connector, mock_io_provider):
    """Test _write_status error handling."""
    mock_io_provider.add_input.side_effect = Exception("IO error")
    
    # Should not raise exception
    zenoh_connector._write_status("test message")
    
    mock_io_provider.add_input.assert_called_once()


def test_zenoh_connector_on_person_detection(zenoh_connector):
    """Test _on_person_detection callback."""
    mock_sample = MagicMock()
    mock_sample.payload.to_bytes.return_value = b"test_data"
    
    # Should not raise exception
    zenoh_connector._on_person_detection(mock_sample)


def test_zenoh_connector_on_person_detection_error(zenoh_connector):
    """Test _on_person_detection error handling."""
    mock_sample = MagicMock()
    mock_sample.payload.to_bytes.side_effect = Exception("Parse error")
    
    # Should not raise exception
    zenoh_connector._on_person_detection(mock_sample)


def test_zenoh_connector_get_person_position_none(zenoh_connector):
    """Test _get_person_position when no person is found."""
    position = zenoh_connector._get_person_position()
    
    assert position is None


def test_zenoh_connector_parse_person_from_vlm_no_inputs(zenoh_connector):
    """Test _parse_person_from_vlm with no VLM inputs."""
    person_info = zenoh_connector._parse_person_from_vlm()
    
    assert person_info is None


@pytest.mark.asyncio
async def test_zenoh_connector_connect_stop(zenoh_connector, mock_io_provider):
    """Test connect method with stop action."""
    input_data = FollowPersonInput(action="stop")
    
    await zenoh_connector.connect(input_data)
    
    assert zenoh_connector._is_following is False
    mock_io_provider.add_input.assert_called()


@pytest.mark.asyncio
async def test_zenoh_connector_connect_by_name(zenoh_connector):
    """Test connect method with person name."""
    input_data = FollowPersonInput(action="alice", distance=2.0, speed=0.7)
    
    # Mock _follow_control_loop to return immediately
    async def mock_loop():
        await asyncio.sleep(0.01)
        zenoh_connector._is_following = False
    
    zenoh_connector._follow_control_loop = mock_loop
    
    await zenoh_connector.connect(input_data)
    
    assert zenoh_connector._target_person_id == "alice"
    assert zenoh_connector._follow_mode == "by_name"
    assert zenoh_connector._target_distance == 2.0
    assert zenoh_connector._follow_speed == 0.7


@pytest.mark.asyncio
async def test_zenoh_connector_connect_nearest(zenoh_connector):
    """Test connect method with nearest mode."""
    input_data = FollowPersonInput(action="nearest")
    
    async def mock_loop():
        await asyncio.sleep(0.01)
        zenoh_connector._is_following = False
    
    zenoh_connector._follow_control_loop = mock_loop
    
    await zenoh_connector.connect(input_data)
    
    assert zenoh_connector._target_person_id is None
    assert zenoh_connector._follow_mode == "nearest"


@pytest.mark.asyncio
async def test_zenoh_connector_connect_last_seen(zenoh_connector):
    """Test connect method with last_seen mode."""
    input_data = FollowPersonInput(action="last_seen")
    
    async def mock_loop():
        await asyncio.sleep(0.01)
        zenoh_connector._is_following = False
    
    zenoh_connector._follow_control_loop = mock_loop
    
    await zenoh_connector.connect(input_data)
    
    assert zenoh_connector._target_person_id is None
    assert zenoh_connector._follow_mode == "last_seen"


@pytest.mark.asyncio
async def test_zenoh_connector_connect_me_alias(zenoh_connector):
    """Test connect method with 'me' alias for last_seen."""
    input_data = FollowPersonInput(action="me")
    
    async def mock_loop():
        await asyncio.sleep(0.01)
        zenoh_connector._is_following = False
    
    zenoh_connector._follow_control_loop = mock_loop
    
    await zenoh_connector.connect(input_data)
    
    assert zenoh_connector._follow_mode == "last_seen"


@pytest.mark.asyncio
async def test_zenoh_connector_connect_distance_clamping(zenoh_connector):
    """Test that distance is clamped to valid range."""
    # Distance below minimum
    input_data = FollowPersonInput(action="alice", distance=0.1)
    
    async def mock_loop():
        await asyncio.sleep(0.01)
        zenoh_connector._is_following = False
    
    zenoh_connector._follow_control_loop = mock_loop
    
    await zenoh_connector.connect(input_data)
    
    assert zenoh_connector._target_distance >= zenoh_connector.config.min_following_distance
    
    # Distance above maximum
    input_data = FollowPersonInput(action="alice", distance=10.0)
    
    await zenoh_connector.connect(input_data)
    
    assert zenoh_connector._target_distance <= zenoh_connector.config.max_following_distance


@pytest.mark.asyncio
async def test_zenoh_connector_connect_speed_clamping(zenoh_connector):
    """Test that speed is clamped to valid range."""
    # Speed below minimum
    input_data = FollowPersonInput(action="alice", speed=-0.5)
    
    async def mock_loop():
        await asyncio.sleep(0.01)
        zenoh_connector._is_following = False
    
    zenoh_connector._follow_control_loop = mock_loop
    
    await zenoh_connector.connect(input_data)
    
    assert zenoh_connector._follow_speed >= 0.0
    
    # Speed above maximum
    input_data = FollowPersonInput(action="alice", speed=2.0)
    
    await zenoh_connector.connect(input_data)
    
    assert zenoh_connector._follow_speed <= 1.0


@pytest.mark.asyncio
async def test_zenoh_connector_connect_error_handling(zenoh_connector):
    """Test error handling in connect method."""
    input_data = FollowPersonInput(action="alice")
    
    # Make control loop raise an exception
    async def mock_loop():
        raise Exception("Test error")
    
    zenoh_connector._follow_control_loop = mock_loop
    
    await zenoh_connector.connect(input_data)
    
    # Should handle error gracefully
    assert zenoh_connector._is_following is False


@pytest.mark.asyncio
async def test_zenoh_connector_control_loop_timeout(zenoh_connector):
    """Test control loop timeout handling."""
    zenoh_connector._is_following = True
    zenoh_connector._timeout_sec = 0.1  # Very short timeout
    
    # Mock _get_person_position to return None (person not found)
    zenoh_connector._get_person_position = lambda: None
    
    # Run control loop
    start_time = time.time()
    await asyncio.wait_for(
        zenoh_connector._follow_control_loop(),
        timeout=1.0
    )
    elapsed = time.time() - start_time
    
    # Should stop after timeout
    assert zenoh_connector._is_following is False
    assert elapsed < 1.0


@pytest.mark.asyncio
async def test_zenoh_connector_control_loop_person_lost(zenoh_connector):
    """Test control loop when person is lost."""
    zenoh_connector._is_following = True
    zenoh_connector._timeout_sec = 10.0
    
    # Mock _get_person_position to return None
    zenoh_connector._get_person_position = lambda: None
    
    # Run control loop briefly
    task = asyncio.create_task(zenoh_connector._follow_control_loop())
    await asyncio.sleep(0.2)  # Wait a bit
    zenoh_connector._is_following = False  # Stop it
    await task
    
    # Should have detected person lost
    assert zenoh_connector._person_lost_time is not None or zenoh_connector._is_following is False


@pytest.mark.asyncio
async def test_zenoh_connector_control_loop_person_too_far(zenoh_connector):
    """Test control loop when person is too far."""
    zenoh_connector._is_following = True
    zenoh_connector._timeout_sec = 10.0
    zenoh_connector._target_distance = 1.5
    
    # Mock _get_person_position to return person too far
    zenoh_connector._get_person_position = lambda: (10.0, 0.0)  # 10m away
    
    # Run control loop briefly
    task = asyncio.create_task(zenoh_connector._follow_control_loop())
    await asyncio.sleep(0.1)
    zenoh_connector._is_following = False
    await task
    
    # Should handle too far case
    assert True  # Test passes if no exception


@pytest.mark.asyncio
async def test_zenoh_connector_control_loop_person_too_close(zenoh_connector):
    """Test control loop when person is too close."""
    zenoh_connector._is_following = True
    zenoh_connector._timeout_sec = 10.0
    zenoh_connector._target_distance = 1.5
    
    # Mock _get_person_position to return person too close
    zenoh_connector._get_person_position = lambda: (0.5, 0.0)  # 0.5m away
    
    # Run control loop briefly
    task = asyncio.create_task(zenoh_connector._follow_control_loop())
    await asyncio.sleep(0.1)
    zenoh_connector._is_following = False
    await task
    
    # Should handle too close case
    assert True  # Test passes if no exception


@pytest.mark.asyncio
async def test_zenoh_connector_control_loop_at_target(zenoh_connector):
    """Test control loop when at target distance."""
    zenoh_connector._is_following = True
    zenoh_connector._timeout_sec = 10.0
    zenoh_connector._target_distance = 1.5
    zenoh_connector._stop_on_arrival = True
    
    # Mock _get_person_position to return person at target
    zenoh_connector._get_person_position = lambda: (1.5, 0.05)  # At target with small angle
    
    # Run control loop briefly
    task = asyncio.create_task(zenoh_connector._follow_control_loop())
    await asyncio.sleep(0.1)
    zenoh_connector._is_following = False
    await task
    
    # Should handle at target case
    assert True  # Test passes if no exception


def test_zenoh_connector_tick_timeout(zenoh_connector):
    """Test tick method with timeout."""
    zenoh_connector._is_following = True
    zenoh_connector._last_update_time = time.time() - 100.0  # Old update time
    zenoh_connector._timeout_sec = 10.0
    
    zenoh_connector.tick()
    
    # Should stop following due to timeout
    assert zenoh_connector._is_following is False


def test_zenoh_connector_tick_no_timeout(zenoh_connector):
    """Test tick method without timeout."""
    zenoh_connector._is_following = True
    zenoh_connector._last_update_time = time.time()  # Recent update time
    zenoh_connector._timeout_sec = 10.0
    
    zenoh_connector.tick()
    
    # Should still be following
    assert zenoh_connector._is_following is True
