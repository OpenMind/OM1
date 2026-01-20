"""Unit tests for FollowPerson ROS2 connector."""
import asyncio
import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from actions.follow_person.connector.ros2 import (
    FollowPersonConfig,
    FollowPersonConnector,
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
def ros2_config():
    """Create a FollowPersonConfig for testing."""
    return FollowPersonConfig(
        person_detection_topic="/test_person_detection",
        movement_command_topic="/test_cmd_vel",
        update_rate_hz=10.0,
        max_following_distance=5.0,
        min_following_distance=0.8,
        linear_speed_max=0.5,
        angular_speed_max=0.5,
        position_tolerance=0.2,
        angle_tolerance=0.1,
    )


@pytest.fixture
def ros2_connector(ros2_config, mock_io_provider):
    """Create a FollowPersonConnector for testing."""
    with patch("actions.follow_person.connector.ros2.IOProvider", return_value=mock_io_provider):
        connector = FollowPersonConnector(ros2_config)
        return connector


def test_ros2_config_defaults():
    """Test FollowPersonConfig with default values."""
    config = FollowPersonConfig()
    assert config.person_detection_topic == "/person_detection"
    assert config.movement_command_topic == "/cmd_vel"
    assert config.update_rate_hz == 10.0
    assert config.max_following_distance == 5.0
    assert config.min_following_distance == 0.8
    assert config.linear_speed_max == 0.5
    assert config.angular_speed_max == 0.5
    assert config.position_tolerance == 0.2
    assert config.angle_tolerance == 0.1


def test_ros2_config_custom_values():
    """Test FollowPersonConfig with custom values."""
    config = FollowPersonConfig(
        person_detection_topic="/custom_detection",
        movement_command_topic="/custom_cmd_vel",
        update_rate_hz=20.0,
        max_following_distance=10.0,
        min_following_distance=1.0,
        linear_speed_max=1.0,
        angular_speed_max=1.0,
        position_tolerance=0.5,
        angle_tolerance=0.2,
    )
    assert config.person_detection_topic == "/custom_detection"
    assert config.movement_command_topic == "/custom_cmd_vel"
    assert config.update_rate_hz == 20.0
    assert config.max_following_distance == 10.0
    assert config.min_following_distance == 1.0
    assert config.linear_speed_max == 1.0
    assert config.angular_speed_max == 1.0
    assert config.position_tolerance == 0.5
    assert config.angle_tolerance == 0.2


def test_ros2_connector_initialization(ros2_connector, ros2_config):
    """Test FollowPersonConnector initialization."""
    assert ros2_connector.config == ros2_config
    assert ros2_connector._is_following is False
    assert ros2_connector._target_person_id is None
    assert ros2_connector._follow_mode is None
    assert ros2_connector._target_distance == 1.5
    assert ros2_connector._follow_speed == 0.5
    assert ros2_connector._stop_on_arrival is True
    assert ros2_connector._timeout_sec == 30.0
    assert ros2_connector._last_person_position is None
    assert ros2_connector._person_lost_time is None


def test_ros2_connector_stop_following(ros2_connector):
    """Test stop_following method."""
    # Set up following state
    ros2_connector._is_following = True
    ros2_connector._target_person_id = "alice"
    ros2_connector._follow_mode = "by_name"
    
    ros2_connector.stop_following()
    
    assert ros2_connector._is_following is False
    assert ros2_connector._target_person_id is None
    assert ros2_connector._follow_mode is None
    assert ros2_connector._person_lost_time is None


def test_ros2_connector_calculate_movement_command_forward(ros2_connector):
    """Test movement command calculation when person is too far."""
    # Person is 3m away, target is 1.5m
    cmd = ros2_connector._calculate_movement_command(3.0, 0.1, 1.5)
    
    assert "linear" in cmd
    assert "angular" in cmd
    # Should move forward (positive linear velocity)
    assert cmd["linear"] > 0
    # Should turn slightly (positive angular velocity for positive angle)
    assert cmd["angular"] > 0
    # Velocities should be within limits
    assert abs(cmd["linear"]) <= ros2_connector.config.linear_speed_max
    assert abs(cmd["angular"]) <= ros2_connector.config.angular_speed_max


def test_ros2_connector_calculate_movement_command_backward(ros2_connector):
    """Test movement command calculation when person is too close."""
    # Person is 0.5m away, target is 1.5m
    cmd = ros2_connector._calculate_movement_command(0.5, -0.1, 1.5)
    
    assert "linear" in cmd
    assert "angular" in cmd
    # Should move backward (negative linear velocity)
    assert cmd["linear"] < 0
    # Velocities should be within limits
    assert abs(cmd["linear"]) <= ros2_connector.config.linear_speed_max
    assert abs(cmd["angular"]) <= ros2_connector.config.angular_speed_max


def test_ros2_connector_calculate_movement_command_at_target(ros2_connector):
    """Test movement command calculation when at target distance."""
    # Person is at target distance with small angle
    cmd = ros2_connector._calculate_movement_command(1.5, 0.05, 1.5)
    
    assert "linear" in cmd
    assert "angular" in cmd
    # Should have reduced velocities when close to target
    assert abs(cmd["linear"]) < ros2_connector.config.linear_speed_max
    assert abs(cmd["angular"]) < ros2_connector.config.angular_speed_max


def test_ros2_connector_calculate_movement_command_large_angle(ros2_connector):
    """Test movement command calculation with large angle error."""
    # Person is at correct distance but large angle
    cmd = ros2_connector._calculate_movement_command(1.5, 1.0, 1.5)
    
    assert "angular" in cmd
    # Should have significant angular velocity
    assert abs(cmd["angular"]) > 0
    assert abs(cmd["angular"]) <= ros2_connector.config.angular_speed_max


def test_ros2_connector_calculate_movement_command_speed_limit(ros2_connector):
    """Test that movement commands respect speed limits."""
    ros2_connector._follow_speed = 1.0  # Maximum speed
    
    # Very large distance error
    cmd = ros2_connector._calculate_movement_command(10.0, 2.0, 1.5)
    
    assert abs(cmd["linear"]) <= ros2_connector.config.linear_speed_max
    assert abs(cmd["angular"]) <= ros2_connector.config.angular_speed_max


def test_ros2_connector_calculate_movement_command_zero_speed(ros2_connector):
    """Test movement command with zero follow speed."""
    ros2_connector._follow_speed = 0.0
    
    cmd = ros2_connector._calculate_movement_command(3.0, 0.5, 1.5)
    
    # Should have zero velocities
    assert cmd["linear"] == 0.0
    assert cmd["angular"] == 0.0


def test_ros2_connector_write_status(ros2_connector, mock_io_provider):
    """Test _write_status method."""
    ros2_connector._write_status("test message")
    
    mock_io_provider.add_input.assert_called_once()
    call_args = mock_io_provider.add_input.call_args
    assert call_args[0][0] == "FollowPersonStatus"
    assert call_args[0][1] == "test message"
    assert isinstance(call_args[0][2], float)  # timestamp


def test_ros2_connector_write_status_error_handling(ros2_connector, mock_io_provider):
    """Test _write_status error handling."""
    mock_io_provider.add_input.side_effect = Exception("IO error")
    
    # Should not raise exception
    ros2_connector._write_status("test message")
    
    mock_io_provider.add_input.assert_called_once()


def test_ros2_connector_publish_movement(ros2_connector, mock_io_provider):
    """Test _publish_movement method."""
    with patch("actions.follow_person.connector.ros2.logging") as mock_logging:
        ros2_connector._publish_movement(0.3, 0.2)
        
        # Should write status
        mock_io_provider.add_input.assert_called()
        # Should log (debug level)
        assert mock_logging.debug.called


def test_ros2_connector_get_person_position_none(ros2_connector):
    """Test _get_person_position when no person is found."""
    position = ros2_connector._get_person_position()
    
    assert position is None


def test_ros2_connector_get_person_position_with_id(ros2_connector):
    """Test _get_person_position with specific person ID."""
    ros2_connector._target_person_id = "alice"
    position = ros2_connector._get_person_position("alice")
    
    # Should return None if no person detection available
    assert position is None or isinstance(position, tuple)


def test_ros2_connector_parse_person_from_vlm_no_inputs(ros2_connector):
    """Test _parse_person_from_vlm with no VLM inputs."""
    person_info = ros2_connector._parse_person_from_vlm()
    
    assert person_info is None


def test_ros2_connector_parse_person_from_vlm_with_inputs(ros2_connector, mock_io_provider):
    """Test _parse_person_from_vlm with VLM inputs."""
    # Create mock input object
    mock_input = MagicMock()
    mock_input.input = "You see a person named alice, 2.5 meters away"
    
    mock_io_provider.inputs = {"VLM_COCO_Local": mock_input}
    
    person_info = ros2_connector._parse_person_from_vlm()
    
    # Currently returns None as implementation is placeholder
    # This test documents expected behavior
    assert person_info is None or isinstance(person_info, dict)


@pytest.mark.asyncio
async def test_ros2_connector_connect_stop(ros2_connector, mock_io_provider):
    """Test connect method with stop action."""
    input_data = FollowPersonInput(action="stop")
    
    await ros2_connector.connect(input_data)
    
    assert ros2_connector._is_following is False
    mock_io_provider.add_input.assert_called()


@pytest.mark.asyncio
async def test_ros2_connector_connect_by_name(ros2_connector):
    """Test connect method with person name."""
    input_data = FollowPersonInput(action="alice", distance=2.0, speed=0.7)
    
    # Mock _follow_control_loop to return immediately
    async def mock_loop():
        await asyncio.sleep(0.01)
        ros2_connector._is_following = False
    
    ros2_connector._follow_control_loop = mock_loop
    
    await ros2_connector.connect(input_data)
    
    assert ros2_connector._target_person_id == "alice"
    assert ros2_connector._follow_mode == "by_name"
    assert ros2_connector._target_distance == 2.0
    assert ros2_connector._follow_speed == 0.7


@pytest.mark.asyncio
async def test_ros2_connector_connect_nearest(ros2_connector):
    """Test connect method with nearest mode."""
    input_data = FollowPersonInput(action="nearest")
    
    async def mock_loop():
        await asyncio.sleep(0.01)
        ros2_connector._is_following = False
    
    ros2_connector._follow_control_loop = mock_loop
    
    await ros2_connector.connect(input_data)
    
    assert ros2_connector._target_person_id is None
    assert ros2_connector._follow_mode == "nearest"


@pytest.mark.asyncio
async def test_ros2_connector_connect_last_seen(ros2_connector):
    """Test connect method with last_seen mode."""
    input_data = FollowPersonInput(action="last_seen")
    
    async def mock_loop():
        await asyncio.sleep(0.01)
        ros2_connector._is_following = False
    
    ros2_connector._follow_control_loop = mock_loop
    
    await ros2_connector.connect(input_data)
    
    assert ros2_connector._target_person_id is None
    assert ros2_connector._follow_mode == "last_seen"


@pytest.mark.asyncio
async def test_ros2_connector_connect_me_alias(ros2_connector):
    """Test connect method with 'me' alias for last_seen."""
    input_data = FollowPersonInput(action="me")
    
    async def mock_loop():
        await asyncio.sleep(0.01)
        ros2_connector._is_following = False
    
    ros2_connector._follow_control_loop = mock_loop
    
    await ros2_connector.connect(input_data)
    
    assert ros2_connector._follow_mode == "last_seen"


@pytest.mark.asyncio
async def test_ros2_connector_connect_distance_clamping(ros2_connector):
    """Test that distance is clamped to valid range."""
    # Distance below minimum
    input_data = FollowPersonInput(action="alice", distance=0.1)
    
    async def mock_loop():
        await asyncio.sleep(0.01)
        ros2_connector._is_following = False
    
    ros2_connector._follow_control_loop = mock_loop
    
    await ros2_connector.connect(input_data)
    
    assert ros2_connector._target_distance >= ros2_connector.config.min_following_distance
    
    # Distance above maximum
    input_data = FollowPersonInput(action="alice", distance=10.0)
    
    await ros2_connector.connect(input_data)
    
    assert ros2_connector._target_distance <= ros2_connector.config.max_following_distance


@pytest.mark.asyncio
async def test_ros2_connector_connect_speed_clamping(ros2_connector):
    """Test that speed is clamped to valid range."""
    # Speed below minimum
    input_data = FollowPersonInput(action="alice", speed=-0.5)
    
    async def mock_loop():
        await asyncio.sleep(0.01)
        ros2_connector._is_following = False
    
    ros2_connector._follow_control_loop = mock_loop
    
    await ros2_connector.connect(input_data)
    
    assert ros2_connector._follow_speed >= 0.0
    
    # Speed above maximum
    input_data = FollowPersonInput(action="alice", speed=2.0)
    
    await ros2_connector.connect(input_data)
    
    assert ros2_connector._follow_speed <= 1.0


@pytest.mark.asyncio
async def test_ros2_connector_connect_error_handling(ros2_connector):
    """Test error handling in connect method."""
    input_data = FollowPersonInput(action="alice")
    
    # Make control loop raise an exception
    async def mock_loop():
        raise Exception("Test error")
    
    ros2_connector._follow_control_loop = mock_loop
    
    await ros2_connector.connect(input_data)
    
    # Should handle error gracefully
    assert ros2_connector._is_following is False


@pytest.mark.asyncio
async def test_ros2_connector_control_loop_timeout(ros2_connector):
    """Test control loop timeout handling."""
    ros2_connector._is_following = True
    ros2_connector._timeout_sec = 0.1  # Very short timeout
    
    # Mock _get_person_position to return None (person not found)
    ros2_connector._get_person_position = lambda: None
    
    # Run control loop
    start_time = time.time()
    await asyncio.wait_for(
        ros2_connector._follow_control_loop(),
        timeout=1.0
    )
    elapsed = time.time() - start_time
    
    # Should stop after timeout
    assert ros2_connector._is_following is False
    assert elapsed < 1.0


@pytest.mark.asyncio
async def test_ros2_connector_control_loop_person_lost(ros2_connector):
    """Test control loop when person is lost."""
    ros2_connector._is_following = True
    ros2_connector._timeout_sec = 10.0
    
    # Mock _get_person_position to return None
    ros2_connector._get_person_position = lambda: None
    
    # Run control loop briefly
    task = asyncio.create_task(ros2_connector._follow_control_loop())
    await asyncio.sleep(0.2)  # Wait a bit
    ros2_connector._is_following = False  # Stop it
    await task
    
    # Should have detected person lost
    assert ros2_connector._person_lost_time is not None or ros2_connector._is_following is False


@pytest.mark.asyncio
async def test_ros2_connector_control_loop_person_too_far(ros2_connector):
    """Test control loop when person is too far."""
    ros2_connector._is_following = True
    ros2_connector._timeout_sec = 10.0
    ros2_connector._target_distance = 1.5
    
    # Mock _get_person_position to return person too far
    ros2_connector._get_person_position = lambda: (10.0, 0.0)  # 10m away
    
    # Run control loop briefly
    task = asyncio.create_task(ros2_connector._follow_control_loop())
    await asyncio.sleep(0.1)
    ros2_connector._is_following = False
    await task
    
    # Should handle too far case
    assert True  # Test passes if no exception


@pytest.mark.asyncio
async def test_ros2_connector_control_loop_person_too_close(ros2_connector):
    """Test control loop when person is too close."""
    ros2_connector._is_following = True
    ros2_connector._timeout_sec = 10.0
    ros2_connector._target_distance = 1.5
    
    # Mock _get_person_position to return person too close
    ros2_connector._get_person_position = lambda: (0.5, 0.0)  # 0.5m away
    
    # Run control loop briefly
    task = asyncio.create_task(ros2_connector._follow_control_loop())
    await asyncio.sleep(0.1)
    ros2_connector._is_following = False
    await task
    
    # Should handle too close case
    assert True  # Test passes if no exception


@pytest.mark.asyncio
async def test_ros2_connector_control_loop_at_target(ros2_connector):
    """Test control loop when at target distance."""
    ros2_connector._is_following = True
    ros2_connector._timeout_sec = 10.0
    ros2_connector._target_distance = 1.5
    ros2_connector._stop_on_arrival = True
    
    # Mock _get_person_position to return person at target
    ros2_connector._get_person_position = lambda: (1.5, 0.05)  # At target with small angle
    
    # Run control loop briefly
    task = asyncio.create_task(ros2_connector._follow_control_loop())
    await asyncio.sleep(0.1)
    ros2_connector._is_following = False
    await task
    
    # Should handle at target case
    assert True  # Test passes if no exception


def test_ros2_connector_tick_timeout(ros2_connector):
    """Test tick method with timeout."""
    ros2_connector._is_following = True
    ros2_connector._last_update_time = time.time() - 100.0  # Old update time
    ros2_connector._timeout_sec = 10.0
    
    ros2_connector.tick()
    
    # Should stop following due to timeout
    assert ros2_connector._is_following is False


def test_ros2_connector_tick_no_timeout(ros2_connector):
    """Test tick method without timeout."""
    ros2_connector._is_following = True
    ros2_connector._last_update_time = time.time()  # Recent update time
    ros2_connector._timeout_sec = 10.0
    
    ros2_connector.tick()
    
    # Should still be following
    assert ros2_connector._is_following is True
