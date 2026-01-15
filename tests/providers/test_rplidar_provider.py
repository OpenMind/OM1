# tests/providers/test_rplidar_provider.py

import math
from unittest.mock import patch

import numpy as np
import pytest

from src.providers.rplidar_provider import RPLidarProvider


@pytest.fixture
def rplidar_provider():
    """
    Fixture to create an RPLidarProvider instance for testing.
    Uses _singleton_class to get the original class and __new__ to avoid running __init__.
    """
    original_class = RPLidarProvider._singleton_class
    provider = original_class.__new__(original_class)
    # Initialize essential attributes that might be needed for tests
    # For example, _generate_movement_string checks self.use_zenoh and self.machine_type
    provider.use_zenoh = False
    provider.machine_type = "go2"  # Default, adjust in specific tests if needed
    provider.turn_left = (
        []
    )  # These are modified by _path_processor, initialize for _generate_movement_string
    provider.turn_right = []
    provider.advance = []
    provider.retreat = False
    return provider


def test_distance_point_to_line_segment_zero_length_segment(rplidar_provider):
    """
    Test distance calculation when the line segment has zero length (x1,y1) == (x2,y2).
    Should return distance from point to that single point.
    """
    px, py = 5.0, 5.0
    x1, y1 = 2.0, 2.0
    x2, y2 = 2.0, 2.0  # Zero length segment

    expected_dist = math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
    result = rplidar_provider.distance_point_to_line_segment(px, py, x1, y1, x2, y2)

    assert result == expected_dist


def test_distance_point_to_line_segment_perpendicular_projection(rplidar_provider):
    """
    Test distance calculation where the closest point on the line segment is the perpendicular projection.
    Segment from (0,0) to (4,0), point at (2, 3). Closest point is (2, 0), distance is 3.
    """
    px, py = 2.0, 3.0
    x1, y1 = 0.0, 0.0
    x2, y2 = 4.0, 0.0

    expected_dist = 3.0
    result = rplidar_provider.distance_point_to_line_segment(px, py, x1, y1, x2, y2)

    assert result == expected_dist


def test_distance_point_to_line_segment_endpoint_projection(rplidar_provider):
    """
    Test distance calculation where the closest point on the line segment is an endpoint.
    Segment from (0,0) to (2,0), point at (-1, 1). Closest point is (0, 0), distance is sqrt(2).
    """
    px, py = -1.0, 1.0
    x1, y1 = 0.0, 0.0
    x2, y2 = 2.0, 0.0

    expected_dist = math.sqrt(2)
    result = rplidar_provider.distance_point_to_line_segment(px, py, x1, y1, x2, y2)

    assert math.isclose(result, expected_dist, abs_tol=1e-9)


def test_create_straight_path_from_angle_zero_degrees(rplidar_provider):
    """
    Test creating a straight path at 0 degrees.
    Should result in a path along the positive y-axis.
    """
    angle_deg = 0.0
    length = 1.0
    num_points = 5  # Use fewer points for easier checking

    path = rplidar_provider._create_straight_path_from_angle(
        angle_deg, length, num_points
    )

    # Shape should be (2, num_points) -> [x_vals, y_vals]
    assert path.shape == (2, num_points)

    x_vals, y_vals = path
    # At 0 degrees, x should be ~0, y should go from 0 to length
    expected_x = np.zeros(num_points)
    expected_y = np.linspace(0.0, length, num_points)

    np.testing.assert_allclose(x_vals, expected_x, atol=1e-10)
    np.testing.assert_allclose(y_vals, expected_y, atol=1e-10)


def test_create_straight_path_from_angle_ninety_degrees(rplidar_provider):
    """
    Test creating a straight path at 90 degrees.
    Should result in a path along the positive x-axis.
    """
    angle_deg = 90.0
    length = 1.0
    num_points = 5

    path = rplidar_provider._create_straight_path_from_angle(
        angle_deg, length, num_points
    )

    assert path.shape == (2, num_points)

    x_vals, y_vals = path
    # At 90 degrees, y should be ~0, x should go from 0 to length
    expected_x = np.linspace(0.0, length, num_points)
    expected_y = np.zeros(num_points)

    np.testing.assert_allclose(x_vals, expected_x, atol=1e-10)
    np.testing.assert_allclose(y_vals, expected_y, atol=1e-10)


def test_initialize_paths(rplidar_provider):
    """
    Test _initialize_paths creates the correct number of paths with correct shapes.
    """
    # Mock path_angles to have predictable values for easier testing
    rplidar_provider.path_angles = [0, 90]
    paths = rplidar_provider._initialize_paths()

    # Should create 2 paths based on path_angles
    assert len(paths) == 2

    # Each path should be a numpy array of shape (2, num_points_default)
    for path in paths:
        assert isinstance(path, np.ndarray)
        # Default num_points in _create_straight_path_from_angle is 30
        assert path.shape == (2, 30)


def test_generate_movement_string_no_valid_paths(rplidar_provider):
    """
    Test _generate_movement_string when no paths are valid.
    """
    valid_paths = []
    expected_string = "You are surrounded by objects and cannot safely move in any direction. DO NOT MOVE."
    result = rplidar_provider._generate_movement_string(valid_paths)
    assert result == expected_string


def test_generate_movement_string_with_turns_and_advance(rplidar_provider):
    """
    Test _generate_movement_string with a mix of valid paths (turns, advance).
    Assumes default use_zenoh=False and machine_type="go2".
    """
    # Simulate state where paths 0, 4, 7 are valid -> turn_left (0), advance (4), turn_right (7)
    rplidar_provider.turn_left = [0]
    rplidar_provider.advance = [4]
    rplidar_provider.turn_right = [7]
    rplidar_provider.retreat = False
    valid_paths = [0, 4, 7]

    result = rplidar_provider._generate_movement_string(valid_paths)

    # The exact string depends on the order parts are appended.
    # It should contain the safe directions and end with 'stand still'.
    # A simplified check:
    assert "safe movement directions are" in result.lower()
    assert "'turn left'" in result
    assert "'move forwards'" in result
    assert "'turn right'" in result
    assert "'stand still'" in result


def test_update_filename(rplidar_provider):
    """
    Test update_filename generates a string with correct format.
    Uses mocking to control the time output.
    """
    test_time = 1700000000.123456
    expected_suffix = "1700000000_123456Z.jsonl"
    with patch("time.time", return_value=test_time):
        filename = rplidar_provider.update_filename()

    assert filename.endswith(expected_suffix)
    assert filename.startswith("dump/lidar_")


# --- Add more tests for other scenarios of _generate_movement_string if needed ---
# e.g., for retreat, for turtlebot4 specific string, etc.
# You might need to adjust provider.use_zenoh and provider.machine_type inside the test.
