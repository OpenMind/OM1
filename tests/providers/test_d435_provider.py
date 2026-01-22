import math

import pytest

from providers.d435_provider import D435Provider


@pytest.fixture
def d435_provider():
    """
    Fixture to create a D435Provider instance for testing.
    """
    original_class = D435Provider._singleton_class  # type: ignore
    provider = original_class.__new__(original_class)
    return provider


def test_calculate_angle_and_distance_origin(d435_provider):
    """
    Test calculation when point is at the origin (0, 0).
    atan2(0, 0) is undefined, but Python returns 0.0.
    Distance should be 0.
    """
    world_x, world_y = 0.0, 0.0
    expected_angle = 0.0
    expected_distance = 0.0

    angle, distance = d435_provider.calculate_angle_and_distance(world_x, world_y)

    # Check distance first, as angle might be undefined at origin
    assert distance == expected_distance
    # Python's atan2 handles (0, 0) as 0.0 degrees
    assert angle == expected_angle


def test_calculate_angle_and_distance_positive_x_axis(d435_provider):
    """
    Test calculation for a point on the positive x-axis (e.g., 3, 0).
    Angle should be 0 degrees, distance is the x-value.
    """
    world_x, world_y = 3.0, 0.0
    expected_angle = 0.0
    expected_distance = 3.0

    angle, distance = d435_provider.calculate_angle_and_distance(world_x, world_y)

    assert angle == expected_angle
    assert distance == expected_distance


def test_calculate_angle_and_distance_positive_y_axis(d435_provider):
    """
    Test calculation for a point on the positive y-axis (e.g., 0, 4).
    Angle should be 90 degrees, distance is the y-value.
    """
    world_x, world_y = 0.0, 4.0
    expected_angle = 90.0
    expected_distance = 4.0

    angle, distance = d435_provider.calculate_angle_and_distance(world_x, world_y)

    assert angle == expected_angle
    assert distance == expected_distance


def test_calculate_angle_and_distance_negative_x_axis(d435_provider):
    """
    Test calculation for a point on the negative x-axis (e.g., -5, 0).
    Angle should be 180 degrees (or -180, depending on atan2 convention, Python returns +180).
    Distance is the absolute x-value.
    """
    world_x, world_y = -5.0, 0.0
    expected_angle = 180.0
    expected_distance = 5.0

    angle, distance = d435_provider.calculate_angle_and_distance(world_x, world_y)

    assert angle == expected_angle
    assert distance == expected_distance


def test_calculate_angle_and_distance_negative_y_axis(d435_provider):
    """
    Test calculation for a point on the negative y-axis (e.g., 0, -6).
    Angle should be -90 degrees.
    Distance is the absolute y-value.
    """
    world_x, world_y = 0.0, -6.0
    expected_angle = -90.0
    expected_distance = 6.0

    angle, distance = d435_provider.calculate_angle_and_distance(world_x, world_y)

    assert angle == expected_angle
    assert distance == expected_distance


def test_calculate_angle_and_distance_quadrant_1(d435_provider):
    """
    Test calculation for a point in Quadrant I (positive x, positive y).
    e.g., (1, 1) -> angle 45 degrees.
    """
    world_x, world_y = 1.0, 1.0
    expected_angle = 45.0
    expected_distance = math.sqrt(2)

    angle, distance = d435_provider.calculate_angle_and_distance(world_x, world_y)

    assert math.isclose(angle, expected_angle, abs_tol=1e-10)
    assert math.isclose(distance, expected_distance, abs_tol=1e-10)


def test_calculate_angle_and_distance_arbitrary_point(d435_provider):
    """
    Test calculation for an arbitrary point (e.g., 3, 4).
    Angle should be atan2(4, 3) in degrees, distance is 5 (3-4-5 triangle).
    """
    world_x, world_y = 3.0, 4.0
    expected_angle = math.degrees(math.atan2(world_y, world_x))
    expected_distance = 5.0

    angle, distance = d435_provider.calculate_angle_and_distance(world_x, world_y)

    assert math.isclose(angle, expected_angle, abs_tol=1e-10)
    assert math.isclose(distance, expected_distance, abs_tol=1e-10)
