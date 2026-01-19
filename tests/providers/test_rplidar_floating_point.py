"""Tests for floating-point edge cases in RPLidar provider."""

import ast
import math
from pathlib import Path

import pytest

from providers.rplidar_provider import RPLidarProvider


@pytest.fixture
def rplidar_provider():
    actual_class = RPLidarProvider._singleton_class  # type: ignore
    provider = actual_class.__new__(actual_class)
    return provider


class TestFloatingPointSourceAnalysis:

    def test_distance_function_uses_epsilon_comparison(self):
        source_file = Path("src/providers/rplidar_provider.py")
        assert source_file.exists(), "rplidar_provider.py not found"

        source_code = source_file.read_text()
        tree = ast.parse(source_code)

        target_func = None
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "distance_point_to_line_segment"
            ):
                target_func = node
                break

        assert target_func is not None, "distance_point_to_line_segment not found"

        has_exact_zero_comparison = False
        for node in ast.walk(target_func):
            if isinstance(node, ast.Compare):
                if len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq):
                    for comparator in node.comparators:
                        if (
                            isinstance(comparator, ast.Constant)
                            and comparator.value == 0
                        ):
                            has_exact_zero_comparison = True
                            break

        assert (
            not has_exact_zero_comparison
        ), "distance_point_to_line_segment uses exact zero comparison (== 0)"


class TestFloatingPointBehavior:

    def test_near_zero_division_causes_overflow(self):
        dx = 1e-16
        dy = 0.0

        exact_zero_check = dx == 0 and dy == 0
        assert not exact_zero_check

        denominator = dx * dx + dy * dy
        assert denominator < 1e-30

        result = 1.0 / denominator
        assert result > 1e30

    def test_epsilon_comparison_catches_near_zero(self):
        dx = 1e-16
        dy = 0.0
        EPSILON = 1e-10

        exact_check = dx == 0 and dy == 0
        assert not exact_check

        epsilon_check = abs(dx) < EPSILON and abs(dy) < EPSILON
        assert epsilon_check

    def test_denominator_check_is_safer(self):
        dx = 1e-16
        dy = 0.0
        EPSILON = 1e-10

        denominator = dx * dx + dy * dy
        assert denominator < EPSILON


class TestDistancePointToLineSegment:

    def test_normal_line_segment(self, rplidar_provider):
        result = rplidar_provider.distance_point_to_line_segment(
            px=0.0, py=1.0, x1=0.0, y1=0.0, x2=2.0, y2=0.0
        )
        assert result == pytest.approx(1.0, rel=1e-9)

    def test_point_on_line_segment(self, rplidar_provider):
        result = rplidar_provider.distance_point_to_line_segment(
            px=1.0, py=0.0, x1=0.0, y1=0.0, x2=2.0, y2=0.0
        )
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_zero_length_segment(self, rplidar_provider):
        result = rplidar_provider.distance_point_to_line_segment(
            px=3.0, py=4.0, x1=0.0, y1=0.0, x2=0.0, y2=0.0
        )
        assert result == pytest.approx(5.0, rel=1e-9)

    def test_near_zero_segment_no_crash(self, rplidar_provider):
        result = rplidar_provider.distance_point_to_line_segment(
            px=1.0, py=1.0, x1=0.0, y1=0.0, x2=1e-16, y2=0.0
        )
        assert not math.isnan(result)
        assert not math.isinf(result)
        assert result == pytest.approx(math.sqrt(2), rel=0.01)

    def test_extremely_small_segment_no_crash(self, rplidar_provider):
        result = rplidar_provider.distance_point_to_line_segment(
            px=1.0, py=1.0, x1=0.0, y1=0.0, x2=1e-200, y2=1e-200
        )
        assert not math.isnan(result)
        assert not math.isinf(result)
        expected = math.sqrt(1.0**2 + 1.0**2)
        assert result == pytest.approx(expected, rel=0.01)

    def test_diagonal_line_segment(self, rplidar_provider):
        result = rplidar_provider.distance_point_to_line_segment(
            px=0.0, py=1.0, x1=0.0, y1=0.0, x2=1.0, y2=1.0
        )
        expected = math.sqrt(2) / 2
        assert result == pytest.approx(expected, rel=1e-6)

    def test_point_beyond_segment_end(self, rplidar_provider):
        result = rplidar_provider.distance_point_to_line_segment(
            px=5.0, py=0.0, x1=0.0, y1=0.0, x2=2.0, y2=0.0
        )
        assert result == pytest.approx(3.0, rel=1e-9)
