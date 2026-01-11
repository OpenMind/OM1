"""
Tests for floating-point edge cases in RPLidar provider.

These tests verify that near-zero floating-point comparisons are handled
safely to prevent division by very small numbers.
"""

import ast
from pathlib import Path


class TestFloatingPointSourceAnalysis:
    """Static analysis tests for floating-point comparisons."""

    def test_distance_function_uses_epsilon_comparison(self):
        """Test that distance_point_to_line_segment uses epsilon-based comparison.

        BUG: Using `dx == 0 and dy == 0` for floating-point comparison is dangerous.
        If dx = 1e-16, the check fails but dx*dx = 1e-32 causes division overflow.

        FIXED: Should use epsilon-based comparison like `abs(dx) < EPSILON`.
        """
        source_file = Path("src/providers/rplidar_provider.py")
        assert source_file.exists(), "rplidar_provider.py not found"

        source_code = source_file.read_text()
        tree = ast.parse(source_code)

        # Find the distance_point_to_line_segment function
        target_func = None
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "distance_point_to_line_segment"
            ):
                target_func = node
                break

        assert target_func is not None, "distance_point_to_line_segment not found"

        # Look for problematic pattern: `dx == 0` or `dy == 0`
        has_exact_zero_comparison = False

        for node in ast.walk(target_func):
            if isinstance(node, ast.Compare):
                # Check for pattern: variable == 0
                if len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq):
                    # Check if comparing to 0
                    for comparator in node.comparators:
                        if (
                            isinstance(comparator, ast.Constant)
                            and comparator.value == 0
                        ):
                            has_exact_zero_comparison = True
                            break

        assert not has_exact_zero_comparison, (
            "BUG FOUND: distance_point_to_line_segment uses exact zero comparison (== 0). "
            "This is dangerous for floating-point numbers. "
            "Should use epsilon-based comparison like abs(value) < EPSILON or check denominator."
        )


class TestFloatingPointBehavior:
    """Tests demonstrating the floating-point division issue."""

    def test_near_zero_division_causes_overflow(self):
        """Prove that dividing by near-zero causes extreme values."""
        dx = 1e-16
        dy = 0.0

        # This check would PASS (dx is not exactly 0)
        exact_zero_check = dx == 0 and dy == 0
        assert not exact_zero_check, "dx is not exactly zero"

        # But the denominator is extremely small
        denominator = dx * dx + dy * dy
        assert denominator < 1e-30, f"Denominator {denominator} is extremely small"

        # Division by this small number causes huge result
        numerator = 1.0
        result = numerator / denominator
        assert result > 1e30, "Division causes extreme value"

    def test_epsilon_comparison_catches_near_zero(self):
        """Prove that epsilon comparison catches near-zero values."""
        dx = 1e-16
        dy = 0.0
        EPSILON = 1e-10

        # Exact zero check FAILS
        exact_check = dx == 0 and dy == 0
        assert not exact_check

        # Epsilon check PASSES (catches the small value)
        epsilon_check = abs(dx) < EPSILON and abs(dy) < EPSILON
        assert epsilon_check, "Epsilon comparison should catch near-zero"

    def test_denominator_check_is_safer(self):
        """Prove that checking denominator directly is safer."""
        dx = 1e-16
        dy = 0.0
        EPSILON = 1e-10

        denominator = dx * dx + dy * dy

        # Denominator check catches the issue
        is_too_small = denominator < EPSILON
        assert is_too_small, "Denominator check should catch near-zero"
