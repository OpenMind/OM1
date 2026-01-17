# tests/providers/test_simple_paths_provider.py

import pytest
from src.providers.simple_paths_provider import SimplePathsProvider


@pytest.fixture
def simple_paths_provider():
    """
    Fixture to create a SimplePathsProvider instance for testing.
    Uses _singleton_class to get the original class and __new__ to avoid running __init__.
    """
    original_class = SimplePathsProvider._singleton_class
    provider = original_class.__new__(original_class)
    # Initialize instance variables that _generate_movement_string relies on
    provider.turn_left = []
    provider.turn_right = []
    provider.advance = []
    provider.retreat = False
    return provider


def test_generate_movement_string_all_options(simple_paths_provider):
    """Test string generation when all movement options are present."""
    simple_paths_provider.turn_left = [0, 1, 2]
    simple_paths_provider.advance = [3, 4, 5]
    simple_paths_provider.turn_right = [6, 7, 8]
    simple_paths_provider.retreat = True

    expected = "The safe movement directions are: {'turn left', 'move forwards', 'turn right', 'move back', 'stand still'}. "
    # Pass a non-empty list to bypass the initial "if not valid_paths:" check
    result = simple_paths_provider._generate_movement_string(["dummy"])
    assert result == expected


def test_generate_movement_string_only_turn_left(simple_paths_provider):
    """Test string generation when only turn_left is populated."""
    simple_paths_provider.turn_left = [0, 1]
    # advance, turn_right, retreat remain as initialized (empty list, False)

    expected = "The safe movement directions are: {'turn left', 'stand still'}. "
    # Pass a non-empty list to bypass the initial "if not valid_paths:" check
    result = simple_paths_provider._generate_movement_string(["dummy"])
    assert result == expected


def test_generate_movement_string_only_advance(simple_paths_provider):
    """Test string generation when only advance is populated."""
    simple_paths_provider.advance = [3, 4, 5]

    expected = "The safe movement directions are: {'move forwards', 'stand still'}. "
    # Pass a non-empty list to bypass the initial "if not valid_paths:" check
    result = simple_paths_provider._generate_movement_string(["dummy"])
    assert result == expected


def test_generate_movement_string_only_turn_right(simple_paths_provider):
    """Test string generation when only turn_right is populated."""
    simple_paths_provider.turn_right = [6, 7, 8]

    expected = "The safe movement directions are: {'turn right', 'stand still'}. "
    # Pass a non-empty list to bypass the initial "if not valid_paths:" check
    result = simple_paths_provider._generate_movement_string(["dummy"])
    assert result == expected


def test_generate_movement_string_only_retreat(simple_paths_provider):
    """Test string generation when only retreat is True."""
    simple_paths_provider.retreat = True

    expected = "The safe movement directions are: {'move back', 'stand still'}. "
    # Pass a non-empty list to bypass the initial "if not valid_paths:" check
    result = simple_paths_provider._generate_movement_string(["dummy"])
    assert result == expected


def test_generate_movement_string_no_options(simple_paths_provider):
    """Test string generation when no movement options are present (empty lists, False)."""
    # All values are already initialized to [] or False in the fixture
    # This test specifically checks the "if not valid_paths:" branch.
    # So, we pass an empty list to trigger it.
    expected = "You are surrounded by objects and cannot safely move in any direction. DO NOT MOVE."
    result = simple_paths_provider._generate_movement_string([])
    assert result == expected


def test_generate_movement_string_none_paths(simple_paths_provider):
    """Test behavior when _valid_paths is None (though logic might not reach this string generation path directly)."""
    # This test focuses on the string generation logic itself.
    # If _valid_paths is None, the initial condition `if not valid_paths:` should trigger.
    # We test the string generation part assuming the internal state determines the output *after* the if check passes.
    # To test the internal state logic, pass a non-empty list.

    # Let's set internal state to non-empty, and pass a non-empty list to bypass 'if not valid_paths'.
    simple_paths_provider.advance = [3, 4, 5]
    expected_with_internal_state = "The safe movement directions are: {'move forwards', 'stand still'}. "
    result = simple_paths_provider._generate_movement_string(["dummy"]) # Pass non-empty list
    assert result == expected_with_internal_state

    # Now, let's reset internal state to empty, but still pass a non-empty list to bypass the first check.
    # This should result in the message with only 'stand still'.
    simple_paths_provider.turn_left = []
    simple_paths_provider.turn_right = []
    simple_paths_provider.advance = []
    simple_paths_provider.retreat = False
    expected_only_stand_still = "The safe movement directions are: {'stand still'}. "
    result_only_stand_still = simple_paths_provider._generate_movement_string(["dummy"]) # Pass non-empty list
    assert result_only_stand_still == expected_only_stand_still

    # Now, test the "surrounded" message by passing an empty list.
    expected_surrounded = "You are surrounded by objects and cannot safely move in any direction. DO NOT MOVE."
    result_surrounded = simple_paths_provider._generate_movement_string([]) # Pass empty list
    assert result_surrounded == expected_surrounded
