import threading
from unittest.mock import MagicMock

import pytest

from providers.simple_paths_provider import SimplePathsProvider


@pytest.fixture
def simple_paths_provider():
    """
    Fixture to create a SimplePathsProvider instance for testing.
    """
    actual_class = SimplePathsProvider._singleton_class  # type: ignore
    provider = actual_class.__new__(actual_class)
    provider.turn_left = []
    provider.turn_right = []
    provider.advance = []
    provider.retreat = False
    provider._valid_paths = []
    provider._lidar_string = ""
    return provider


def test_generate_movement_string_all_options(simple_paths_provider):
    """Test string generation when all movement options are present."""
    simple_paths_provider.turn_left = [0, 1, 2]
    simple_paths_provider.advance = [3, 4, 5]
    simple_paths_provider.turn_right = [6, 7, 8]
    simple_paths_provider.retreat = True

    expected = "The safe movement directions are: {'turn left', 'move forwards', 'turn right', 'move back', 'stand still'}. "
    result = simple_paths_provider._generate_movement_string(["dummy"])
    assert result == expected


def test_generate_movement_string_only_turn_left(simple_paths_provider):
    """Test string generation when only turn_left is populated."""
    simple_paths_provider.turn_left = [0, 1]

    expected = "The safe movement directions are: {'turn left', 'stand still'}. "
    result = simple_paths_provider._generate_movement_string(["dummy"])
    assert result == expected


def test_generate_movement_string_only_advance(simple_paths_provider):
    """Test string generation when only advance is populated."""
    simple_paths_provider.advance = [3, 4, 5]

    expected = "The safe movement directions are: {'move forwards', 'stand still'}. "
    result = simple_paths_provider._generate_movement_string(["dummy"])
    assert result == expected


def test_generate_movement_string_only_turn_right(simple_paths_provider):
    """Test string generation when only turn_right is populated."""
    simple_paths_provider.turn_right = [6, 7, 8]

    expected = "The safe movement directions are: {'turn right', 'stand still'}. "
    result = simple_paths_provider._generate_movement_string(["dummy"])
    assert result == expected


def test_generate_movement_string_only_retreat(simple_paths_provider):
    """Test string generation when only retreat is True."""
    simple_paths_provider.retreat = True

    expected = "The safe movement directions are: {'move back', 'stand still'}. "
    result = simple_paths_provider._generate_movement_string(["dummy"])
    assert result == expected


def test_generate_movement_string_no_options(simple_paths_provider):
    """Test string generation when no movement options are present (empty lists, False)."""
    expected = "You are surrounded by objects and cannot safely move in any direction. DO NOT MOVE."
    result = simple_paths_provider._generate_movement_string([])
    assert result == expected


def test_generate_movement_string_none_paths(simple_paths_provider):
    """Test behavior when _valid_paths is None (though logic might not reach this string generation path directly)."""
    simple_paths_provider.advance = [3, 4, 5]
    expected_with_internal_state = (
        "The safe movement directions are: {'move forwards', 'stand still'}. "
    )
    result = simple_paths_provider._generate_movement_string(["dummy"])
    assert result == expected_with_internal_state

    simple_paths_provider.turn_left = []
    simple_paths_provider.turn_right = []
    simple_paths_provider.advance = []
    simple_paths_provider.retreat = False
    expected_only_stand_still = "The safe movement directions are: {'stand still'}. "
    result_only_stand_still = simple_paths_provider._generate_movement_string(["dummy"])
    assert result_only_stand_still == expected_only_stand_still

    expected_surrounded = "You are surrounded by objects and cannot safely move in any direction. DO NOT MOVE."
    result_surrounded = simple_paths_provider._generate_movement_string([])
    assert result_surrounded == expected_surrounded


def test_movement_options(simple_paths_provider):
    """Test movement_options property returns correct dict structure."""
    simple_paths_provider.turn_left = [0, 1]
    simple_paths_provider.advance = [3, 4, 5]
    simple_paths_provider.turn_right = [7, 8]
    simple_paths_provider.retreat = True

    result = simple_paths_provider.movement_options
    assert result == {
        "turn_left": [0, 1],
        "advance": [3, 4, 5],
        "turn_right": [7, 8],
        "retreat": True,
    }


def test_movement_options_empty(simple_paths_provider):
    """Test movement_options property with default empty values."""
    result = simple_paths_provider.movement_options
    assert result == {
        "turn_left": [],
        "advance": [],
        "turn_right": [],
        "retreat": False,
    }


def test_stop(simple_paths_provider):
    """Test stop method sets stop event, sends STOP to queue, and joins threads."""
    simple_paths_provider._stop_event = threading.Event()
    mock_queue = MagicMock()
    simple_paths_provider.control_queue = mock_queue

    mock_processor_thread = MagicMock()
    mock_derived_thread = MagicMock()
    simple_paths_provider._simple_paths_processor_thread = mock_processor_thread
    simple_paths_provider._simple_paths_derived_thread = mock_derived_thread

    simple_paths_provider.stop()

    assert simple_paths_provider._stop_event.is_set()
    mock_queue.put.assert_called_once_with("STOP")
    mock_processor_thread.join.assert_called_once()
    mock_derived_thread.join.assert_called_once()
