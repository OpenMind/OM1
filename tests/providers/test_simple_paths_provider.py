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
