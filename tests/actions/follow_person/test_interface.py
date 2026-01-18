"""Unit tests for FollowPerson interface."""
import pytest
from actions.follow_person.interface import FollowMode, FollowPersonInput

def test_follow_mode_enum():
    """Test FollowMode enum values."""
    assert FollowMode.BY_NAME == "by_name"
    assert FollowMode.NEAREST == "nearest"
    assert FollowMode.LAST_SEEN == "last_seen"
    assert FollowMode.STOP == "stop"

def test_follow_person_input_defaults():
    """Test FollowPersonInput with defaults."""
    input_data = FollowPersonInput(action="alice")
    assert input_data.action == "alice"
    assert input_data.distance == 1.5
    assert input_data.speed == 0.5
    assert input_data.stop_on_arrival is True
    assert input_data.timeout_sec == 30.0

def test_follow_person_input_all_fields():
    """Test FollowPersonInput with all fields."""
    input_data = FollowPersonInput(
        action="nearest",
        distance=2.0,
        speed=0.7,
        stop_on_arrival=False,
        timeout_sec=60.0
    )
    assert input_data.action == "nearest"
    assert input_data.distance == 2.0
    assert input_data.speed == 0.7
    assert input_data.stop_on_arrival is False
    assert input_data.timeout_sec == 60.0

def test_all_follow_modes():
    """Test all follow modes are accessible."""
    modes = [
        FollowMode.BY_NAME,
        FollowMode.NEAREST,
        FollowMode.LAST_SEEN,
        FollowMode.STOP,
    ]
    assert len(modes) == 4
    for mode in modes:
        assert isinstance(mode, str)
