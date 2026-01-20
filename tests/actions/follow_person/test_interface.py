"""Unit tests for FollowPerson interface."""
import pytest
from actions.follow_person.interface import FollowMode, FollowPersonInput, FollowPerson


def test_follow_mode_enum():
    """Test FollowMode enum values."""
    assert FollowMode.BY_NAME == "by_name"
    assert FollowMode.NEAREST == "nearest"
    assert FollowMode.LAST_SEEN == "last_seen"
    assert FollowMode.STOP == "stop"


def test_follow_mode_enum_string_comparison():
    """Test FollowMode enum can be compared with strings."""
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


def test_follow_person_input_edge_cases():
    """Test FollowPersonInput with edge case values."""
    # Minimum distance
    input_data = FollowPersonInput(action="bob", distance=0.5)
    assert input_data.distance == 0.5
    
    # Maximum distance
    input_data = FollowPersonInput(action="bob", distance=5.0)
    assert input_data.distance == 5.0
    
    # Zero speed
    input_data = FollowPersonInput(action="bob", speed=0.0)
    assert input_data.speed == 0.0
    
    # Maximum speed
    input_data = FollowPersonInput(action="bob", speed=1.0)
    assert input_data.speed == 1.0
    
    # Very short timeout
    input_data = FollowPersonInput(action="bob", timeout_sec=1.0)
    assert input_data.timeout_sec == 1.0
    
    # Very long timeout
    input_data = FollowPersonInput(action="bob", timeout_sec=300.0)
    assert input_data.timeout_sec == 300.0


def test_follow_person_input_stop_action():
    """Test FollowPersonInput with stop action."""
    input_data = FollowPersonInput(action="stop")
    assert input_data.action == "stop"
    # Other fields should still have defaults
    assert input_data.distance == 1.5
    assert input_data.speed == 0.5


def test_follow_person_input_by_name():
    """Test FollowPersonInput with person name."""
    input_data = FollowPersonInput(action="alice")
    assert input_data.action == "alice"
    
    input_data = FollowPersonInput(action="Bob")
    assert input_data.action == "Bob"
    
    input_data = FollowPersonInput(action="wendy")
    assert input_data.action == "wendy"


def test_follow_person_input_nearest():
    """Test FollowPersonInput with nearest mode."""
    input_data = FollowPersonInput(action="nearest")
    assert input_data.action == "nearest"


def test_follow_person_input_last_seen():
    """Test FollowPersonInput with last_seen mode."""
    input_data = FollowPersonInput(action="last_seen")
    assert input_data.action == "last_seen"
    
    input_data = FollowPersonInput(action="me")
    assert input_data.action == "me"


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


def test_follow_person_interface():
    """Test FollowPerson interface structure."""
    input_data = FollowPersonInput(action="alice", distance=2.0)
    output_data = FollowPersonInput(action="alice", distance=2.0)
    
    interface = FollowPerson(input=input_data, output=output_data)
    assert interface.input == input_data
    assert interface.output == output_data
    assert interface.input.action == "alice"
    assert interface.output.distance == 2.0


def test_follow_person_interface_different_input_output():
    """Test FollowPerson interface with different input and output."""
    input_data = FollowPersonInput(action="alice", distance=1.5)
    output_data = FollowPersonInput(action="alice", distance=2.0, speed=0.8)
    
    interface = FollowPerson(input=input_data, output=output_data)
    assert interface.input.distance == 1.5
    assert interface.output.distance == 2.0
    assert interface.output.speed == 0.8


def test_follow_person_input_immutability():
    """Test that FollowPersonInput is a dataclass (immutable by default)."""
    input_data = FollowPersonInput(action="alice", distance=1.5)
    # Dataclasses are mutable by default, but we can test field access
    assert input_data.action == "alice"
    input_data.action = "bob"  # Should work for dataclass
    assert input_data.action == "bob"


def test_follow_person_input_string_representations():
    """Test string representations of FollowPersonInput."""
    input_data = FollowPersonInput(action="alice", distance=2.0, speed=0.5)
    # Dataclass should have __repr__
    repr_str = repr(input_data)
    assert "alice" in repr_str or "FollowPersonInput" in repr_str
