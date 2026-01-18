"""Unit tests for PostureDetection interface."""
import pytest
from actions.posture_detection.interface import (
    PostureDetection,
    PostureDetectionInput,
    PostureSeverity,
    PostureType,
)

def test_posture_type_enum():
    """Test PostureType enum values."""
    assert PostureType.GOOD == "good"
    assert PostureType.SLUMPED == "slumped"
    assert PostureType.LEANING == "leaning"
    assert PostureType.HUNCHED == "hunched"
    assert PostureType.ASYMMETRIC == "asymmetric"
    assert PostureType.LAYING == "laying"

def test_posture_severity_enum():
    """Test PostureSeverity enum values."""
    assert PostureSeverity.MILD == "mild"
    assert PostureSeverity.MODERATE == "moderate"
    assert PostureSeverity.SEVERE == "severe"

def test_posture_detection_input_defaults():
    """Test PostureDetectionInput with defaults."""
    input_data = PostureDetectionInput(
        posture_type=PostureType.SLUMPED,
        severity=PostureSeverity.MODERATE
    )
    assert input_data.posture_type == PostureType.SLUMPED
    assert input_data.severity == PostureSeverity.MODERATE
    assert input_data.duration_minutes == 0.0
    assert input_data.person_name == ""
    assert input_data.recommendation == ""

def test_posture_detection_input_all_fields():
    """Test PostureDetectionInput with all fields."""
    input_data = PostureDetectionInput(
        posture_type=PostureType.HUNCHED,
        severity=PostureSeverity.SEVERE,
        duration_minutes=45.5,
        person_name="Alice",
        recommendation="Sit up straight and take a break"
    )
    assert input_data.posture_type == PostureType.HUNCHED
    assert input_data.severity == PostureSeverity.SEVERE
    assert input_data.duration_minutes == 45.5
    assert input_data.person_name == "Alice"
    assert input_data.recommendation == "Sit up straight and take a break"

def test_posture_detection_interface():
    """Test PostureDetection interface structure."""
    input_data = PostureDetectionInput(
        posture_type=PostureType.GOOD,
        severity=PostureSeverity.MILD
    )
    detection = PostureDetection(input=input_data, output=input_data)
    assert detection.input == input_data
    assert detection.output == input_data

def test_all_posture_types():
    """Test all posture types are accessible."""
    types = [
        PostureType.GOOD,
        PostureType.SLUMPED,
        PostureType.LEANING,
        PostureType.HUNCHED,
        PostureType.ASYMMETRIC,
        PostureType.LAYING,
    ]
    assert len(types) == 6
    for posture_type in types:
        assert isinstance(posture_type, str)

def test_all_severity_levels():
    """Test all severity levels are accessible."""
    severities = [
        PostureSeverity.MILD,
        PostureSeverity.MODERATE,
        PostureSeverity.SEVERE,
    ]
    assert len(severities) == 3
    for severity in severities:
        assert isinstance(severity, str)
