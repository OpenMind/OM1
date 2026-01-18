"""Unit tests for FestivalGreeting interface."""
import pytest
from actions.festival_greeting.interface import FestivalGreeting, FestivalGreetingInput, FestivalType

def test_festival_type_enum():
    """Test FestivalType enum values."""
    assert FestivalType.CHINESE_NEW_YEAR == "chinese_new_year"
    assert FestivalType.MID_AUTUMN == "mid_autumn"
    assert FestivalType.CHRISTMAS == "christmas"
    assert FestivalType.NEW_YEAR == "new_year"

def test_festival_greeting_input_defaults():
    """Test FestivalGreetingInput with defaults."""
    input_data = FestivalGreetingInput(festival_type=FestivalType.CHRISTMAS)
    assert input_data.festival_type == FestivalType.CHRISTMAS
    assert input_data.message == ""
    assert input_data.recipient_name == ""

def test_festival_greeting_input_all_fields():
    """Test FestivalGreetingInput with all fields."""
    input_data = FestivalGreetingInput(
        festival_type=FestivalType.CHINESE_NEW_YEAR,
        message="Happy New Year!",
        recipient_name="Alice"
    )
    assert input_data.festival_type == FestivalType.CHINESE_NEW_YEAR
    assert input_data.message == "Happy New Year!"
    assert input_data.recipient_name == "Alice"

def test_festival_greeting_interface():
    """Test FestivalGreeting interface structure."""
    input_data = FestivalGreetingInput(festival_type=FestivalType.BIRTHDAY)
    greeting = FestivalGreeting(input=input_data, output=input_data)
    assert greeting.input == input_data
    assert greeting.output == input_data
    assert greeting.input.festival_type == FestivalType.BIRTHDAY

def test_all_festival_types():
    """Test all festival types are accessible."""
    types = [
        FestivalType.CHINESE_NEW_YEAR,
        FestivalType.MID_AUTUMN,
        FestivalType.DRAGON_BOAT,
        FestivalType.NATIONAL_DAY,
        FestivalType.CHRISTMAS,
        FestivalType.NEW_YEAR,
        FestivalType.VALENTINE,
        FestivalType.BIRTHDAY,
        FestivalType.CUSTOM,
    ]
    assert len(types) == 9
    for festival_type in types:
        assert isinstance(festival_type, str)
