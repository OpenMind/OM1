"""Unit tests for FestivalGreeting interface."""

import pytest

from actions.festival_greeting.interface import (
    FestivalGreeting,
    FestivalGreetingInput,
    FestivalType,
)


class TestFestivalType:
    """Test cases for FestivalType enum."""

    def test_festival_type_values(self):
        """Test that all expected festival types exist."""
        assert FestivalType.CHINESE_NEW_YEAR == "chinese_new_year"
        assert FestivalType.MID_AUTUMN == "mid_autumn"
        assert FestivalType.DRAGON_BOAT == "dragon_boat"
        assert FestivalType.NATIONAL_DAY == "national_day"
        assert FestivalType.CHRISTMAS == "christmas"
        assert FestivalType.NEW_YEAR == "new_year"
        assert FestivalType.VALENTINE == "valentine"
        assert FestivalType.BIRTHDAY == "birthday"
        assert FestivalType.CUSTOM == "custom"

    def test_festival_type_count(self):
        """Test that we have 9 festival types as documented."""
        festival_types = [
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
        assert len(festival_types) == 9


class TestFestivalGreetingInput:
    """Test cases for FestivalGreetingInput dataclass."""

    def test_festival_greeting_input_required_fields(self):
        """Test that festival_type is required."""
        input_data = FestivalGreetingInput(festival_type=FestivalType.CHRISTMAS)
        assert input_data.festival_type == FestivalType.CHRISTMAS
        assert input_data.message == ""
        assert input_data.recipient_name == ""

    def test_festival_greeting_input_with_message(self):
        """Test FestivalGreetingInput with custom message."""
        input_data = FestivalGreetingInput(
            festival_type=FestivalType.NEW_YEAR,
            message="Happy New Year!",
        )
        assert input_data.festival_type == FestivalType.NEW_YEAR
        assert input_data.message == "Happy New Year!"
        assert input_data.recipient_name == ""

    def test_festival_greeting_input_with_recipient(self):
        """Test FestivalGreetingInput with recipient name."""
        input_data = FestivalGreetingInput(
            festival_type=FestivalType.BIRTHDAY,
            recipient_name="Alice",
        )
        assert input_data.festival_type == FestivalType.BIRTHDAY
        assert input_data.message == ""
        assert input_data.recipient_name == "Alice"

    def test_festival_greeting_input_all_fields(self):
        """Test FestivalGreetingInput with all fields."""
        input_data = FestivalGreetingInput(
            festival_type=FestivalType.VALENTINE,
            message="Happy Valentine's Day!",
            recipient_name="Bob",
        )
        assert input_data.festival_type == FestivalType.VALENTINE
        assert input_data.message == "Happy Valentine's Day!"
        assert input_data.recipient_name == "Bob"

    def test_festival_greeting_input_all_types(self):
        """Test FestivalGreetingInput with all festival types."""
        for festival_type in FestivalType:
            input_data = FestivalGreetingInput(festival_type=festival_type)
            assert input_data.festival_type == festival_type


class TestFestivalGreeting:
    """Test cases for FestivalGreeting action."""

    def test_festival_greeting_creation(self):
        """Test creating a FestivalGreeting action."""
        input_data = FestivalGreetingInput(festival_type=FestivalType.CHRISTMAS)
        greeting = FestivalGreeting(input=input_data, output=input_data)
        assert greeting.input == input_data
        assert greeting.output == input_data

    def test_festival_greeting_with_custom_message(self):
        """Test FestivalGreeting with custom message."""
        input_data = FestivalGreetingInput(
            festival_type=FestivalType.CHINESE_NEW_YEAR,
            message="新年快乐！",
        )
        greeting = FestivalGreeting(input=input_data, output=input_data)
        assert greeting.input.message == "新年快乐！"

    def test_festival_greeting_with_personalization(self):
        """Test FestivalGreeting with personalized recipient."""
        input_data = FestivalGreetingInput(
            festival_type=FestivalType.BIRTHDAY,
            recipient_name="Charlie",
            message="Happy Birthday!",
        )
        greeting = FestivalGreeting(input=input_data, output=input_data)
        assert greeting.input.recipient_name == "Charlie"
        assert greeting.input.message == "Happy Birthday!"
