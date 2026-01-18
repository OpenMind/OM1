"""Unit tests for FestivalProvider."""
from datetime import date, datetime, timedelta
from unittest.mock import patch
import pytest
from providers.festival_provider import FestivalProvider

@pytest.fixture(autouse=True)
def reset_singleton():
    if hasattr(FestivalProvider, "reset"):
        FestivalProvider.reset()
    yield
    if hasattr(FestivalProvider, "reset"):
        FestivalProvider.reset()

def test_singleton_instance():
    instance1 = FestivalProvider()
    instance2 = FestivalProvider()
    assert instance1 is instance2

def test_festivals_list_initialized():
    provider = FestivalProvider()
    assert isinstance(provider.festivals, list)
    assert len(provider.festivals) > 0

@patch("providers.festival_provider.datetime")
def test_get_today_festivals(mock_datetime):
    test_date = date(2025, 1, 29)
    mock_datetime.now.return_value.date.return_value = test_date
    mock_datetime.strptime = datetime.strptime
    provider = FestivalProvider()
    today_festivals = provider.get_today_festivals()
    assert isinstance(today_festivals, list)

@patch("providers.festival_provider.datetime")
def test_get_upcoming_festivals_default(mock_datetime):
    test_date = date(2025, 1, 22)
    mock_datetime.now.return_value.date.return_value = test_date
    mock_datetime.strptime = datetime.strptime
    provider = FestivalProvider()
    upcoming = provider.get_upcoming_festivals()
    assert isinstance(upcoming, list)
    for festival in upcoming:
        assert "days_until" in festival
        assert 0 < festival["days_until"] <= 7

@patch("providers.festival_provider.datetime")
def test_get_upcoming_festivals_custom_days(mock_datetime):
    test_date = date(2025, 1, 1)
    mock_datetime.now.return_value.date.return_value = test_date
    mock_datetime.strptime = datetime.strptime
    provider = FestivalProvider()
    upcoming = provider.get_upcoming_festivals(days_ahead=30)
    assert isinstance(upcoming, list)

@patch("providers.festival_provider.datetime")
def test_get_reminder_festivals(mock_datetime):
    provider = FestivalProvider()
    if provider.festivals:
        festival = provider.festivals[0]
        festival_date = datetime.strptime(festival["date"], "%Y-%m-%d").date()
        reminder_days = festival.get("reminder_days", [7, 3, 1])
        if reminder_days:
            test_date = festival_date - timedelta(days=reminder_days[0])
            mock_datetime.now.return_value.date.return_value = test_date
            mock_datetime.strptime = datetime.strptime
            reminder_festivals = provider.get_reminder_festivals()
            assert isinstance(reminder_festivals, list)

def test_add_custom_festival_default():
    provider = FestivalProvider()
    initial_count = len(provider.festivals)
    provider.add_custom_festival(
        name="Custom Festival",
        english_name="Custom",
        festival_type="custom_test",
        date="2025-06-15",
    )
    assert len(provider.festivals) == initial_count + 1

def test_get_festival_by_type_existing():
    provider = FestivalProvider()
    if provider.festivals:
        existing_type = provider.festivals[0]["type"]
        festival = provider.get_festival_by_type(existing_type)
        assert festival is not None

def test_get_festival_by_type_nonexistent():
    provider = FestivalProvider()
    festival = provider.get_festival_by_type("nonexistent_type_xyz")
    assert festival is None

@patch("providers.festival_provider.datetime")
def test_get_upcoming_festivals_zero_days(mock_datetime):
    test_date = date(2025, 1, 1)
    mock_datetime.now.return_value.date.return_value = test_date
    mock_datetime.strptime = datetime.strptime
    provider = FestivalProvider()
    upcoming = provider.get_upcoming_festivals(days_ahead=0)
    assert upcoming == []
