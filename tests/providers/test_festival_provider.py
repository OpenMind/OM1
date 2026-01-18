"""Unit tests for FestivalProvider."""

from datetime import date, datetime, timedelta
from unittest.mock import patch

import pytest

from providers.festival_provider import FestivalProvider


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instance between tests."""
    if hasattr(FestivalProvider, "reset"):
        FestivalProvider.reset()
    yield
    if hasattr(FestivalProvider, "reset"):
        FestivalProvider.reset()


class TestFestivalProviderInitialization:
    """Test cases for FestivalProvider initialization."""

    def test_singleton_instance(self):
        """Test that FestivalProvider is a singleton."""
        instance1 = FestivalProvider()
        instance2 = FestivalProvider()
        assert instance1 is instance2

    def test_festivals_list_initialized(self):
        """Test that festivals list is initialized."""
        provider = FestivalProvider()
        assert isinstance(provider.festivals, list)
        assert len(provider.festivals) > 0


class TestGetTodayFestivals:
    """Test cases for get_today_festivals method."""

    @patch("providers.festival_provider.datetime")
    def test_get_today_festivals_when_festival_is_today(self, mock_datetime):
        """Test getting festivals that occur today."""
        test_date = date(2025, 1, 29)  # Chinese New Year example date
        mock_datetime.now.return_value.date.return_value = test_date
        mock_datetime.strptime = datetime.strptime

        provider = FestivalProvider()
        today_festivals = provider.get_today_festivals()

        assert isinstance(today_festivals, list)


class TestGetUpcomingFestivals:
    """Test cases for get_upcoming_festivals method."""

    @patch("providers.festival_provider.datetime")
    def test_get_upcoming_festivals_default_days(self, mock_datetime):
        """Test getting upcoming festivals with default 7 days."""
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
    def test_get_upcoming_festivals_empty_when_no_festivals(self, mock_datetime):
        """Test that upcoming festivals returns empty for past dates."""
        test_date = date(2026, 12, 31)
        mock_datetime.now.return_value.date.return_value = test_date
        mock_datetime.strptime = datetime.strptime

        provider = FestivalProvider()
        upcoming = provider.get_upcoming_festivals(days_ahead=7)

        # Should be empty or only future festivals
        for festival in upcoming:
            assert festival["days_until"] > 0


class TestAddCustomFestival:
    """Test cases for add_custom_festival method."""

    def test_add_custom_festival_default_reminder_days(self):
        """Test adding custom festival with default reminder days."""
        provider = FestivalProvider()
        initial_count = len(provider.festivals)

        provider.add_custom_festival(
            name="Custom Festival",
            english_name="Custom",
            festival_type="custom_test",
            date="2025-06-15",
        )

        assert len(provider.festivals) == initial_count + 1
        custom_festival = next(
            (f for f in provider.festivals if f["type"] == "custom_test"), None
        )
        assert custom_festival is not None
        assert custom_festival["reminder_days"] == [7, 3, 1]


class TestGetFestivalByType:
    """Test cases for get_festival_by_type method."""

    def test_get_festival_by_type_existing_type(self):
        """Test getting festival by existing type."""
        provider = FestivalProvider()
        if provider.festivals:
            existing_type = provider.festivals[0]["type"]
            festival = provider.get_festival_by_type(existing_type)

            assert festival is not None
            assert festival["type"] == existing_type

    def test_get_festival_by_type_nonexistent_type(self):
        """Test getting festival by nonexistent type returns None."""
        provider = FestivalProvider()
        festival = provider.get_festival_by_type("nonexistent_type_xyz")

        assert festival is None
e New Year
        mock_datetime.now.return_value.date.return_value = test_date
        mock_datetime.strptime = datetime.strptime

        provider = FestivalProvider()
        upcoming = provider.get_upcoming_festivals()

        assert isinstance(upcoming, list)
        # All festivals should have days_until field
        for festival in upcoming:
            assert "days_until" in festival
            assert 0 < festival["days_until"] <= 7

    @patch("providers.festival_provider.datetime")
    def test_get_upcoming_festivals_custom_days(self, mock_datetime):
        """Test getting upcoming festivals with custom days_ahead."""
        test_date = date(2025, 1, 1)
        mock_datetime.now.return_value.date.return_value = test_date
        mock_datetime.strptime = datetime.strptime

        provider = FestivalProvider()
        upcoming = provider.get_upcoming_festivals(days_ahead=30)

        assert isinstance(upcoming, list)
        for festival in upcoming:
            assert "days_until" in festival
            assert 0 < festival["days_until"] <= 30

    @patch("providers.festival_provider.datetime")
    def test_get_upcoming_festivals_sorted(self, mock_datetime):
        """Test that upcoming festivals are sorted by days_until."""
        test_date = date(2025, 1, 1)
        mock_datetime.now.return_value.date.return_value = test_date
        mock_datetime.strptime = datetime.strptime

        provider = FestivalProvider()
        upcoming = provider.get_upcoming_festivals(days_ahead=100)

        # Check if sorted (should be ascending by days_until)
        if len(upcoming) > 1:
            days_until_values = [f["days_until"] for f in upcoming]
            assert days_until_values == sorted(days_until_values)

    @patch("providers.festival_provider.datetime")
    def test_get_upcoming_festivals_excludes_past_festivals(self, mock_datetime):
        """Test that past festivals are excluded."""
        test_date = date(2025, 12, 31)  # After most festivals
        mock_datetime.now.return_value.date.return_value = test_date
        mock_datetime.strptime = datetime.strptime

        provider = FestivalProvider()
        upcoming = provider.get_upcoming_festivals(days_ahead=365)

        # All festivals should be in the future (days_until > 0)
        for festival in upcoming:
            assert festival["days_until"] > 0

    @patch("providers.festival_provider.datetime")
    def test_get_upcoming_festivals_excludes_today(self, mock_datetime):
        """Test that festivals today are excluded from upcoming."""
        # Find a festival date and test on that day
        provider = FestivalProvider()
        if provider.festivals:
            festival_date_str = provider.festivals[0]["date"]
            festival_date = datetime.strptime(festival_date_str, "%Y-%m-%d").date()

            mock_datetime.now.return_value.date.return_value = festival_date
            mock_datetime.strptime = datetime.strptime

            upcoming = provider.get_upcoming_festivals()
            # Today's festivals should not be in upcoming (days_until must be > 0)
            for festival in upcoming:
                assert festival["days_until"] > 0


class TestGetReminderFestivals:
    """Test cases for get_reminder_festivals method."""

    @patch("providers.festival_provider.datetime")
    def test_get_reminder_festivals_on_reminder_day(self, mock_datetime):
        """Test getting festivals that should be reminded today."""
        # Test with a festival that has reminder_days = [7, 3, 1]
        provider = FestivalProvider()
        if provider.festivals:
            festival = provider.festivals[0]
            festival_date = datetime.strptime(festival["date"], "%Y-%m-%d").date()
            reminder_days = festival.get("reminder_days", [7, 3, 1])

            # Set today to be 7 days before festival
            if reminder_days:
                test_date = festival_date - timedelta(days=reminder_days[0])
                mock_datetime.now.return_value.date.return_value = test_date
                mock_datetime.strptime = datetime.strptime

                reminder_festivals = provider.get_reminder_festivals()
                # Should include the festival since we're on a reminder day
                assert any(
                    f["type"] == festival["type"] and f["days_until"] == reminder_days[0]
                    for f in reminder_festivals
                )

    @patch("providers.festival_provider.datetime")
    def test_get_reminder_festivals_not_on_reminder_day(self, mock_datetime):
        """Test that festivals not on reminder days are excluded."""
        provider = FestivalProvider()
        if provider.festivals:
            festival = provider.festivals[0]
            festival_date = datetime.strptime(festival["date"], "%Y-%m-%d").date()
            reminder_days = festival.get("reminder_days", [7, 3, 1])

            # Set today to be 5 days before (not a reminder day)
            test_date = festival_date - timedelta(days=5)
            mock_datetime.now.return_value.date.return_value = test_date
            mock_datetime.strptime = datetime.strptime

            reminder_festivals = provider.get_reminder_festivals()
            # Should not include this festival
            if reminder_days and 5 not in reminder_days:
                assert not any(
                    f["type"] == festival["type"] and f["days_until"] == 5
                    for f in reminder_festivals
                )


class TestAddCustomFestival:
    """Test cases for add_custom_festival method."""

    def test_add_custom_festival_default_reminder_days(self):
        """Test adding custom festival with default reminder days."""
        provider = FestivalProvider()
        initial_count = len(provider.festivals)

        provider.add_custom_festival(
            name="Custom Festival",
            english_name="Custom",
            festival_type="custom_test",
            date="2025-06-15",
        )

        assert len(provider.festivals) == initial_count + 1
        custom_festival = next(
            (f for f in provider.festivals if f["type"] == "custom_test"), None
        )
        assert custom_festival is not None
        assert custom_festival["reminder_days"] == [7, 3, 1]

    def test_add_custom_festival_custom_reminder_days(self):
        """Test adding custom festival with custom reminder days."""
        provider = FestivalProvider()
        custom_reminder_days = [14, 7, 1]

        provider.add_custom_festival(
            name="Custom Festival 2",
            english_name="Custom2",
            festival_type="custom_test2",
            date="2025-07-20",
            reminder_days=custom_reminder_days,
        )

        custom_festival = next(
            (f for f in provider.festivals if f["type"] == "custom_test2"), None
        )
        assert custom_festival is not None
        assert custom_festival["reminder_days"] == custom_reminder_days

    def test_add_custom_festival_all_fields(self):
        """Test that all fields are correctly set when adding custom festival."""
        provider = FestivalProvider()

        provider.add_custom_festival(
            name="测试节日",
            english_name="Test Festival",
            festival_type="test_type",
            date="2025-08-10",
            reminder_days=[10, 5],
        )

        custom_festival = next(
            (f for f in provider.festivals if f["type"] == "test_type"), None
        )
        assert custom_festival is not None
        assert custom_festival["name"] == "测试节日"
        assert custom_festival["english_name"] == "Test Festival"
        assert custom_festival["type"] == "test_type"
        assert custom_festival["date"] == "2025-08-10"
        assert custom_festival["reminder_days"] == [10, 5]


class TestGetFestivalByType:
    """Test cases for get_festival_by_type method."""

    def test_get_festival_by_type_existing_type(self):
        """Test getting festival by existing type."""
        provider = FestivalProvider()
        if provider.festivals:
            existing_type = provider.festivals[0]["type"]
            festival = provider.get_festival_by_type(existing_type)

            assert festival is not None
            assert festival["type"] == existing_type

    def test_get_festival_by_type_nonexistent_type(self):
        """Test getting festival by nonexistent type returns None."""
        provider = FestivalProvider()
        festival = provider.get_festival_by_type("nonexistent_type_xyz")

        assert festival is None

    def test_get_festival_by_type_case_sensitive(self):
        """Test that type matching is case-sensitive."""
        provider = FestivalProvider()
        if provider.festivals:
            existing_type = provider.festivals[0]["type"]
            # Try with different case
            if existing_type:
                upper_type = existing_type.upper()
                if upper_type != existing_type:
                    festival = provider.get_festival_by_type(upper_type)
                    # Should not match if case is different
                    if upper_type not in [f["type"] for f in provider.festivals]:
                        assert festival is None


class TestEdgeCases:
    """Edge case and boundary tests."""

    @patch("providers.festival_provider.datetime")
    def test_empty_festivals_list(self, mock_datetime):
        """Test behavior with empty festivals list."""
        provider = FestivalProvider()
        original_festivals = provider.festivals.copy()
        provider.festivals = []

        today_festivals = provider.get_today_festivals()
        upcoming = provider.get_upcoming_festivals()
        reminder_festivals = provider.get_reminder_festivals()

        assert today_festivals == []
        assert upcoming == []
        assert reminder_festivals == []

        # Restore for other tests
        provider.festivals = original_festivals

    @patch("providers.festival_provider.datetime")
    def test_negative_days_ahead(self, mock_datetime):
        """Test get_upcoming_festivals with negative days_ahead."""
        test_date = date(2025, 1, 1)
        mock_datetime.now.return_value.date.return_value = test_date
        mock_datetime.strptime = datetime.strptime

        provider = FestivalProvider()
        upcoming = provider.get_upcoming_festivals(days_ahead=-1)

        # Should return empty list or handle gracefully
        assert isinstance(upcoming, list)

    @patch("providers.festival_provider.datetime")
    def test_zero_days_ahead(self, mock_datetime):
        """Test get_upcoming_festivals with zero days_ahead."""
        test_date = date(2025, 1, 1)
        mock_datetime.now.return_value.date.return_value = test_date
        mock_datetime.strptime = datetime.strptime

        provider = FestivalProvider()
        upcoming = provider.get_upcoming_festivals(days_ahead=0)

        # Should return empty list (no festivals within 0 days)
        assert upcoming == []

    def test_add_custom_festival_empty_reminder_days(self):
        """Test adding custom festival with empty reminder_days list."""
        provider = FestivalProvider()
        initial_count = len(provider.festivals)

        provider.add_custom_festival(
            name="Empty Reminders",
            english_name="Empty",
            festival_type="empty_reminders",
            date="2025-09-01",
            reminder_days=[],
        )

        assert len(provider.festivals) == initial_count + 1
        custom_festival = next(
            (f for f in provider.festivals if f["type"] == "empty_reminders"), None
        )
        assert custom_festival is not None
        assert custom_festival["reminder_days"] == []
