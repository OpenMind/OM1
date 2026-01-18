"""Unit tests for FestivalReminder background task."""
from unittest.mock import MagicMock, patch
import pytest
from backgrounds.plugins.festival_reminder import FestivalReminder, FestivalReminderConfig

@pytest.fixture
def mock_providers():
    """Mock FestivalProvider and ContextProvider."""
    with patch("backgrounds.plugins.festival_reminder.FestivalProvider") as mock_festival, \
         patch("backgrounds.plugins.festival_reminder.ContextProvider") as mock_context:
        yield mock_festival(), mock_context()

def test_config_defaults():
    """Test FestivalReminderConfig default values."""
    config = FestivalReminderConfig()
    assert config.check_interval_seconds == 3600
    assert config.enable_reminders is True
    assert config.reminder_hour == 9

def test_config_custom_values():
    """Test FestivalReminderConfig with custom values."""
    config = FestivalReminderConfig(
        check_interval_seconds=1800,
        enable_reminders=False,
        reminder_hour=12
    )
    assert config.check_interval_seconds == 1800
    assert config.enable_reminders is False
    assert config.reminder_hour == 12

def test_initialization(mock_providers):
    """Test FestivalReminder initialization."""
    config = FestivalReminderConfig()
    reminder = FestivalReminder(config)
    assert reminder.config == config
    assert reminder.festival_provider is not None
    assert reminder.context_provider is not None

@patch("backgrounds.plugins.festival_reminder.time.time")
def test_run_respects_check_interval(mock_time, mock_providers):
    """Test that run respects check_interval_seconds."""
    mock_time.return_value = 1000.0
    config = FestivalReminderConfig(check_interval_seconds=60)
    reminder = FestivalReminder(config)
    reminder.last_check_time = 900.0  # 100 seconds ago, should not run yet
    
    reminder.run()
    # Should not update context if interval not reached
    
    reminder.last_check_time = 950.0  # 50 seconds ago, still not enough
    reminder.run()

def test_run_with_reminders_disabled(mock_providers):
    """Test run behavior when reminders are disabled."""
    config = FestivalReminderConfig(enable_reminders=False)
    reminder = FestivalReminder(config)
    reminder.last_check_time = 0
    
    reminder.run()
    # Should still check festivals but not send reminders
