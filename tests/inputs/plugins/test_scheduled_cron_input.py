import json
import os
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from inputs.plugins.schedule_cron_job_input import ScheduleCronJobInput, ScheduleCronJobInputConfig


@pytest.fixture
def plugin(tmp_path):
    schedule_file = str(tmp_path / "cron.json")
    config = ScheduleCronJobInputConfig(schedule_file=schedule_file)
    with patch("inputs.plugins.schedule_cron_job_input.IOProvider"):
        with patch("inputs.plugins.schedule_cron_job_input.SleepTickerProvider"):
            return ScheduleCronJobInput(config)


class TestParseScheduleTime:
    def test_space_with_seconds(self, plugin):
        dt = plugin._parse_schedule_time("2026-04-07 10:30:00")
        assert dt == datetime(2026, 4, 7, 10, 30, 0)

    def test_T_with_seconds(self, plugin):
        dt = plugin._parse_schedule_time("2026-04-07T10:30:00")
        assert dt == datetime(2026, 4, 7, 10, 30, 0)

    def test_space_without_seconds(self, plugin):
        dt = plugin._parse_schedule_time("2026-04-07 10:30")
        assert dt == datetime(2026, 4, 7, 10, 30)

    def test_T_without_seconds(self, plugin):
        dt = plugin._parse_schedule_time("2026-04-07T10:30")
        assert dt == datetime(2026, 4, 7, 10, 30)

    def test_strips_whitespace(self, plugin):
        dt = plugin._parse_schedule_time("  2026-04-07 10:30:00  ")
        assert dt == datetime(2026, 4, 7, 10, 30, 0)

    def test_invalid_returns_none(self, plugin):
        result = plugin._parse_schedule_time("not-a-date")
        assert result is None


class TestRecurrenceDelta:
    def test_empty_string(self, plugin):
        assert plugin._recurrence_delta("") is None

    def test_once(self, plugin):
        assert plugin._recurrence_delta("once") is None

    def test_once_uppercase(self, plugin):
        assert plugin._recurrence_delta("ONCE") is None

    def test_hourly(self, plugin):
        assert plugin._recurrence_delta("hourly") == timedelta(hours=1)

    def test_daily(self, plugin):
        assert plugin._recurrence_delta("daily") == timedelta(days=1)

    def test_weekly(self, plugin):
        assert plugin._recurrence_delta("weekly") == timedelta(weeks=1)

    def test_every_minutes(self, plugin):
        assert plugin._recurrence_delta("every 30m") == timedelta(minutes=30)

    def test_every_hours(self, plugin):
        assert plugin._recurrence_delta("every 2h") == timedelta(hours=2)

    def test_every_days(self, plugin):
        assert plugin._recurrence_delta("every 3d") == timedelta(days=3)

    def test_every_seconds(self, plugin):
        assert plugin._recurrence_delta("every 10s") == timedelta(seconds=10)

    def test_every_full_word_minutes(self, plugin):
        assert plugin._recurrence_delta("every 5 minutes") == timedelta(minutes=5)

    def test_every_full_word_hours(self, plugin):
        assert plugin._recurrence_delta("every 1 hour") == timedelta(hours=1)

    def test_unknown_pattern_returns_none(self, plugin):
        assert plugin._recurrence_delta("fortnightly") is None


class TestIsDue:
    def _make_entry(self, schedule_time: str) -> dict:
        return {"schedule_time": schedule_time}

    def test_past_entry_is_due(self, plugin):
        now = datetime(2026, 4, 7, 12, 0, 0)
        entry = self._make_entry("2026-04-07 11:59:00")
        assert plugin._is_due(entry, now) is True

    def test_exact_time_is_due(self, plugin):
        now = datetime(2026, 4, 7, 12, 0, 0)
        entry = self._make_entry("2026-04-07 12:00:00")
        assert plugin._is_due(entry, now) is True

    def test_future_entry_not_due(self, plugin):
        now = datetime(2026, 4, 7, 12, 0, 0)
        entry = self._make_entry("2026-04-07 13:00:00")
        assert plugin._is_due(entry, now) is False

    def test_missing_schedule_time_not_due(self, plugin):
        assert plugin._is_due({}, datetime.now()) is False

    def test_run_previous_false_skips_old_entries(self, plugin):
        plugin.config.run_previous = False
        plugin._startup_time = datetime(2026, 4, 7, 12, 0, 0)
        now = datetime(2026, 4, 7, 12, 0, 0)
        entry = self._make_entry("2026-04-07 11:00:00")
        assert plugin._is_due(entry, now) is False

    def test_run_previous_true_includes_old_entries(self, plugin):
        plugin.config.run_previous = True
        plugin._startup_time = datetime(2026, 4, 7, 12, 0, 0)
        now = datetime(2026, 4, 7, 12, 0, 0)
        entry = self._make_entry("2026-04-07 11:00:00")
        assert plugin._is_due(entry, now) is True


class TestFileIO:
    def test_write_and_read(self, plugin, tmp_path):
        entries = [{"function": "foo", "timestamp": 1000.0}]
        plugin._write_all(entries)
        result = plugin._read_file()
        assert result == entries

    def test_read_missing_file_returns_empty(self, tmp_path):
        config = ScheduleCronJobInputConfig(schedule_file=str(tmp_path / "nonexistent.json"))
        with patch("inputs.plugins.schedule_cron_job_input.IOProvider"):
            with patch("inputs.plugins.schedule_cron_job_input.SleepTickerProvider"):
                p = ScheduleCronJobInput(config)
        os.remove(config.schedule_file)
        result = p._read_file()
        assert result == []

    def test_read_invalid_json_returns_empty(self, plugin):
        with open(plugin.config.schedule_file, "w") as f:
            f.write("not json")
        result = plugin._read_file()
        assert result == []

    def test_read_non_list_json_returns_empty(self, plugin):
        with open(plugin.config.schedule_file, "w") as f:
            json.dump({"key": "value"}, f)
        result = plugin._read_file()
        assert result == []


class TestFileMtimeReload:
    def test_get_file_mtime_returns_float(self, plugin):
        mtime = plugin._get_file_mtime()
        assert isinstance(mtime, float)
        assert mtime > 0

    def test_get_file_mtime_missing_file_returns_zero(self, plugin):
        os.remove(plugin.config.schedule_file)
        assert plugin._get_file_mtime() == 0.0

    def test_reload_if_changed_detects_external_write(self, plugin):
        new_entries = [{"function": "external_task", "timestamp": 5000.0}]
        with open(plugin.config.schedule_file, "w") as f:
            json.dump(new_entries, f)
        plugin._reload_if_changed()
        assert len(plugin._entries) == 1
        assert plugin._entries[0]["function"] == "external_task"

    def test_reload_if_changed_no_op_when_unchanged(self, plugin):
        plugin._entries = [{"function": "cached"}]
        plugin._reload_if_changed()
        assert plugin._entries == [{"function": "cached"}]


class TestTick:
    def _past_entry(self, function="speak", recurrence=""):
        return {
            "function": function,
            "schedule_time": "2020-01-01 00:00:00",
            "timestamp": 1000.0,
            "recurrence": recurrence,
        }

    def test_one_time_entry_removed_after_tick(self, plugin):
        entry = self._past_entry()
        plugin._entries = [entry]
        plugin._write_all(plugin._entries)
        with patch("inputs.plugins.schedule_cron_job_input.SleepTickerProvider"):
            plugin._tick()
        assert plugin._entries == []

    def test_one_time_entry_dispatched(self, plugin):
        entry = self._past_entry()
        plugin._entries = [entry]
        plugin._write_all(plugin._entries)
        with patch("inputs.plugins.schedule_cron_job_input.SleepTickerProvider"):
            plugin._tick()
        assert len(plugin.messages) == 1
        assert plugin.messages[0].message == "speak"

    def test_recurring_entry_rescheduled(self, plugin):
        entry = self._past_entry(recurrence="daily")
        plugin._entries = [entry]
        plugin._write_all(plugin._entries)
        with patch("inputs.plugins.schedule_cron_job_input.SleepTickerProvider"):
            plugin._tick()
        assert len(plugin._entries) == 1
        rescheduled = plugin._entries[0]
        new_dt = datetime.strptime(rescheduled["schedule_time"], "%Y-%m-%d %H:%M:%S")
        assert new_dt > datetime.now()

    def test_future_entry_not_dispatched(self, plugin):
        entry = {
            "function": "speak",
            "schedule_time": "2099-01-01 00:00:00",
            "timestamp": 9999999999.0,
            "recurrence": "",
        }
        plugin._entries = [entry]
        plugin._write_all(plugin._entries)
        original_entries = list(plugin._entries)
        plugin._tick()
        assert plugin._entries == original_entries
        assert plugin.messages == []

    def test_multiple_due_entries_all_dispatched(self, plugin):
        entries = [self._past_entry(function="task_a"), self._past_entry(function="task_b")]
        plugin._entries = entries
        plugin._write_all(plugin._entries)
        with patch("inputs.plugins.schedule_cron_job_input.SleepTickerProvider"):
            plugin._tick()
        assert len(plugin.messages) == 2
        assert plugin.messages[0].message == "task_a"
        assert plugin.messages[1].message == "task_b"


class TestFormattedLatestBuffer:
    def test_empty_buffer_returns_none(self, plugin):
        assert plugin.formatted_latest_buffer() is None

    def test_single_message_flushed(self, plugin):
        from inputs.base import Message

        plugin.messages = [Message(timestamp=1.0, message="hello")]
        result = plugin.formatted_latest_buffer()
        assert "hello" in result
        assert plugin.messages == []

    def test_multiple_messages_drained_one_at_a_time(self, plugin):
        from inputs.base import Message

        plugin.messages = [
            Message(timestamp=1.0, message="task_a"),
            Message(timestamp=2.0, message="task_b"),
            Message(timestamp=3.0, message="task_c"),
        ]
        with patch("inputs.plugins.schedule_cron_job_input.SleepTickerProvider"):
            r1 = plugin.formatted_latest_buffer()
            assert "task_a" in r1
            assert len(plugin.messages) == 2

            r2 = plugin.formatted_latest_buffer()
            assert "task_b" in r2
            assert len(plugin.messages) == 1

            r3 = plugin.formatted_latest_buffer()
            assert "task_c" in r3
            assert plugin.messages == []

            assert plugin.formatted_latest_buffer() is None
