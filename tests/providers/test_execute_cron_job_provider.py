import json
import os
import threading
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from providers.execute_cron_job_provider import ExecuteCronJobProvider


@pytest.fixture(autouse=True)
def reset_provider():
    ExecuteCronJobProvider.reset()
    yield
    ExecuteCronJobProvider.reset()


@pytest.fixture
def provider(tmp_path):
    schedule_file = str(tmp_path / "cron.json")
    return ExecuteCronJobProvider(schedule_file=schedule_file, poll_interval=0.05)


# ---------------------------------------------------------------------------
# _parse_schedule_time
# ---------------------------------------------------------------------------

class TestParseScheduleTime:
    def test_space_with_seconds(self, provider):
        dt = provider._parse_schedule_time("2026-04-07 10:30:00")
        assert dt == datetime(2026, 4, 7, 10, 30, 0)

    def test_T_with_seconds(self, provider):
        dt = provider._parse_schedule_time("2026-04-07T10:30:00")
        assert dt == datetime(2026, 4, 7, 10, 30, 0)

    def test_space_without_seconds(self, provider):
        dt = provider._parse_schedule_time("2026-04-07 10:30")
        assert dt == datetime(2026, 4, 7, 10, 30)

    def test_T_without_seconds(self, provider):
        dt = provider._parse_schedule_time("2026-04-07T10:30")
        assert dt == datetime(2026, 4, 7, 10, 30)

    def test_strips_whitespace(self, provider):
        dt = provider._parse_schedule_time("  2026-04-07 10:30:00  ")
        assert dt == datetime(2026, 4, 7, 10, 30, 0)

    def test_invalid_returns_none(self, provider):
        result = provider._parse_schedule_time("not-a-date")
        assert result is None


# ---------------------------------------------------------------------------
# _recurrence_delta
# ---------------------------------------------------------------------------

class TestRecurrenceDelta:
    def test_empty_string(self, provider):
        assert provider._recurrence_delta("") is None

    def test_once(self, provider):
        assert provider._recurrence_delta("once") is None

    def test_once_uppercase(self, provider):
        assert provider._recurrence_delta("ONCE") is None

    def test_hourly(self, provider):
        assert provider._recurrence_delta("hourly") == timedelta(hours=1)

    def test_daily(self, provider):
        assert provider._recurrence_delta("daily") == timedelta(days=1)

    def test_weekly(self, provider):
        assert provider._recurrence_delta("weekly") == timedelta(weeks=1)

    def test_every_minutes(self, provider):
        assert provider._recurrence_delta("every 30m") == timedelta(minutes=30)

    def test_every_hours(self, provider):
        assert provider._recurrence_delta("every 2h") == timedelta(hours=2)

    def test_every_days(self, provider):
        assert provider._recurrence_delta("every 3d") == timedelta(days=3)

    def test_every_seconds(self, provider):
        assert provider._recurrence_delta("every 10s") == timedelta(seconds=10)

    def test_every_full_word_minutes(self, provider):
        assert provider._recurrence_delta("every 5 minutes") == timedelta(minutes=5)

    def test_every_full_word_hours(self, provider):
        assert provider._recurrence_delta("every 1 hour") == timedelta(hours=1)

    def test_unknown_pattern_returns_none(self, provider):
        assert provider._recurrence_delta("fortnightly") is None


# ---------------------------------------------------------------------------
# _is_due
# ---------------------------------------------------------------------------

class TestIsDue:
    def _make_entry(self, schedule_time: str) -> dict:
        return {"schedule_time": schedule_time}

    def test_past_entry_is_due(self, provider):
        now = datetime(2026, 4, 7, 12, 0, 0)
        entry = self._make_entry("2026-04-07 11:59:00")
        assert provider._is_due(entry, now) is True

    def test_exact_time_is_due(self, provider):
        now = datetime(2026, 4, 7, 12, 0, 0)
        entry = self._make_entry("2026-04-07 12:00:00")
        assert provider._is_due(entry, now) is True

    def test_future_entry_not_due(self, provider):
        now = datetime(2026, 4, 7, 12, 0, 0)
        entry = self._make_entry("2026-04-07 13:00:00")
        assert provider._is_due(entry, now) is False

    def test_missing_schedule_time_not_due(self, provider):
        assert provider._is_due({}, datetime.now()) is False

    def test_run_previous_false_skips_old_entries(self, provider):
        provider.run_previous = False
        provider._start_dt = datetime(2026, 4, 7, 12, 0, 0)
        now = datetime(2026, 4, 7, 12, 0, 0)
        entry = self._make_entry("2026-04-07 11:00:00")
        assert provider._is_due(entry, now) is False

    def test_run_previous_true_includes_old_entries(self, provider):
        provider.run_previous = True
        provider._start_dt = datetime(2026, 4, 7, 12, 0, 0)
        now = datetime(2026, 4, 7, 12, 0, 0)
        entry = self._make_entry("2026-04-07 11:00:00")
        assert provider._is_due(entry, now) is True


# ---------------------------------------------------------------------------
# _read_file / _write_all
# ---------------------------------------------------------------------------

class TestFileIO:
    def test_write_and_read(self, provider, tmp_path):
        entries = [{"function": "foo", "timestamp": 1000.0}]
        provider._write_all(entries)
        result = provider._read_file()
        assert result == entries

    def test_read_missing_file_returns_empty(self, provider):
        result = provider._read_file()
        assert result == []

    def test_read_invalid_json_returns_empty(self, provider):
        os.makedirs(os.path.dirname(provider.schedule_file), exist_ok=True)
        with open(provider.schedule_file, "w") as f:
            f.write("not json")
        result = provider._read_file()
        assert result == []

    def test_read_non_list_json_returns_empty(self, provider):
        os.makedirs(os.path.dirname(provider.schedule_file), exist_ok=True)
        with open(provider.schedule_file, "w") as f:
            json.dump({"key": "value"}, f)
        result = provider._read_file()
        assert result == []


# ---------------------------------------------------------------------------
# _add_entry
# ---------------------------------------------------------------------------

class TestAddEntry:
    def test_add_entry_appears_in_cache(self, provider):
        entry = {"function": "speak", "timestamp": 9999.0, "schedule_time": "2026-04-07 10:00:00"}
        provider._add_entry(entry)
        assert entry in provider._entries

    def test_entries_sorted_by_timestamp(self, provider):
        provider._add_entry({"function": "b", "timestamp": 200.0, "schedule_time": "2026-04-07 10:00:00"})
        provider._add_entry({"function": "a", "timestamp": 100.0, "schedule_time": "2026-04-07 09:00:00"})
        assert provider._entries[0]["function"] == "a"
        assert provider._entries[1]["function"] == "b"

    def test_add_entry_persists_to_file(self, provider):
        entry = {"function": "speak", "timestamp": 9999.0, "schedule_time": "2026-04-07 10:00:00"}
        provider._add_entry(entry)
        assert os.path.exists(provider.schedule_file)
        with open(provider.schedule_file) as f:
            data = json.load(f)
        assert any(e["function"] == "speak" for e in data)


# ---------------------------------------------------------------------------
# start / stop
# ---------------------------------------------------------------------------

class TestStartStop:
    def test_start_creates_thread(self, provider):
        provider.start()
        assert provider._thread is not None
        assert provider._thread.is_alive()
        provider.stop()

    def test_start_idempotent(self, provider):
        provider.start()
        thread1 = provider._thread
        provider.start()  # second call should be a no-op
        assert provider._thread is thread1
        provider.stop()

    def test_stop_joins_thread(self, provider):
        provider.start()
        provider.stop()
        assert not provider._thread.is_alive()


# ---------------------------------------------------------------------------
# _tick — one-time entry dispatch and removal
# ---------------------------------------------------------------------------

class TestTick:
    def _past_entry(self, function="speak", recurrence=""):
        return {
            "function": function,
            "schedule_time": "2020-01-01 00:00:00",
            "timestamp": 1000.0,
            "recurrence": recurrence,
        }

    def test_one_time_entry_removed_after_tick(self, provider):
        entry = self._past_entry()
        provider._entries = [entry]

        with patch.object(provider, "_dispatch"):
            provider._tick()

        assert provider._entries == []

    def test_one_time_entry_dispatched(self, provider):
        entry = self._past_entry()
        provider._entries = [entry]
        dispatched = []

        def fake_dispatch(e):
            dispatched.append(e)

        # Replace threading.Thread with a fake that runs the target synchronously
        # when .start() is called, so we can assert without sleeps.
        class SyncThread:
            def __init__(self, target=None, args=(), **kwargs):
                self._target = target
                self._args = args

            def start(self):
                if self._target:
                    self._target(*self._args)

        with patch.object(provider, "_dispatch", side_effect=fake_dispatch):
            with patch("providers.execute_cron_job_provider.threading.Thread", SyncThread):
                provider._tick()

        assert len(dispatched) == 1
        assert dispatched[0]["function"] == "speak"

    def test_recurring_entry_rescheduled(self, provider):
        entry = self._past_entry(recurrence="daily")
        provider._entries = [entry]

        with patch("providers.execute_cron_job_provider.threading.Thread"):
            provider._tick()

        assert len(provider._entries) == 1
        rescheduled = provider._entries[0]
        new_dt = datetime.strptime(rescheduled["schedule_time"], "%Y-%m-%d %H:%M:%S")
        assert new_dt > datetime.now()

    def test_future_entry_not_dispatched(self, provider):
        entry = {
            "function": "speak",
            "schedule_time": "2099-01-01 00:00:00",
            "timestamp": 9999999999.0,
            "recurrence": "",
        }
        provider._entries = [entry]
        original_entries = list(provider._entries)

        with patch.object(provider, "_dispatch") as mock_dispatch:
            provider._tick()

        mock_dispatch.assert_not_called()
        assert provider._entries == original_entries
