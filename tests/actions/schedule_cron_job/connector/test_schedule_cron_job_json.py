import json
import os

import pytest

from actions.schedule_cron_job.connector.schedule_cron_job_json import (
    ScheduleCronJobConfig,
    ScheduleCronJobJSONConnector,
)
from actions.schedule_cron_job.interface import ScheduleCronJobInput


@pytest.fixture
def config(tmp_path):
    return ScheduleCronJobConfig(schedule_file=str(tmp_path / "cron.json"))


@pytest.fixture
def connector(config):
    return ScheduleCronJobJSONConnector(config)


class TestScheduleCronJobConfig:
    def test_default_schedule_file(self):
        cfg = ScheduleCronJobConfig()
        assert cfg.schedule_file == "config/cron_job/cron.json"

    def test_custom_schedule_file(self):
        cfg = ScheduleCronJobConfig(schedule_file="/tmp/test_cron.json")
        assert cfg.schedule_file == "/tmp/test_cron.json"


class TestScheduleCronJobJSONConnectorInit:
    def test_schedule_file_set(self):
        cfg = ScheduleCronJobConfig()
        connector = ScheduleCronJobJSONConnector(cfg)
        assert connector.schedule_file == "config/cron_job/cron.json"

    def test_custom_schedule_file(self):
        cfg = ScheduleCronJobConfig(schedule_file="/tmp/custom.json")
        connector = ScheduleCronJobJSONConnector(cfg)
        assert connector.schedule_file == "/tmp/custom.json"


class TestParseScheduleTime:
    def test_format_space_with_seconds(self, connector):
        ts = connector._parse_schedule_time("2026-04-07 10:30:00")
        assert ts > 0

    def test_format_T_with_seconds(self, connector):
        ts = connector._parse_schedule_time("2026-04-07T10:30:00")
        assert ts > 0

    def test_format_space_without_seconds(self, connector):
        ts = connector._parse_schedule_time("2026-04-07 10:30")
        assert ts > 0

    def test_format_T_without_seconds(self, connector):
        ts = connector._parse_schedule_time("2026-04-07T10:30")
        assert ts > 0

    def test_strips_whitespace(self, connector):
        ts = connector._parse_schedule_time("  2026-04-07 10:30:00  ")
        assert ts > 0

    def test_invalid_format_raises(self, connector):
        with pytest.raises(ValueError, match="Could not parse schedule_time"):
            connector._parse_schedule_time("not-a-date")


class TestReadWriteEntries:
    def test_read_empty_returns_empty_list(self, connector):
        assert connector._read_entries() == []

    def test_write_and_read_round_trip(self, connector):
        entries = [{"function": "foo", "timestamp": 1000.0}]
        connector._write_entries(entries)
        result = connector._read_entries()
        assert result == entries

    def test_read_invalid_json_returns_empty(self, connector):
        with open(connector.schedule_file, "w") as f:
            f.write("not json")
        assert connector._read_entries() == []

    def test_read_non_list_returns_empty(self, connector):
        with open(connector.schedule_file, "w") as f:
            json.dump({"key": "value"}, f)
        assert connector._read_entries() == []


class TestLockedAppend:
    def test_locked_append_creates_file(self, connector):
        entry = {"function": "speak", "timestamp": 1000.0}
        connector._locked_append(entry)
        assert os.path.exists(connector.schedule_file)
        with open(connector.schedule_file) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["function"] == "speak"

    def test_locked_append_sorts_by_timestamp(self, connector):
        connector._locked_append({"function": "b", "timestamp": 200.0})
        connector._locked_append({"function": "a", "timestamp": 100.0})
        with open(connector.schedule_file) as f:
            data = json.load(f)
        assert data[0]["function"] == "a"
        assert data[1]["function"] == "b"


class TestConnect:
    @pytest.mark.asyncio
    async def test_connect_persists_entry_to_file(self, connector):
        inp = ScheduleCronJobInput(
            schedule_time="2026-04-07 10:00:00",
            function="speak",
            recurrence="daily",
        )
        await connector.connect(inp)

        with open(connector.schedule_file) as f:
            data = json.load(f)
        assert len(data) == 1
        entry = data[0]
        assert entry["function"] == "speak"
        assert entry["args"] == {}
        assert entry["recurrence"] == "daily"
        assert entry["schedule_time"] == "2026-04-07 10:00:00"
        assert "timestamp" in entry
        assert "registered_at" in entry

    @pytest.mark.asyncio
    async def test_connect_invalid_schedule_time_logs_and_returns(self, connector):
        inp = ScheduleCronJobInput(
            schedule_time="not-a-date",
            function="speak",
        )
        await connector.connect(inp)

        assert not os.path.exists(connector.schedule_file)

    @pytest.mark.asyncio
    async def test_connect_default_recurrence_is_empty(self, connector):
        inp = ScheduleCronJobInput(
            schedule_time="2026-04-07 10:00:00",
            function="speak",
        )
        await connector.connect(inp)

        with open(connector.schedule_file) as f:
            data = json.load(f)
        assert data[0]["recurrence"] == ""
