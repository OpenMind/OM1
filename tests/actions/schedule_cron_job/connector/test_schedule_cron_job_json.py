from unittest.mock import patch

import pytest

from actions.schedule_cron_job.connector.schedule_cron_job_json import (
    ScheduleCronJobConfig,
    ScheduleCronJobJSONConnector,
)
from actions.schedule_cron_job.interface import ScheduleCronJobInput


@pytest.fixture
def config():
    return ScheduleCronJobConfig()


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
    def test_schedule_file_set(self, config):
        connector = ScheduleCronJobJSONConnector(config)
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


class TestConnect:
    @pytest.mark.asyncio
    async def test_connect_calls_add_entry(self, connector):
        with patch("inputs.plugins.schedule_cron_job_input.ScheduledCronInput.add_entry") as mock_add:
            inp = ScheduleCronJobInput(
                schedule_time="2026-04-07 10:00:00",
                function="speak",
                recurrence="daily",
            )
            await connector.connect(inp)

        mock_add.assert_called_once()
        entry = mock_add.call_args[0][0]
        assert entry["function"] == "speak"
        assert entry["args"] == {}
        assert entry["recurrence"] == "daily"
        assert entry["schedule_time"] == "2026-04-07 10:00:00"
        assert "timestamp" in entry
        assert "registered_at" in entry

    @pytest.mark.asyncio
    async def test_connect_invalid_schedule_time_logs_and_returns(self, connector):
        with patch("inputs.plugins.schedule_cron_job_input.ScheduledCronInput.add_entry") as mock_add:
            inp = ScheduleCronJobInput(
                schedule_time="not-a-date",
                function="speak",
            )
            await connector.connect(inp)

        mock_add.assert_not_called()

    @pytest.mark.asyncio
    async def test_connect_default_recurrence_is_empty(self, connector):
        with patch("inputs.plugins.schedule_cron_job_input.ScheduledCronInput.add_entry") as mock_add:
            inp = ScheduleCronJobInput(
                schedule_time="2026-04-07 10:00:00",
                function="speak",
            )
            await connector.connect(inp)

        entry = mock_add.call_args[0][0]
        assert entry["recurrence"] == ""
