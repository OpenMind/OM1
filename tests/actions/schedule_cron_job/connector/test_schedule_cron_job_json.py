from unittest.mock import MagicMock, patch

import pytest

from actions.schedule_cron_job.connector.schedule_cron_job_json import (
    ScheduleCronJobConfig,
    ScheduleCronJobJSONConnector,
)
from actions.schedule_cron_job.interface import ScheduleCronJobInput


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset ExecuteCronJobProvider singleton between tests."""
    from providers.execute_cron_job_provider import ExecuteCronJobProvider

    ExecuteCronJobProvider.reset()  # type: ignore
    yield
    ExecuteCronJobProvider.reset()  # type: ignore


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
        mock_provider = MagicMock()
        with patch(
            "providers.execute_cron_job_provider.ExecuteCronJobProvider",
            return_value=mock_provider,
        ):
            inp = ScheduleCronJobInput(
                schedule_time="2026-04-07 10:00:00",
                function="speak",
                args='{"action": "hello"}',
                recurrence="daily",
            )
            await connector.connect(inp)

        mock_provider._add_entry.assert_called_once()
        entry = mock_provider._add_entry.call_args[0][0]
        assert entry["function"] == "speak"
        assert entry["args"] == {"action": "hello"}
        assert entry["recurrence"] == "daily"
        assert entry["schedule_time"] == "2026-04-07 10:00:00"
        assert "timestamp" in entry
        assert "registered_at" in entry

    @pytest.mark.asyncio
    async def test_connect_invalid_schedule_time_logs_and_returns(self, connector):
        mock_provider = MagicMock()
        with patch(
            "providers.execute_cron_job_provider.ExecuteCronJobProvider",
            return_value=mock_provider,
        ):
            inp = ScheduleCronJobInput(
                schedule_time="not-a-date",
                function="speak",
            )
            await connector.connect(inp)

        mock_provider._add_entry.assert_not_called()

    @pytest.mark.asyncio
    async def test_connect_non_json_args_stored_as_raw(self, connector):
        mock_provider = MagicMock()
        with patch(
            "providers.execute_cron_job_provider.ExecuteCronJobProvider",
            return_value=mock_provider,
        ):
            inp = ScheduleCronJobInput(
                schedule_time="2026-04-07 10:00:00",
                function="speak",
                args="not-json",
            )
            await connector.connect(inp)

        entry = mock_provider._add_entry.call_args[0][0]
        assert entry["args"] == {"raw": "not-json"}

    @pytest.mark.asyncio
    async def test_connect_non_dict_json_args_wrapped(self, connector):
        mock_provider = MagicMock()
        with patch(
            "providers.execute_cron_job_provider.ExecuteCronJobProvider",
            return_value=mock_provider,
        ):
            inp = ScheduleCronJobInput(
                schedule_time="2026-04-07 10:00:00",
                function="speak",
                args='"just a string"',
            )
            await connector.connect(inp)

        entry = mock_provider._add_entry.call_args[0][0]
        assert entry["args"] == {"value": "just a string"}

    @pytest.mark.asyncio
    async def test_connect_default_recurrence_is_empty(self, connector):
        mock_provider = MagicMock()
        with patch(
            "providers.execute_cron_job_provider.ExecuteCronJobProvider",
            return_value=mock_provider,
        ):
            inp = ScheduleCronJobInput(
                schedule_time="2026-04-07 10:00:00",
                function="speak",
            )
            await connector.connect(inp)

        entry = mock_provider._add_entry.call_args[0][0]
        assert entry["recurrence"] == ""
