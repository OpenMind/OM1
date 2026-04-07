import json
import logging
import time
from datetime import datetime

from actions.base import ActionConfig, ActionConnector
from actions.schedule_cron_job.interface import ScheduleCronJobInput
from pydantic import Field

logger = logging.getLogger(__name__)


class ScheduleCronJobConfig(ActionConfig):
    """Configuration for the ScheduleCronJobJSONConnector."""

    schedule_file: str = Field(
        default="config/cron_job/cron.json",
        description="Path to the JSON file where scheduled cron jobs are persisted.",
    )


class ScheduleCronJobJSONConnector(ActionConnector[ScheduleCronJobConfig, ScheduleCronJobInput]):
    """Connector that persists scheduled cron jobs via the ExecuteCronJobProvider cache.

    Delegates storage to ``ExecuteCronJobProvider._add_entry()`` so that the
    in-memory cache and the JSON file are always updated together in a single
    atomic operation.

    Each entry written has the shape::

        {
            "timestamp":     <float>   # Unix timestamp parsed from schedule_time
            "schedule_time": <str>     # original date string from the LLM
            "function":      <str>     # function/command name
            "args":          <dict>    # parsed arguments
            "recurrence":    <str>     # "" | "once" | "daily" | "weekly" |
                                       # "hourly" | "every Xm/Xh/Xd"
            "registered_at": <float>   # wall-clock time of registration
        }

    Entries are sorted by ascending timestamp so the heartbeat process can
    stop scanning as soon as it reaches a future entry.
    """

    def __init__(self, config: ScheduleCronJobConfig) -> None:
        super().__init__(config)
        self.schedule_file: str = config.schedule_file

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    _DATE_FORMATS = (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M",
    )

    def _parse_schedule_time(self, schedule_time: str) -> float:
        for fmt in self._DATE_FORMATS:
            try:
                return datetime.strptime(schedule_time.strip(), fmt).timestamp()
            except ValueError:
                continue
        raise ValueError(
            f"Could not parse schedule_time '{schedule_time}'. "
            f"Expected format: 'YYYY-MM-DD HH:MM:SS'."
        )

    # ------------------------------------------------------------------
    # ActionConnector interface
    # ------------------------------------------------------------------

    async def connect(self, output_interface: ScheduleCronJobInput) -> None:
        """Persist a scheduled cron job entry and register it with ExecuteCronJobProvider."""
        try:
            timestamp = self._parse_schedule_time(output_interface.schedule_time)
        except ValueError as exc:
            logger.error("ScheduleCronJob: %s", exc)
            return

        try:
            args = json.loads(output_interface.args)
            if not isinstance(args, dict):
                args = {"value": args}
        except (json.JSONDecodeError, TypeError):
            logger.warning(
                "Could not parse args as JSON, storing as raw string: %s",
                output_interface.args,
            )
            args = {"raw": output_interface.args}

        recurrence = getattr(output_interface, "recurrence", "") or ""

        entry = {
            "timestamp": timestamp,
            "schedule_time": output_interface.schedule_time,
            "function": output_interface.function,
            "args": args,
            "recurrence": recurrence,
            "registered_at": time.time(),
        }

        # Update the provider's in-memory cache and flush to file atomically.
        from providers.execute_cron_job_provider import ExecuteCronJobProvider

        ExecuteCronJobProvider()._add_entry(entry)

        logger.info(
            "Scheduled cron job registered: function=%s at '%s' (timestamp=%.3f, recurrence=%r)",
            output_interface.function,
            output_interface.schedule_time,
            timestamp,
            recurrence,
        )
