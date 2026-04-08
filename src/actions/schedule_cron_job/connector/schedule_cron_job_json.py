import logging
import time
from datetime import datetime

from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.schedule_cron_job.interface import ScheduleCronJobInput

logger = logging.getLogger(__name__)


class ScheduleCronJobConfig(ActionConfig):
    """Configuration for the ScheduleCronJobJSONConnector."""

    schedule_file: str = Field(
        default="config/cron_job/cron.json",
        description="Path to the JSON file where scheduled cron jobs are persisted.",
    )


class ScheduleCronJobJSONConnector(ActionConnector[ScheduleCronJobConfig, ScheduleCronJobInput]):
    """Connector that persists scheduled cron jobs via ScheduledCronInput.

    Delegates storage to ``ScheduledCronInput.add_entry()`` so that the
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
            f"Could not parse schedule_time '{schedule_time}'. " f"Expected format: 'YYYY-MM-DD HH:MM:SS'."
        )

    async def connect(self, output_interface: ScheduleCronJobInput) -> None:
        """Persist a scheduled cron job entry via ScheduledCronInput."""
        try:
            timestamp = self._parse_schedule_time(output_interface.schedule_time)
        except ValueError as exc:
            logger.error("ScheduleCronJob: %s", exc)
            return

        args: dict = {}
        recurrence = output_interface.recurrence or ""

        entry = {
            "timestamp": timestamp,
            "schedule_time": output_interface.schedule_time,
            "function": output_interface.function,
            "args": args,
            "recurrence": recurrence,
            "registered_at": time.time(),
        }

        from inputs.plugins.scheduled_cron_input import ScheduledCronInput

        ScheduledCronInput.add_entry(entry)

        logger.info(
            "Scheduled cron job registered: function=%s at '%s' (timestamp=%.3f, recurrence=%r)",
            output_interface.function,
            output_interface.schedule_time,
            timestamp,
            recurrence,
        )
