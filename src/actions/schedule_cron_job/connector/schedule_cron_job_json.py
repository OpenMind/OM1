import logging
import time
from datetime import datetime

from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.schedule_cron_job.interface import ScheduleCronJobInput
from inputs.plugins.schedule_cron_job_input import ScheduledCronInput

logger = logging.getLogger(__name__)


class ScheduleCronJobConfig(ActionConfig):
    """Configuration for the ScheduleCronJobJSONConnector."""

    schedule_file: str = Field(
        default="config/cron_job/cron.json",
        description="Path to the JSON file where scheduled cron jobs are persisted.",
    )


class ScheduleCronJobJSONConnector(ActionConnector[ScheduleCronJobConfig, ScheduleCronJobInput]):
    """
    Connector that persists scheduled cron jobs to a JSON file.

    Delegates storage to ScheduledCronInput.add_entry() so that the in-memory
    cache and the JSON file are updated together. Entries are sorted by ascending
    timestamp to preserve deterministic processing order for consumers of the
    persisted schedule.
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
        """
        Parse a schedule time string into a Unix timestamp.

        Parameters
        ----------
        schedule_time : str
            Date/time string to parse. Supported formats:
            'YYYY-MM-DD HH:MM:SS', 'YYYY-MM-DDTHH:MM:SS',
            'YYYY-MM-DD HH:MM', 'YYYY-MM-DDTHH:MM'.

        Returns
        -------
        float
            Unix timestamp corresponding to schedule_time.
        """
        for fmt in self._DATE_FORMATS:
            try:
                return datetime.strptime(schedule_time.strip(), fmt).timestamp()
            except ValueError:
                continue
        raise ValueError(
            f"Could not parse schedule_time '{schedule_time}'. " f"Expected format: 'YYYY-MM-DD HH:MM:SS'."
        )

    async def connect(self, output_interface: ScheduleCronJobInput) -> None:
        """
        Persist a scheduled cron job entry to the in-memory cache and JSON file.

        Parses the schedule time from output_interface, builds an entry dict, and
        delegates to ScheduledCronInput.add_entry() so that both the in-memory cache
        and the JSON file are updated atomically. If the schedule_time cannot be
        parsed, logs an error and returns without writing anything.

        Parameters
        ----------
        output_interface : ScheduleCronJobInput
            Action output containing schedule_time (datetime string), function
            (name of the task to dispatch), and optional recurrence pattern.
        """
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

        ScheduledCronInput.add_entry(entry)

        logger.info(
            "Scheduled cron job registered: function=%s at '%s' (timestamp=%.3f, recurrence=%r)",
            output_interface.function,
            output_interface.schedule_time,
            timestamp,
            recurrence,
        )
