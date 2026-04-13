import fcntl
import json
import logging
import os
import time
from datetime import datetime

from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.schedule_cron_job.interface import ScheduleCronJobInput


class ScheduleCronJobConfig(ActionConfig):
    """
    Configuration for the ScheduleCronJobJSONConnector.

    Parameters
    ----------
    schedule_file : str
        Path to the JSON file where scheduled cron jobs are persisted.
        Defaults to ``"config/cron_job/cron.json"``.
        Please make sure this is the same path as the schedule_file of corresponding ScheduleCronJobInput
    """

    schedule_file: str = Field(
        default="config/cron_job/cron.json",
        description="Path to the JSON file where scheduled cron jobs are persisted.",
    )


class ScheduleCronJobJSONConnector(ActionConnector[ScheduleCronJobConfig, ScheduleCronJobInput]):
    """
    Connector that persists scheduled cron jobs to a JSON file.

    Reads the current entries from the JSON file, appends the new entry, sorts
    by ascending timestamp, and writes back atomically. The input plugin detects
    the file change via mtime and reloads on its next tick.
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
        for date_format in self._DATE_FORMATS:
            try:
                return datetime.strptime(schedule_time.strip(), date_format).timestamp()
            except ValueError:
                continue
        raise ValueError(
            f"Could not parse schedule_time '{schedule_time}'. " f"Expected format: 'YYYY-MM-DD HH:MM:SS'."
        )

    def _read_entries(self) -> list:
        """
        Read the current entries from the JSON schedule file.

        Returns
        -------
        list
            List of entry dicts, or an empty list if the file does not exist
            or contains invalid JSON.
        """
        try:
            with open(self.schedule_file, "r") as file_handle:
                data = json.load(file_handle)
                return data if isinstance(data, list) else []
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _write_entries(self, entries: list) -> None:
        """
        Write entries to the JSON schedule file.

        Parameters
        ----------
        entries : list
            List of entry dicts to serialise.
        """
        dir_name = os.path.dirname(self.schedule_file)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(self.schedule_file, "w") as file_handle:
            json.dump(entries, file_handle, indent=2)

    def _locked_append(self, entry: dict) -> None:
        """
        Append an entry to the schedule file under an exclusive file lock.

        Acquires the lock on the JSON file itself, reads current entries,
        appends, sorts, and writes back. This prevents races with the input
        plugin's own read-modify-write cycle.

        Parameters
        ----------
        entry : dict
            The schedule entry to append.
        """
        dir_name = os.path.dirname(self.schedule_file)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(self.schedule_file, "a") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                existing_entries = self._read_entries()
                existing_entries.append(entry)
                existing_entries.sort(key=lambda e: e.get("timestamp", 0))
                self._write_entries(existing_entries)
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    async def connect(self, output_interface: ScheduleCronJobInput) -> None:
        """
        Persist a scheduled cron job entry to the JSON file.

        Reads existing entries, appends the new one, sorts by timestamp, and
        writes back atomically. The input plugin will detect the file change
        via mtime on its next tick.

        Parameters
        ----------
        output_interface : ScheduleCronJobInput
            Action output containing schedule_time (datetime string), function
            (name of the task to dispatch), and optional recurrence pattern.
        """
        try:
            timestamp = self._parse_schedule_time(output_interface.schedule_time)
        except ValueError as exc:
            logging.error("ScheduleCronJob: %s", exc)
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

        self._locked_append(entry)

        logging.info(
            "Scheduled cron job registered: function=%s at '%s' (timestamp=%.3f, recurrence=%r)",
            output_interface.function,
            output_interface.schedule_time,
            timestamp,
            recurrence,
        )
