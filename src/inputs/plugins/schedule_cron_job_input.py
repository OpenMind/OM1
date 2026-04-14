import asyncio
import fcntl
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from typing import List, Optional

from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider
from providers.sleep_ticker_provider import SleepTickerProvider


class ScheduleCronJobInputConfig(SensorConfig):
    """Configuration for the ScheduleCronJobInput plugin.

    Parameters
    ----------
    schedule_file : str
        Path to the JSON file where scheduled cron jobs are persisted.
        Defaults to ``"config/cron_job/cron.json"``.
        Please make sure this is the same path as the schedule_file of corresponding ScheduleCronJobJSON action
    run_previous : bool
        If True, dispatch tasks whose schedule_time predates plugin startup.
        If False, silently skip stale entries. Defaults to True.
    """

    schedule_file: str = Field(default="config/cron_job/cron.json", description="Path to the JSON cron schedule file")
    run_previous: bool = Field(default=True, description="If True, dispatch tasks scheduled before startup")


class ScheduleCronJobInput(FuserInput[ScheduleCronJobInputConfig, Optional[str]]):
    """
    Input plugin that polls a JSON schedule file every second and dispatches due entries to the LLM.

    Manages both one-time and recurring tasks. Supported recurrence patterns:
    "" or "once" for one-time execution; "hourly", "daily", "weekly" for fixed
    intervals; "every Xm", "every Xh", "every Xd" for custom intervals.
    """

    _DATE_FORMATS = (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M",
    )

    _EVERY_PATTERN = re.compile(
        r"^every\s+(\d+)\s*(s|m|h|d|seconds?|minutes?|hours?|days?|weeks?)$",
        re.IGNORECASE,
    )

    _UNIT_MAP = {
        "s": "seconds",
        "m": "minutes",
        "h": "hours",
        "d": "days",
        "w": "weeks",
    }

    def __init__(self, config: ScheduleCronJobInputConfig):
        """
        Initialize ScheduleCronJobInput.

        Parameters
        ----------
        config : ScheduleCronJobInputConfig
            Configuration for the scheduled cron input plugin.
        """
        super().__init__(config)
        self.messages: list[Message] = []
        self.descriptor_for_LLM = "User Command"
        self.io_provider = IOProvider()
        self._startup_time: Optional[datetime] = datetime.now().replace(microsecond=0)
        self._entries: List[dict] = []
        if not os.path.exists(self.config.schedule_file):
            self._write_all([])
        self._entries = self._read_file()
        self._last_known_file_mtime: float = self._get_file_mtime()
        logging.info(
            "ScheduleCronJobInput initialized: polling %s every 1s, run_previous=%s, loaded %d entries",
            self.config.schedule_file,
            self.config.run_previous,
            len(self._entries),
        )

    def _get_file_mtime(self) -> float:
        """
        Return the modification time of the schedule file.

        Returns
        -------
        float
            File mtime as a Unix timestamp, or 0.0 if the file does not exist.
        """
        try:
            return os.path.getmtime(self.config.schedule_file)
        except OSError:
            return 0.0

    def _reload_if_changed(self) -> None:
        """
        Reload entries from disk if the file has been modified externally.
        """
        current_mtime = self._get_file_mtime()
        if current_mtime != self._last_known_file_mtime:
            self._entries = self._read_file()
            self._last_known_file_mtime = current_mtime
            logging.info(
                "ScheduleCronJobInput: file changed, reloaded %d entries",
                len(self._entries),
            )

    def _read_file(self) -> list:
        """
        Read cron entries from the JSON schedule file on disk.

        Returns
        -------
        list
            List of entry dicts loaded from the file, or an empty list if the
            file does not exist or contains invalid JSON.
        """
        try:
            with open(self.config.schedule_file, "r") as file_handle:
                data = json.load(file_handle)
                return data if isinstance(data, list) else []
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _write_all(self, entries: list) -> None:
        """
        Write all entries to the JSON schedule file.

        Creates the parent directory if it does not exist. Updates
        ``_last_known_file_mtime`` immediately so that ``_reload_if_changed``
        does not re-read our own write.

        Parameters
        ----------
        entries : list
            List of entry dicts to serialise. Typically ``self._entries`` after
            sorting by timestamp.
        """
        dir_name = os.path.dirname(self.config.schedule_file)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(self.config.schedule_file, "w") as file_handle:
            json.dump(entries, file_handle, indent=2)
        self._last_known_file_mtime = self._get_file_mtime()

    def _parse_schedule_time(self, schedule_time: str) -> Optional[datetime]:
        """
        Parse a schedule time string into a datetime object.

        Parameters
        ----------
        schedule_time : str
            Date/time string to parse. Supported formats:
            'YYYY-MM-DD HH:MM:SS', 'YYYY-MM-DDTHH:MM:SS',
            'YYYY-MM-DD HH:MM', 'YYYY-MM-DDTHH:MM'.

        Returns
        -------
        Optional[datetime]
            Parsed datetime on success, or None if no format matched.
        """
        for date_format in self._DATE_FORMATS:
            try:
                return datetime.strptime(schedule_time.strip(), date_format)
            except ValueError:
                continue
        logging.warning("ScheduleCronJobInput: could not parse schedule_time '%s'", schedule_time)
        return None

    def _recurrence_delta(self, recurrence: str) -> Optional[timedelta]:
        """
        Convert a recurrence pattern string into a ``timedelta``.

        Parameters
        ----------
        recurrence : str
            Pattern describing how often the task repeats. Recognised values:

            * ``""`` or ``"once"`` — one-time task; returns ``None``.
            * ``"hourly"`` — every 1 hour.
            * ``"daily"`` — every 24 hours.
            * ``"weekly"`` — every 7 days.
            * ``"every N <unit>"`` — arbitrary interval where ``<unit>`` is one of
              ``s``/``seconds``, ``m``/``minutes``, ``h``/``hours``,
              ``d``/``days``, ``w``/``weeks`` (singular or plural, case-insensitive).

        Returns
        -------
        Optional[timedelta]
            ``timedelta`` matching the pattern, or ``None`` for a one-time task.
            Unknown patterns log a warning and also return ``None``.
        """
        normalized_recurrence = recurrence.strip().lower()
        if not normalized_recurrence or normalized_recurrence == "once":
            return None
        if normalized_recurrence == "hourly":
            return timedelta(hours=1)
        if normalized_recurrence == "daily":
            return timedelta(days=1)
        if normalized_recurrence == "weekly":
            return timedelta(weeks=1)
        match_result = self._EVERY_PATTERN.match(normalized_recurrence)
        if match_result:
            interval_value = int(match_result.group(1))
            raw_time_unit = (
                match_result.group(2).rstrip("s") if len(match_result.group(2)) > 1 else match_result.group(2)
            )
            unit_key = raw_time_unit[0]  # first letter is always the canonical short form
            return timedelta(**{self._UNIT_MAP[unit_key]: interval_value})
        logging.warning("ScheduleCronJobInput: unknown recurrence pattern '%s'", recurrence)
        return None

    def _is_due(self, entry: dict, current_time: datetime) -> bool:
        """
        Determine whether a scheduled entry should be dispatched on this tick.

        Returns ``False`` if ``schedule_time`` is missing, unparsable, or still
        in the future. When ``run_previous`` is ``False``, also returns ``False``
        for entries whose ``schedule_time`` predates the plugin startup time,
        allowing stale entries to be silently skipped rather than replayed.

        Parameters
        ----------
        entry : dict
            Schedule entry dict; must contain a ``"schedule_time"`` key with a
            parseable datetime string.
        current_time : datetime
            Current datetime (microseconds stripped) used as the reference point.

        Returns
        -------
        bool
            ``True`` if the entry is due and should be dispatched; ``False``
            otherwise.
        """
        schedule_time = entry.get("schedule_time", "")
        if not schedule_time:
            return False
        entry_scheduled_time = self._parse_schedule_time(schedule_time)
        if entry_scheduled_time is None or entry_scheduled_time > current_time:
            return False
        if not self.config.run_previous and self._startup_time is not None:
            if entry_scheduled_time < self._startup_time:
                return False
        return True

    def _tick(self) -> None:
        """Check for due entries, dispatch them, and reschedule or remove each one."""
        schedule_dir = os.path.dirname(self.config.schedule_file)
        if schedule_dir:
            os.makedirs(schedule_dir, exist_ok=True)
        with open(self.config.schedule_file, "a") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                self._reload_if_changed()

                current_time = datetime.now().replace(microsecond=0)

                due_entries = [e for e in self._entries if self._is_due(e, current_time)]

                if not due_entries:
                    if self.messages:
                        SleepTickerProvider().skip_sleep = True
                    return

                remaining_entries = []
                for entry in self._entries:
                    if not self._is_due(entry, current_time):
                        remaining_entries.append(entry)
                        continue
                    recurrence = entry.get("recurrence", "")
                    recurrence_interval = self._recurrence_delta(recurrence)
                    if recurrence_interval is None:
                        logging.info(
                            "ScheduleCronJobInput: removing completed one-time task '%s'",
                            entry.get("function"),
                        )
                    else:
                        scheduled_time = self._parse_schedule_time(entry["schedule_time"])
                        if scheduled_time is not None:
                            next_scheduled_time = scheduled_time + recurrence_interval
                            while next_scheduled_time <= current_time:
                                next_scheduled_time += recurrence_interval
                            entry["schedule_time"] = next_scheduled_time.strftime("%Y-%m-%d %H:%M:%S")
                            entry["timestamp"] = next_scheduled_time.timestamp()
                            entry["last_run_at"] = current_time.strftime("%Y-%m-%d %H:%M:%S")
                            logging.info(
                                "ScheduleCronJobInput: recurring task '%s' rescheduled to %s",
                                entry.get("function"),
                                entry["schedule_time"],
                            )
                            remaining_entries.append(entry)
                self._entries = sorted(remaining_entries, key=lambda e: e.get("timestamp", 0))
                self._write_all(self._entries)
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

        for entry in due_entries:
            function_name = entry.get("function", "")
            logging.info("ScheduleCronJobInput: dispatching '%s'", function_name)
            self.messages.append(Message(timestamp=time.time(), message=function_name))
        SleepTickerProvider().skip_sleep = True

    async def _poll(self) -> Optional[str]:
        """
        Sleep 1 second, then run a tick to dispatch any due entries.

        Returns
        -------
        Optional[str]
            Always None; dispatched messages are pushed to self.messages directly.
        """
        await asyncio.sleep(1.0)
        self._tick()
        return None

    async def raw_to_text(self, raw_input: Optional[str]):
        """
        No-op: messages are pushed to self.messages directly by _tick() and consumed via formatted_latest_buffer().

        Parameters
        ----------
        raw_input : Optional[str]
            Unused.
        """
        pass

    def formatted_latest_buffer(self) -> Optional[str]:
        """
        Return the latest message formatted for the LLM, or None if the buffer is empty.

        Returns
        -------
        Optional[str]
            Formatted message string, or None if no messages are pending.
        """
        if not self.messages:
            return None

        message = self.messages.pop(0)
        result = f"\nINPUT: {self.descriptor_for_LLM}\n" f"// START\n{message.message}\n// END\n"
        self.io_provider.add_input(self.descriptor_for_LLM, message.message, time.time())
        return result
