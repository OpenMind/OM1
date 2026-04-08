import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from typing import ClassVar, List, Optional

from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider
from providers.sleep_ticker_provider import SleepTickerProvider

logger = logging.getLogger(__name__)


class ScheduledCronInputConfig(SensorConfig):
    """Configuration for the ScheduledCronInput plugin.

    Parameters
    ----------
    schedule_file : str
        Path to the JSON file where scheduled cron jobs are persisted.
        Defaults to ``"config/cron_job/cron.json"``.
    run_previous : bool
        If True, dispatch tasks whose schedule_time predates plugin startup.
        If False, silently skip stale entries. Defaults to True.
    """

    schedule_file: str = Field(default="config/cron_job/cron.json", description="Path to the JSON cron schedule file")
    run_previous: bool = Field(default=True, description="If True, dispatch tasks scheduled before startup")


class ScheduledCronInput(FuserInput[ScheduledCronInputConfig, Optional[str]]):
    """
    Input plugin that polls a JSON schedule file every second and dispatches due entries to the LLM.

    Manages both one-time and recurring tasks. Supported recurrence patterns:
    "" or "once" for one-time execution; "hourly", "daily", "weekly" for fixed
    intervals; "every Xm", "every Xh", "every Xd" for custom intervals.
    """

    # Reference to the active instance so add_entry() can reach it.
    _instance: ClassVar[Optional["ScheduledCronInput"]] = None

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

    def __init__(self, config: ScheduledCronInputConfig):
        """
        Initialize ScheduledCronInput.

        Parameters
        ----------
        config : ScheduledCronInputConfig
            Configuration for the scheduled cron input plugin.
        """
        super().__init__(config)
        self.messages: list[Message] = []
        self.descriptor_for_LLM = "User Command"
        self.io_provider = IOProvider()
        self._start_dt: Optional[datetime] = datetime.now().replace(microsecond=0)
        self._entries: List[dict] = []
        if not os.path.exists(self.config.schedule_file):
            self._write_all([])
        self._entries = self._read_file()
        logger.info(
            "ScheduledCronInput initialized: polling %s every 1s, run_previous=%s, loaded %d entries",
            self.config.schedule_file,
            self.config.run_previous,
            len(self._entries),
        )
        ScheduledCronInput._instance = self

    @classmethod
    def add_entry(cls, entry: dict) -> None:
        """
        Add a new schedule entry to the active instance.

        Parameters
        ----------
        entry : dict
            Schedule entry dict as produced by ScheduleCronJobJSONConnector.
        """
        if cls._instance is None:
            logging.warning("ScheduledCronInput: add_entry() called before instance created, entry dropped")
            return
        cls._instance._add_entry(entry)

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
            with open(self.config.schedule_file, "r") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _write_all(self, entries: list) -> None:
        """
        Write all entries to the JSON schedule file atomically via a temp file.

        Creates the parent directory if it does not exist, writes to a ``.tmp``
        sibling, then uses ``os.replace`` for an atomic rename so a partial write
        never corrupts the live file.

        Parameters
        ----------
        entries : list
            List of entry dicts to serialise. Typically ``self._entries`` after
            sorting by timestamp.
        """
        dir_name = os.path.dirname(self.config.schedule_file)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        temp_path = self.config.schedule_file + ".tmp"
        with open(temp_path, "w") as f:
            json.dump(entries, f, indent=2)
        os.replace(temp_path, self.config.schedule_file)

    def _add_entry(self, entry: dict) -> None:
        """
        Append a new entry to the in-memory cache and flush to disk.

        Appends the entry, re-sorts by ascending timestamp, then writes the
        full list via ``_write_all`` so the file and cache stay consistent.

        Parameters
        ----------
        entry : dict
            Schedule entry dict containing at least ``timestamp``,
            ``schedule_time``, ``function``, ``args``, ``recurrence``, and
            ``registered_at`` keys.
        """
        self._entries.append(entry)
        self._entries.sort(key=lambda e: e.get("timestamp", 0))
        self._write_all(self._entries)

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
        for fmt in self._DATE_FORMATS:
            try:
                return datetime.strptime(schedule_time.strip(), fmt)
            except ValueError:
                continue
        logger.warning("ScheduledCronInput: could not parse schedule_time '%s'", schedule_time)
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
        r = recurrence.strip().lower()
        if not r or r == "once":
            return None
        if r == "hourly":
            return timedelta(hours=1)
        if r == "daily":
            return timedelta(days=1)
        if r == "weekly":
            return timedelta(weeks=1)
        m = self._EVERY_PATTERN.match(r)
        if m:
            n = int(m.group(1))
            unit = m.group(2).rstrip("s") if len(m.group(2)) > 1 else m.group(2)
            unit = unit[0]  # first letter is always the canonical short form
            return timedelta(**{self._UNIT_MAP[unit]: n})
        logger.warning("ScheduledCronInput: unknown recurrence pattern '%s'", recurrence)
        return None

    def _is_due(self, entry: dict, now_dt: datetime) -> bool:
        """
        Determine whether a scheduled entry should be dispatched on this tick.

        Returns ``False`` if ``schedule_time`` is missing, unparseable, or still
        in the future. When ``run_previous`` is ``False``, also returns ``False``
        for entries whose ``schedule_time`` predates the plugin startup time,
        allowing stale entries to be silently skipped rather than replayed.

        Parameters
        ----------
        entry : dict
            Schedule entry dict; must contain a ``"schedule_time"`` key with a
            parseable datetime string.
        now_dt : datetime
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
        entry_dt = self._parse_schedule_time(schedule_time)
        if entry_dt is None or entry_dt > now_dt:
            return False
        if not self.config.run_previous and self._start_dt is not None:
            if entry_dt < self._start_dt:
                return False
        return True

    def _tick(self) -> None:
        """Check for due entries, dispatch them, and reschedule or remove each one."""
        now_dt = datetime.now().replace(microsecond=0)

        due = [e for e in self._entries if self._is_due(e, now_dt)]

        if not due:
            return

        keep = []
        for entry in self._entries:
            if not self._is_due(entry, now_dt):
                keep.append(entry)
                continue
            recurrence = entry.get("recurrence", "")
            delta = self._recurrence_delta(recurrence)
            if delta is None:
                logger.info(
                    "ScheduledCronInput: removing completed one-time task '%s'",
                    entry.get("function"),
                )
            else:
                current_dt = self._parse_schedule_time(entry["schedule_time"])
                if current_dt is not None:
                    next_dt = current_dt + delta
                    while next_dt <= now_dt:
                        next_dt += delta
                    entry["schedule_time"] = next_dt.strftime("%Y-%m-%d %H:%M:%S")
                    entry["timestamp"] = next_dt.timestamp()
                    entry["last_run_at"] = now_dt.strftime("%Y-%m-%d %H:%M:%S")
                    logger.info(
                        "ScheduledCronInput: recurring task '%s' rescheduled to %s",
                        entry.get("function"),
                        entry["schedule_time"],
                    )
                    keep.append(entry)
        self._entries = sorted(keep, key=lambda e: e.get("timestamp", 0))
        self._write_all(self._entries)

        for entry in due:
            function_name = entry.get("function", "")
            logger.info("ScheduledCronInput: dispatching '%s'", function_name)
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

        msg = self.messages[-1]
        result = f"\nINPUT: {self.descriptor_for_LLM}\n" f"// START\n{msg.message}\n// END\n"
        self.io_provider.add_input(self.descriptor_for_LLM, msg.message, time.time())
        self.messages = []
        return result
