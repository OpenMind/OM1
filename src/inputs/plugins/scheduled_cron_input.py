import asyncio
import json
import logging
import os
import re
import threading
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
    """Configuration for the ScheduledCronInput plugin."""

    input_name: str = Field(default="User Async Task", description="Label shown to the LLM for this input")
    schedule_file: str = Field(default="config/cron_job/cron.json", description="Path to the JSON cron schedule file")
    run_previous: bool = Field(default=True, description="If True, dispatch tasks scheduled before startup")


class ScheduledCronInput(FuserInput[ScheduledCronInputConfig, Optional[str]]):
    """
    Input plugin that polls a JSON schedule file every second, dispatches due
    entries to the LLM, and manages one-time vs recurring tasks.

    Recurrence patterns (stored in entry field ``"recurrence"``)
    ---------------------------------------------------------------
    * ``""`` or ``"once"`` — run once (default)
    * ``"hourly"``          — repeat every 60 minutes
    * ``"daily"``           — repeat every 24 hours
    * ``"weekly"``          — repeat every 7 days
    * ``"every Xm"``        — repeat every X minutes  (e.g. ``"every 30m"``)
    * ``"every Xh"``        — repeat every X hours    (e.g. ``"every 2h"``)
    * ``"every Xd"``        — repeat every X days     (e.g. ``"every 3d"``)
    """

    # Reference to the active instance so add_entry() can reach it.
    _instance: ClassVar[Optional["ScheduledCronInput"]] = None

    # Set to True when formatted_latest_buffer() returns a message this tick,
    # so cortex can filter schedule_cron_job from the LLM response and prevent
    # cron-triggered ticks from re-registering the same job.
    cron_triggered: bool = False

    _DATE_FORMATS = (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M",
    )

    # Accepts both short suffixes (s/m/h/d) and full English words.
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
        super().__init__(config)
        self.messages: list[Message] = []
        self.descriptor_for_LLM = self.config.input_name
        self.io_provider = IOProvider()
        self._file_lock = threading.Lock()
        self._start_dt: Optional[datetime] = datetime.now().replace(microsecond=0)
        self._entries: List[dict] = []
        with self._file_lock:
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
        """Add a new schedule entry. Called by ScheduleCronJobJSONConnector."""
        if cls._instance is None:
            logging.warning("ScheduledCronInput: add_entry() called before instance created, entry dropped")
            return
        cls._instance._add_entry(entry)

    def _read_file(self) -> list:
        """Read entries from disk (caller must hold _file_lock)."""
        try:
            with open(self.config.schedule_file, "r") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _write_all(self, entries: list) -> None:
        """Atomic write via a temp file (caller must hold _file_lock)."""
        dir_name = os.path.dirname(self.config.schedule_file)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        temp_path = self.config.schedule_file + ".tmp"
        with open(temp_path, "w") as f:
            json.dump(entries, f, indent=2)
        os.replace(temp_path, self.config.schedule_file)

    def _add_entry(self, entry: dict) -> None:
        """Add entry to in-memory cache and flush to file atomically."""
        with self._file_lock:
            self._entries.append(entry)
            self._entries.sort(key=lambda e: e.get("timestamp", 0))
            self._write_all(self._entries)

    def _parse_schedule_time(self, schedule_time: str) -> Optional[datetime]:
        for fmt in self._DATE_FORMATS:
            try:
                return datetime.strptime(schedule_time.strip(), fmt)
            except ValueError:
                continue
        logger.warning("ScheduledCronInput: could not parse schedule_time '%s'", schedule_time)
        return None

    def _recurrence_delta(self, recurrence: str) -> Optional[timedelta]:
        """Return the timedelta for a recurrence pattern, or None for one-time."""
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
        """Return True if the entry should be dispatched right now."""
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
        now_dt = datetime.now().replace(microsecond=0)

        with self._file_lock:
            due = [e for e in self._entries if self._is_due(e, now_dt)]

        if not due:
            return

        with self._file_lock:
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
        await asyncio.sleep(1.0)
        self._tick()
        return None

    async def raw_to_text(self, raw_input: Optional[str]):
        """No-op: messages are pushed to messages directly and consumed via formatted_latest_buffer()."""
        pass

    def formatted_latest_buffer(self) -> Optional[str]:
        """Return the latest message formatted for the LLM, or None if empty."""
        if not self.messages:
            ScheduledCronInput.cron_triggered = False
            return None

        msg = self.messages[-1]
        result = f"\nINPUT: {self.descriptor_for_LLM}\n" f"// START\n{msg.message}\n// END\n"
        self.io_provider.add_input(self.descriptor_for_LLM, msg.message, time.time())
        self.messages = []
        ScheduledCronInput.cron_triggered = True
        return result
