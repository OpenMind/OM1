import asyncio
import json
import logging
import os
import re
import threading

import typing as T
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from actions.orchestrator import ActionOrchestrator
    from mcp_servers.client import MCPClientManager

from actions.base import AgentAction
from llm.output_model import Action
from providers.arg_normalizer import build_arg_normalizer

from .singleton import singleton

logger = logging.getLogger(__name__)

# Registry mapping llm_label -> callable.
# Populated automatically via register_actions(); can also be extended manually.
FUNCTION_REGISTRY: Dict[str, Callable[..., Any]] = {}


def _build_action_wrapper(agent_action: AgentAction) -> Callable[..., None]:
    """Return a plain (non-async) callable that drives agent_action.connector.connect().

    Uses ``build_arg_normalizer`` to robustly coerce LLM-produced args before
    instantiating the input dataclass.  Handles field aliasing (e.g. 'sentence'
    → 'action'), unknown-field pruning, single-field heuristic, and type
    coercion (Enum, int, float, bool, Optional[T]).
    """

    input_type = T.get_type_hints(agent_action.interface)["input"]
    input_type_hints = T.get_type_hints(input_type)
    normalize = build_arg_normalizer(input_type, input_type_hints, agent_action.llm_label)

    def wrapper(**args: Any) -> None:
        normalized = normalize(args)
        try:
            input_interface = input_type(**normalized)
        except TypeError as exc:
            logger.error(
                "execute_cron_job: failed to instantiate '%s' input with args %s: %s",
                agent_action.llm_label,
                normalized,
                exc,
            )
            return
        asyncio.run(agent_action.connector.connect(input_interface))

    return wrapper


@singleton
class ExecuteCronJobProvider:
    """Heartbeat provider that dispatches scheduled function calls.

    Loads ``schedule_file`` once at startup into an in-memory cache
    (``_entries``).  Every ``poll_interval`` seconds the cache is scanned for
    due entries; any match is dispatched in a daemon thread.  One-time entries
    are marked ``"completed": true``; recurring entries are rescheduled to
    their next occurrence.  All mutations update both the in-memory cache and
    the file atomically so they stay in sync.

    Recurrence patterns (stored in entry field ``"recurrence"``)
    ---------------------------------------------------------------
    * ``""`` or ``"once"`` — run once (default)
    * ``"hourly"``          — repeat every 60 minutes
    * ``"daily"``           — repeat every 24 hours
    * ``"weekly"``          — repeat every 7 days
    * ``"every Xm"``        — repeat every X minutes  (e.g. ``"every 30m"``)
    * ``"every Xh"``        — repeat every X hours    (e.g. ``"every 2h"``)
    * ``"every Xd"``        — repeat every X days     (e.g. ``"every 3d"``)

    Parameters
    ----------
    schedule_file:
        Path to the JSON file written by ``ScheduleCronJobJSONConnector``.
    poll_interval:
        How often (in seconds) to scan the cache. Matches ``interval`` in the
        ``cron_job`` config block.
    run_previous:
        If ``True``, tasks whose scheduled second is before the provider's
        start time are also dispatched (catch-up behaviour).
        If ``False``, only tasks scheduled for the current second or later
        (relative to when ``start()`` was called) are dispatched.

    Typical setup::

        provider = ExecuteCronJobProvider()
        provider.register_actions(runtime_config.agent_actions)
        provider.start()
    """


    def __init__(
        self,
        schedule_file: str = "config/cron_job/cron.json",
        poll_interval: float = 1.0,
        run_previous: bool = True,
        use_program_input: bool = False,
    ) -> None:
        self.schedule_file = schedule_file
        self.poll_interval = poll_interval
        self.run_previous = run_previous
        self.use_program_input = use_program_input

        self._start_dt: Optional[datetime] = None  # set when start() is called
        self._stop_event = threading.Event()
        self._file_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

        # In-memory cache — loaded from file on start(), kept in sync thereafter.
        self._entries: List[dict] = []

        self._mcp_client: Optional["MCPClientManager"] = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._action_orchestrator: Optional["ActionOrchestrator"] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_actions(self, agent_actions: List[AgentAction]) -> None:
        """Populate FUNCTION_REGISTRY from a list of AgentActions."""
        for action in agent_actions:
            FUNCTION_REGISTRY[action.llm_label] = _build_action_wrapper(action)
            logger.info(
                "ExecuteCronJobProvider: registered action '%s'", action.llm_label
            )

    def register_mcp(
        self,
        mcp_client: "MCPClientManager",
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Register an MCP client so scheduled MCP tool calls can be dispatched."""
        self._mcp_client = mcp_client
        self._event_loop = loop
        logger.info("ExecuteCronJobProvider: MCP client registered")

    def register_action_orchestrator(
        self,
        orchestrator: "ActionOrchestrator",
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Register the ActionOrchestrator for normal orchestration dispatch."""
        self._action_orchestrator = orchestrator
        self._event_loop = loop
        logger.info("ExecuteCronJobProvider: ActionOrchestrator registered")

    def start(self) -> None:
        """Start the background polling thread and load entries into cache."""
        if self._thread and self._thread.is_alive():
            return
        if not os.path.exists(self.schedule_file):
            self._write_all([])
        with self._file_lock:
            self._entries = self._read_file()
        self._start_dt = datetime.now().replace(microsecond=0)
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, name="ExecuteCronJobProvider", daemon=True
        )
        self._thread.start()
        logger.info(
            "ExecuteCronJobProvider started (polling %s every %.1fs, run_previous=%s, "
            "loaded %d entries)",
            self.schedule_file,
            self.poll_interval,
            self.run_previous,
            len(self._entries),
        )

    def stop(self) -> None:
        """Signal the polling thread to stop and wait for it to finish."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("ExecuteCronJobProvider stopped")

    def _add_entry(self, entry: dict) -> None:
        """Add a new entry to both the in-memory cache and the file.

        Called by ``ScheduleCallJSONConnector`` so the cache stays in sync
        without requiring a file re-read.  Safe to call before ``start()``
        — entries added early will be in the cache when the thread launches.
        """
        with self._file_lock:
            # If start() hasn't run yet, seed the cache from disk first.
            if not self._entries and os.path.exists(self.schedule_file):
                self._entries = self._read_file()
            self._entries.append(entry)
            self._entries.sort(key=lambda e: e.get("timestamp", 0))
            self._write_all(self._entries)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_file(self) -> list:
        """Read entries from disk (no lock — caller must hold _file_lock)."""
        try:
            with open(self.schedule_file, "r") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _write_all(self, entries: list) -> None:
        """Atomic write via a temp file (no lock — caller must hold _file_lock)."""
        os.makedirs(os.path.dirname(self.schedule_file), exist_ok=True)
        temp_path = self.schedule_file + ".tmp"
        with open(temp_path, "w") as f:
            json.dump(entries, f, indent=2)
        os.replace(temp_path, self.schedule_file)

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception:
                logger.exception("ExecuteCronJobProvider: error during tick")
            self._stop_event.wait(self.poll_interval)

    _DATE_FORMATS = (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M",
    )

    def _parse_schedule_time(self, schedule_time: str) -> Optional[datetime]:
        for fmt in self._DATE_FORMATS:
            try:
                return datetime.strptime(schedule_time.strip(), fmt)
            except ValueError:
                continue
        logger.warning(
            "ExecuteCronJobProvider: could not parse schedule_time '%s'", schedule_time
        )
        return None

    # ------------------------------------------------------------------
    # Recurrence helpers
    # ------------------------------------------------------------------

    # Accepts both short suffixes (s/m/h/d) and full English words
    # (second(s), minute(s), hour(s), day(s), week(s)).
    _EVERY_PATTERN = re.compile(
        r"^every\s+(\d+)\s*(s|m|h|d|seconds?|minutes?|hours?|days?|weeks?)$",
        re.IGNORECASE,
    )

    # Map normalised unit prefixes to their timedelta keyword.
    _UNIT_MAP = {
        "s": "seconds",
        "m": "minutes",
        "h": "hours",
        "d": "days",
        "w": "weeks",
    }

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
            # Normalise plural full words: "second" → "s", "minute" → "m", etc.
            unit = unit[0]  # first letter is always the canonical short form
            return timedelta(**{self._UNIT_MAP[unit]: n})
        logger.warning(
            "ExecuteCronJobProvider: unknown recurrence pattern '%s'", recurrence
        )
        return None

    def _is_due(self, entry: dict, now_dt: datetime) -> bool:
        """Return True if the entry should be dispatched right now."""
        schedule_time = entry.get("schedule_time", "")
        if not schedule_time:
            return False
        entry_dt = self._parse_schedule_time(schedule_time)
        if entry_dt is None or entry_dt > now_dt:
            return False
        if not self.run_previous and self._start_dt is not None:
            if entry_dt < self._start_dt:
                return False
        return True

    def _tick(self) -> None:
        now_dt = datetime.now().replace(microsecond=0)

        # Snapshot of currently due entries from the in-memory cache.
        with self._file_lock:
            due = [e for e in self._entries if self._is_due(e, now_dt)]

        if not due:
            return

        # Update cache and file: remove one-time entries, advance recurring ones.
        with self._file_lock:
            keep = []
            for entry in self._entries:
                if not self._is_due(entry, now_dt):
                    keep.append(entry)
                    continue
                recurrence = entry.get("recurrence", "")
                delta = self._recurrence_delta(recurrence)
                if delta is None:
                    # One-time task — drop it entirely.
                    logger.info(
                        "ExecuteCronJobProvider: removing completed one-time task '%s'",
                        entry.get("function"),
                    )
                else:
                    # Recurring task — advance to next occurrence and keep it.
                    current_dt = self._parse_schedule_time(entry["schedule_time"])
                    if current_dt is not None:
                        # Skip forward in one step to the first future occurrence,
                        # so a stale/past entry doesn't fire on every tick until
                        # it catches up to now.
                        next_dt = current_dt + delta
                        while next_dt <= now_dt:
                            next_dt += delta
                        entry["schedule_time"] = next_dt.strftime("%Y-%m-%d %H:%M:%S")
                        entry["timestamp"] = next_dt.timestamp()
                        entry["last_run_at"] = now_dt.strftime("%Y-%m-%d %H:%M:%S")
                        logger.info(
                            "ExecuteCronJobProvider: recurring task '%s' rescheduled to %s",
                            entry.get("function"),
                            entry["schedule_time"],
                        )
                        keep.append(entry)
            self._entries = sorted(keep, key=lambda e: e.get("timestamp", 0))
            self._write_all(self._entries)

        for entry in due:
            threading.Thread(
                target=self._dispatch,
                args=(entry,),
                daemon=True,
            ).start()

    def _dispatch(self, entry: dict) -> None:
        function_name: str = entry.get("function", "")
        args: dict = entry.get("args", {})

        # --- Known agent action: always dispatch directly, regardless of mode ---
        if function_name in FUNCTION_REGISTRY:
            logger.info(
                "ExecuteCronJobProvider: dispatching action '%s' via orchestrator, args=%s",
                function_name,
                args,
            )
            if self._action_orchestrator is not None and self._event_loop is not None:
                action = Action(type=function_name, value=json.dumps(args))
                future = asyncio.run_coroutine_threadsafe(
                    self._action_orchestrator.promise([action]),
                    self._event_loop,
                )
                try:
                    future.result(timeout=30)
                except Exception:
                    logger.exception(
                        "ExecuteCronJobProvider: error dispatching action '%s'",
                        function_name,
                    )
            else:
                logger.warning(
                    "ExecuteCronJobProvider: ActionOrchestrator not registered, "
                    "skipping action '%s'",
                    function_name,
                )
            return

        # --- MCP tool: inject the stored command into ScheduledCronInput ---
        if (
            self._mcp_client is not None
            and self._mcp_client.is_mcp_tool(function_name)
        ):
            from inputs.plugins.scheduled_cron_input import ScheduledCronInput

            command = args.get("command", function_name)
            logger.info(
                "ExecuteCronJobProvider: injecting MCP command '%s' into ScheduledCronInput: %s",
                function_name,
                command,
            )
            ScheduledCronInput.inject(command)
            return

        # --- use_program_input fallback: function is a natural-language command ---
        # Entries created in use_program_input mode store the stripped user request
        # in `function` (not a registered action name), so inject it into the LLM.
        if self.use_program_input:
            from inputs.plugins.scheduled_cron_input import ScheduledCronInput

            logger.info(
                "ExecuteCronJobProvider: injecting natural-language command '%s' into ScheduledCronInput",
                function_name,
            )
            ScheduledCronInput.inject(function_name)
            return

        logger.warning(
            "ExecuteCronJobProvider: no function registered for '%s'", function_name
        )
