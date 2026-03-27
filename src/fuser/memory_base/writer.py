import logging
from datetime import datetime
from pathlib import Path
from typing import Optional


class MemoryWriter:
    """Write interactions and summaries to daily markdown files.

    Parameters
    ----------
    memory_root : str or Path, optional
        Root directory for memory storage (contains MEMORY.md and daily/).
        Defaults to ``<project_root>/memory``.
    """

    def __init__(
        self,
        memory_root: Optional[str | Path] = None,
    ):
        if memory_root is None:
            project_root = Path(__file__).parent.parent.parent.parent
            memory_root = project_root / "memory"
        self.memory_root = Path(memory_root)
        self.daily_dir = self.memory_root / "daily"
        self.memory_file = self.memory_root / "MEMORY.md"
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """Create memory directories if they don't exist."""
        self.daily_dir.mkdir(parents=True, exist_ok=True)
        if not self.memory_file.exists():
            self.memory_file.write_text(
                "# Long-Term Memory\n\n"
                "<!-- Persistent facts, preferences, and important context -->\n"
            )

    def _get_daily_path(self) -> Path:
        """Get the path for today's daily log file."""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.daily_dir / f"{today}.md"

    def append_interaction(
        self,
        user_msg: str,
        actions: list,
    ) -> None:
        """Append a user-robot interaction to today's daily log.

        Parameters
        ----------
        user_msg : str
            The user's input (typically from Voice).
        actions : list
            The robot's output actions.
        """
        if not user_msg.strip():
            return

        daily_path = self._get_daily_path()
        timestamp = datetime.now().strftime("%H:%M:%S")

        robot_msg = " | ".join(f"{a.type}: {a.value}" for a in actions if a.value)

        entry = f"\n## {timestamp}\n"
        entry += f"- **User**: {user_msg.strip()}\n"
        entry += f"- **Robot**: {robot_msg}\n"

        try:
            with open(daily_path, "a", encoding="utf-8") as f:
                f.write(entry)
            logging.debug(f"Memory: appended interaction to {daily_path.name}")
        except Exception as e:
            logging.error(f"Memory: failed to write interaction: {e}")

    def append_summary(self, summary: str) -> None:
        """Append a history summary to today's daily log.

        Parameters
        ----------
        summary : str
            The LLM-generated summary of recent history.
        """
        if not summary.strip():
            return

        daily_path = self._get_daily_path()
        timestamp = datetime.now().strftime("%H:%M:%S")

        entry = f"\n## Summary {timestamp}\n"
        entry += f"{summary.strip()}\n"

        try:
            with open(daily_path, "a", encoding="utf-8") as f:
                f.write(entry)
            logging.debug(f"Memory: appended summary to {daily_path.name}")
        except Exception as e:
            logging.error(f"Memory: failed to write summary: {e}")

    def write_fact(self, fact: str) -> None:
        """Append a persistent fact to MEMORY.md.

        Parameters
        ----------
        fact : str
            The fact or preference to persist.
        """
        if not fact.strip():
            return

        entry = f"\n- {fact.strip()}\n"

        try:
            with open(self.memory_file, "a", encoding="utf-8") as f:
                f.write(entry)
            logging.info(f"Memory: wrote fact to MEMORY.md: {fact[:50]}...")
        except Exception as e:
            logging.error(f"Memory: failed to write fact: {e}")
