"""Memory writer for persisting interactions to daily markdown logs.

Appends user-robot interactions to date-stamped markdown files
in the memory/daily/ directory. Optionally updates the in-memory
embedding index incrementally after each write.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from fuser.memory.indexer import MemoryIndex


class MemoryWriter:
    """Write interactions and summaries to daily markdown files.

    Parameters
    ----------
    memory_root : str or Path, optional
        Root directory for memory storage (contains MEMORY.md and daily/).
        Defaults to ``<project_root>/memory``.
    index : MemoryIndex, optional
        Shared in-memory index for incremental updates.
    """

    def __init__(
        self,
        memory_root: Optional[str | Path] = None,
        index: Optional["MemoryIndex"] = None,
    ):
        if memory_root is None:
            project_root = Path(__file__).parent.parent.parent.parent
            memory_root = project_root / "memory"
        self.memory_root = Path(memory_root)
        self.daily_dir = self.memory_root / "daily"
        self.memory_file = self.memory_root / "MEMORY.md"
        self.index = index
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

    async def append_interaction(
        self,
        user_msg: str,
        robot_msg: str,
        mode: Optional[str] = None,
    ) -> None:
        """Append a user-robot interaction to today's daily log.

        Also incrementally adds the new chunk to the embedding index
        if one is configured.

        Parameters
        ----------
        user_msg : str
            The user's input (typically from Voice).
        robot_msg : str
            The robot's response/actions.
        mode : str, optional
            Current operating mode name.
        """
        if not user_msg.strip():
            return

        daily_path = self._get_daily_path()
        timestamp = datetime.now().strftime("%H:%M:%S")
        mode_label = f" ({mode})" if mode else ""

        entry = f"\n## {timestamp}{mode_label}\n"
        entry += f"- **User**: {user_msg.strip()}\n"
        entry += f"- **Robot**: {robot_msg.strip()}\n"

        try:
            await asyncio.to_thread(self._write_file, daily_path, entry)
            logging.debug(f"Memory: appended interaction to {daily_path.name}")
        except Exception as e:
            logging.error(f"Memory: failed to write interaction: {e}")
            return

        # Incremental index update
        if self.index:
            try:
                await self.index.add_chunk(
                    text=entry.strip(),
                    metadata={
                        "source": daily_path.name,
                        "timestamp": timestamp,
                    },
                )
            except Exception as e:
                logging.error(f"Memory: failed to index chunk: {e}")

    async def append_summary(
        self,
        summary: str,
        mode: Optional[str] = None,
    ) -> None:
        """Append a history summary to today's daily log.

        Parameters
        ----------
        summary : str
            The LLM-generated summary of recent history.
        mode : str, optional
            The mode that generated this summary.
        """
        if not summary.strip():
            return

        daily_path = self._get_daily_path()
        timestamp = datetime.now().strftime("%H:%M:%S")
        mode_label = f" ({mode})" if mode else ""

        entry = f"\n## Summary {timestamp}{mode_label}\n"
        entry += f"{summary.strip()}\n"

        try:
            await asyncio.to_thread(self._write_file, daily_path, entry)
            logging.debug(f"Memory: appended summary to {daily_path.name}")
        except Exception as e:
            logging.error(f"Memory: failed to write summary: {e}")

    async def write_fact(self, fact: str) -> None:
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
            await asyncio.to_thread(self._write_file, self.memory_file, entry)
            logging.info(f"Memory: wrote fact to MEMORY.md: {fact[:50]}...")
        except Exception as e:
            logging.error(f"Memory: failed to write fact: {e}")

    @staticmethod
    def _write_file(path: Path, content: str) -> None:
        """Synchronous file append (called via asyncio.to_thread)."""
        with open(path, "a", encoding="utf-8") as f:
            f.write(content)
