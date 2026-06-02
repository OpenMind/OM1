import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from fuser.knowledge_base.base_retriever import Document
from providers.singleton import singleton


@singleton
class MemoryWriter:
    """Write interactions and summaries to daily markdown files.

    Parameters
    ----------
    memory_root : str or Path, optional
        Root directory for memory storage.
    """

    def __init__(
        self,
        memory_root: str | Path | None = None,
    ):
        if memory_root is None:
            project_root = Path(__file__).parent.parent.parent.parent
            memory_root = project_root / "memory"
        self.memory_root = Path(memory_root)
        self.daily_dir = self.memory_root / "daily"
        self.users_dir = self.memory_root / "users"
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """Create memory directories if they don't exist."""
        self.daily_dir.mkdir(parents=True, exist_ok=True)
        self.users_dir.mkdir(parents=True, exist_ok=True)

    def _get_daily_path(self) -> Path:
        """Get the path for today's daily log file."""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.daily_dir / f"{today}.md"

    def ensure_user_dir(self, user_id: str) -> Path:
        """Create the user directory and initialize profile/facts if needed.

        Parameters
        ----------
        user_id : str
            Normalized user identity.

        Returns
        -------
        Path
            Path to the user directory.
        """
        user_dir = self.users_dir / user_id
        user_dir.mkdir(parents=True, exist_ok=True)

        profile_path = user_dir / "profile.json"
        if not profile_path.exists():
            now = datetime.now().isoformat(timespec="seconds")
            profile = {
                "user_id": user_id,
                "display_name": user_id.title(),
                "first_seen": now,
                "last_seen": now,
                "interaction_count": 0,
            }
            profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
            logging.info(f"Memory: created user profile for {user_id}")

        facts_path = user_dir / "facts.json"
        if not facts_path.exists():
            facts = {"user_id": user_id, "facts": []}
            facts_path.write_text(json.dumps(facts, indent=2), encoding="utf-8")

        return user_dir

    def update_user_profile(self, user_id: str) -> None:
        """Update last_seen and increment interaction_count."""
        profile_path = self.users_dir / user_id / "profile.json"
        if not profile_path.exists():
            return

        try:
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            profile["last_seen"] = datetime.now().isoformat(timespec="seconds")
            profile["interaction_count"] = profile.get("interaction_count", 0) + 1
            profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
        except Exception as e:
            logging.warning(f"Memory: failed to update profile for {user_id}: {e}")

    def append_interaction(
        self,
        user_msg: str,
        actions: list,
        user_id: Optional[str] = None,
    ) -> None:
        """Append a user-robot interaction to today's daily log.

        Parameters
        ----------
        user_msg : str
            ASR input.
        actions : list
            LLM output actions.
        user_id : str, optional
            Identity of the user (from face recognition).
        """
        if not user_msg.strip():
            return

        if user_id:
            self.ensure_user_dir(user_id)
            self.update_user_profile(user_id)

        daily_path = self._get_daily_path()
        timestamp = datetime.now().strftime("%H:%M:%S")

        robot_msg = " | ".join(f"{a.type}: {a.value}" for a in actions if a.value)

        entry = f"\n## {timestamp}\n"
        if user_id:
            entry += f"[User: {user_id}]\n"
        entry += f"- **User**: {user_msg.strip()}\n"
        entry += f"- **Robot**: {robot_msg}\n"

        try:
            with open(daily_path, "a", encoding="utf-8") as f:
                f.write(entry)
            logging.debug(f"Memory: appended interaction to {daily_path.name}")
        except Exception as e:
            logging.error(f"Memory: failed to write interaction: {e}")

    async def append_to_index(
        self,
        user_msg: str,
        actions: list,
        user_id: Optional[str] = None,
    ) -> None:
        """Embed and insert a new interaction into the in-memory index.

        Skipped if the index has not been initialized yet.

        Parameters
        ----------
        user_msg : str
            ASR input.
        actions : list
            LLM output actions.
        user_id : str, optional
            Identity of the user (from face recognition).
        """
        if not user_msg.strip():
            return

        from fuser.memory_base.reader import MemoryReader

        reader = MemoryReader()
        if not reader._index_initialized:
            return

        try:
            date_str = datetime.now().strftime("%Y-%m-%d")
            timestamp = datetime.now().strftime("%H:%M:%S")
            robot_msg = " | ".join(f"{a.type}: {a.value}" for a in actions if a.value)

            user_tag = f"[User: {user_id}]\n" if user_id else ""
            text = (
                f"[Date: {date_str}]\n## {timestamp}\n{user_tag}"
                f"- **User**: {user_msg.strip()}\n"
                f"- **Robot**: {robot_msg}"
            )
            metadata = {"source": f"{date_str}.md"}
            if user_id:
                metadata["user_id"] = user_id
            chunk = Document(text=text, metadata=metadata)
            await reader.index.add_chunk(chunk)
        except Exception as e:
            logging.warning(f"Memory: write-through index failed: {e}")
