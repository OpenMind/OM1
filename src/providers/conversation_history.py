from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict


class ConversationHistory:
    """
    Extremely simple, JSON-serializable conversation history.

    - In-memory list[dict]
    - Each message: { "role": str, "content": str }
    - Optional disk persistence
    """

    def __init__(
        self,
        path: Path | None = None,
        max_length: int = 50,
        autosave: bool = True,
    ):
        self.path = path
        self.max_length = max_length
        self.autosave = autosave
        self._messages: List[Dict[str, str]] = []

        if self.path:
            self._load()

    # ---------- public API ----------

    def add(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})
        if len(self._messages) > self.max_length:
            self._messages = self._messages[-self.max_length :]

        if self.autosave:
            self._save()

    def clear(self) -> None:
        self._messages.clear()
        if self.autosave:
            self._save()

    def messages(self) -> List[Dict[str, str]]:
        return list(self._messages)

    # ---------- persistence ----------

    def _load(self) -> None:
        if not self.path.exists():
            return

        try:
            data = json.loads(self.path.read_text())
            if isinstance(data, list):
                self._messages = [
                    m for m in data
                    if isinstance(m, dict)
                    and "role" in m
                    and "content" in m
                ]
        except Exception:
            # Never break runtime on corrupted history
            self._messages = []

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(self._messages, indent=2)
            )
        except Exception:
            # Persistence must be best-effort only
            pass


