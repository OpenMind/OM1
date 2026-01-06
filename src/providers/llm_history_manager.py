import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional


class LLMHistoryManager:
    """
    Conversation history manager.

    Guarantees that:
    - content is ALWAYS a string
    - returned messages are OpenAI-safe
    """

    def __init__(self, config, client=None):
        self.enabled = bool(getattr(config, "history_length", 0))
        self.history_length = int(getattr(config, "history_length", 0))
        self.client = client

        self._messages: List[Dict[str, str]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, role: str, content: Any, meta: Optional[Dict] = None) -> None:
        if not self.enabled:
            return

        msg = {
            "role": role,
            "content": self._to_string(content),
        }

        self._messages.append(msg)

        # Keep only last N messages
        if self.history_length > 0:
            self._messages = self._messages[-self.history_length :]

    def messages(self) -> List[Dict[str, str]]:
        """
        Return history as OpenAI-safe messages.
        """
        return [
            {
                "role": m.get("role", "assistant"),
                "content": self._to_string(m.get("content", "")),
            }
            for m in self._messages
        ]

    def clear(self) -> None:
        self._messages.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_string(self, value: Any) -> str:
        if value is None:
            return ""

        if isinstance(value, str):
            return value

        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)
