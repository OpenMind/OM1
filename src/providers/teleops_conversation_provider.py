import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Any, Dict, List

import requests

from .singleton import singleton

# --- NEW: disk persistence helpers ---
# The repo seems to treat "src/" as the python path, so "history_store" is usually importable directly.
# This fallback keeps it working in different execution modes.
try:
    from history_store import load_history, append_message, default_history_path
except Exception:  # pragma: no cover
    from ..history_store import load_history, append_message, default_history_path


class MessageType(Enum):
    """
    Enumeration for message types in the conversation.
    """

    USER = "user"
    ROBOT = "robot"


@dataclass
class ConversationMessage:
    """
    Represents a conversation message with type, content, and timestamp.
    """

    message_type: MessageType
    content: str
    timestamp: float

    def to_dict(self) -> dict:
        """
        Convert the ConversationMessage to a dictionary.

        Returns
        -------
        dict
            Dictionary representation of the ConversationMessage.
        """
        return {
            "type": self.message_type.value,
            "content": self.content,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationMessage":
        """
        Create a ConversationMessage from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing message data.

        Returns
        -------
        ConversationMessage
            The created ConversationMessage instance.
        """
        return cls(
            message_type=MessageType(data.get("type", MessageType.USER.value)),
            content=data.get("content", ""),
            timestamp=data.get("timestamp", 0.0),
        )


@singleton
class TeleopsConversationProvider:
    """
    Singleton class to manage conversation messages with a Teleops backend.

    NEW:
    - Persist messages locally to a JSONL file on disk
    - Reload existing history on startup (if present)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openmind.org/api/core/teleops/conversation",
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.executor = ThreadPoolExecutor(max_workers=1)

        # --- NEW: persistent conversation history (disk) ---
        self.history_path = default_history_path()
        self.disk_history: List[Dict[str, Any]] = load_history(self.history_path)

    def store_user_message(self, content: str) -> None:
        """
        Store a user message in the conversation.

        Parameters
        ----------
        content : str
            The content of the user message.
        """
        message = ConversationMessage(
            message_type=MessageType.USER,
            content=content.strip(),
            timestamp=time.time(),
        )
        self._store_message(message)

    def store_robot_message(self, content: str) -> None:
        """
        Store a robot message in the conversation.

        Parameters
        ----------
        content : str
            The content of the robot message.
        """
        message = ConversationMessage(
            message_type=MessageType.ROBOT,
            content=content.strip(),
            timestamp=time.time(),
        )
        self._store_message(message)

    def _to_disk_record(self, message: ConversationMessage) -> Dict[str, Any]:
        """
        Convert internal message to a disk-friendly record.
        """
        role = "assistant" if message.message_type == MessageType.ROBOT else "user"
        return {
            "role": role,
            "content": message.content,
            "timestamp": message.timestamp,
        }

    def _persist_to_disk(self, message: ConversationMessage) -> None:
        """
        Persist message to JSONL on disk (append-only).
        """
        record = self._to_disk_record(message)
        append_message(self.history_path, record)
        self.disk_history.append(record)

    def _store_message_worker(self, message: ConversationMessage) -> None:
        """
        Worker method to store a conversation message via HTTP POST.

        Parameters
        ----------
        message : ConversationMessage
            The conversation message to store.
        """
        if not message.content or not message.content.strip():
            logging.debug("Empty content, skipping conversation storage")
            return

        # --- NEW: always persist locally so history survives restarts ---
        # This ensures "save to disk and reload if exists" works even if:
        # - API key is missing
        # - network is down
        # - backend returns non-200
        try:
            self._persist_to_disk(message)
        except Exception as e:
            logging.debug(f"Error persisting conversation history to disk: {str(e)}")

        # Existing behavior: attempt remote storage only if API key exists
        if self.api_key is None or self.api_key == "":
            logging.debug("API key is missing. Skipping remote conversation storage.")
            return

        try:
            request = requests.post(
                self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=message.to_dict(),
                timeout=2,
            )

            if request.status_code == 200:
                logging.debug(
                    f"Successfully stored {message.message_type.value} message to conversation"
                )
            else:
                logging.debug(
                    f"Failed to store {message.message_type.value} message: {request.status_code} - {request.text}"
                )
        except Exception as e:
            logging.debug(
                f"Error storing {message.message_type.value} conversation message: {str(e)}"
            )

    def _store_message(self, message: ConversationMessage) -> None:
        """
        Submit the message storage task to the executor.

        Parameters
        ----------
        message : ConversationMessage
            The conversation message to store.
        """
        self.executor.submit(self._store_message_worker, message)

    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Return locally persisted history (loaded on startup + appended as messages arrive).
        """
        if limit is None:
            return list(self.disk_history)
        return list(self.disk_history[-limit:])

    def is_enabled(self) -> bool:
        """
        Check if the Teleops conversation provider is enabled.

        Returns
        -------
        bool
            True if the API key is set, False otherwise.
        """
        return self.api_key is not None and self.api_key != ""
