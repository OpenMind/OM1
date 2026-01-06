import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .singleton import singleton


@dataclass
class ConversationEntry:
    """
    A dataclass representing a single conversation entry with prompt and output.

    Parameters
    ----------
    tick : int
        The tick number when this conversation occurred.
    timestamp : float
        Unix timestamp when the entry was created.
    mode : str, optional
        The operational mode during this conversation (default is None).
    prompt : str, optional
        The prompt sent to the LLM (default is None).
    output : str, optional
        The output received from the LLM (default is None).
    prompt_timestamp : float, optional
        Unix timestamp when the prompt was recorded (default is None).
    output_timestamp : float, optional
        Unix timestamp when the output was recorded (default is None).
    """

    tick: int
    timestamp: float
    mode: Optional[str] = None
    prompt: Optional[str] = None
    output: Optional[str] = None
    prompt_timestamp: Optional[float] = None
    output_timestamp: Optional[float] = None

    def to_dict(self) -> Dict:
        """
        Convert the conversation entry to a dictionary for JSON serialization.

        Returns
        -------
        dict
            Dictionary representation of the conversation entry.
        """
        return {
            "tick": self.tick,
            "timestamp": self.timestamp,
            "mode": self.mode,
            "prompt": self.prompt,
            "output": self.output,
            "prompt_timestamp": self.prompt_timestamp,
            "output_timestamp": self.output_timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ConversationEntry":
        """
        Create a ConversationEntry from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing conversation entry data.

        Returns
        -------
        ConversationEntry
            A new ConversationEntry instance.
        """
        return cls(
            tick=data["tick"],
            timestamp=data["timestamp"],
            mode=data.get("mode"),
            prompt=data.get("prompt"),
            output=data.get("output"),
            prompt_timestamp=data.get("prompt_timestamp"),
            output_timestamp=data.get("output_timestamp"),
        )


@singleton
class ConversationHistoryProvider:
    """
    A thread-safe singleton provider for managing conversation history with disk persistence.

    This provider records prompts and outputs from conversations, automatically saves
    them to disk, and can reload history on startup. It uses atomic writes to prevent
    data corruption and handles corrupted files gracefully.

    Parameters
    ----------
    filepath : str, optional
        Path to the JSON file for persistence (default is "data/conversation_history.json").
    enable_auto_save : bool, optional
        Whether to enable automatic saving (default is True).
    auto_save_interval : int, optional
        Number of new entries before triggering auto-save (default is 10).
    """

    def __init__(
        self,
        filepath: str = "data/conversation_history.json",
        enable_auto_save: bool = True,
        auto_save_interval: int = 10,
    ):
        """
        Initialize the ConversationHistoryProvider with thread lock and storage.
        """
        self._lock: threading.Lock = threading.Lock()
        self._history: Dict[int, ConversationEntry] = {}
        self._filepath: str = filepath
        self._enable_auto_save: bool = enable_auto_save
        self._auto_save_interval: int = auto_save_interval
        self._unsaved_count: int = 0

        # Ensure data directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Try to load existing history on startup
        self._load_on_startup()

    def _load_on_startup(self) -> None:
        """
        Load existing conversation history from disk on provider initialization.

        Returns
        -------
        None
        """
        if os.path.exists(self._filepath):
            try:
                loaded_entries = self.load_from_disk()
                if loaded_entries > 0:
                    logging.info(
                        f"Loaded {loaded_entries} conversation entries from {self._filepath}"
                    )
            except Exception as error:
                logging.warning(
                    f"Could not load conversation history on startup: {error}"
                )

    def record_prompt(self, tick: int, prompt: str, mode: Optional[str] = None) -> None:
        """
        Record a prompt for a given tick.

        If an entry for this tick already exists, updates it with the prompt.
        Otherwise, creates a new entry.

        Parameters
        ----------
        tick : int
            The tick number.
        prompt : str
            The prompt text to record.
        mode : str, optional
            The operational mode during this prompt (default is None).

        Returns
        -------
        None
        """
        with self._lock:
            current_time = time.time()

            if tick in self._history:
                entry = self._history[tick]
                entry.prompt = prompt
                entry.prompt_timestamp = current_time
                if mode is not None:
                    entry.mode = mode
            else:
                entry = ConversationEntry(
                    tick=tick,
                    timestamp=current_time,
                    mode=mode,
                    prompt=prompt,
                    prompt_timestamp=current_time,
                )
                self._history[tick] = entry
                self._unsaved_count += 1

            self._auto_save()

    def record_output(self, tick: int, output: str) -> None:
        """
        Record an output for a given tick.

        If an entry for this tick already exists, updates it with the output.
        Otherwise, creates a new entry with only the output.

        Parameters
        ----------
        tick : int
            The tick number.
        output : str
            The output text to record.

        Returns
        -------
        None
        """
        with self._lock:
            current_time = time.time()

            if tick in self._history:
                entry = self._history[tick]
                entry.output = output
                entry.output_timestamp = current_time
            else:
                entry = ConversationEntry(
                    tick=tick,
                    timestamp=current_time,
                    output=output,
                    output_timestamp=current_time,
                )
                self._history[tick] = entry
                self._unsaved_count += 1

            self._auto_save()

    def _auto_save(self) -> None:
        """
        Trigger automatic save if the unsaved count reaches the threshold.

        This method is called internally after recording new entries.
        It does not use additional locking as it's always called within a lock context.

        Returns
        -------
        None
        """
        if self._enable_auto_save and self._unsaved_count >= self._auto_save_interval:
            try:
                self._save_to_disk_unlocked()
                self._unsaved_count = 0
            except Exception as error:
                logging.error(f"Auto-save failed: {error}")

    def save_to_disk(self, filepath: Optional[str] = None) -> bool:
        """
        Save the conversation history to disk using atomic writes.

        Writes to a temporary file first, then atomically renames it to prevent
        corruption if the process is interrupted.

        Parameters
        ----------
        filepath : str, optional
            Custom filepath to save to. If None, uses the default filepath.

        Returns
        -------
        bool
            True if save was successful, False otherwise.
        """
        with self._lock:
            return self._save_to_disk_unlocked(filepath)

    def _save_to_disk_unlocked(self, filepath: Optional[str] = None) -> bool:
        """
        Internal method to save to disk without acquiring the lock.

        Parameters
        ----------
        filepath : str, optional
            Custom filepath to save to.

        Returns
        -------
        bool
            True if save was successful, False otherwise.
        """
        target_path = filepath or self._filepath
        tmp_path = f"{target_path}.tmp"

        try:
            # Prepare data for serialization
            data = {
                "version": "1.0",
                "entries": [entry.to_dict() for entry in self._history.values()],
            }

            # Write to temporary file
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            # Atomic rename
            os.replace(tmp_path, target_path)

            logging.info(
                f"Saved {len(self._history)} conversation entries to {target_path}"
            )
            return True

        except Exception as error:
            logging.error(f"Failed to save conversation history: {error}")
            # Clean up temporary file if it exists
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            return False

    def load_from_disk(self, filepath: Optional[str] = None) -> int:
        """
        Load conversation history from disk.

        Handles corrupted files by backing them up and starting fresh.

        Parameters
        ----------
        filepath : str, optional
            Custom filepath to load from. If None, uses the default filepath.

        Returns
        -------
        int
            Number of entries loaded.
        """
        with self._lock:
            source_path = filepath or self._filepath

            if not os.path.exists(source_path):
                logging.debug(f"No conversation history file found at {source_path}")
                return 0

            try:
                with open(source_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Version check (for future compatibility)
                version = data.get("version", "1.0")
                if version != "1.0":
                    logging.warning(
                        f"Conversation history version mismatch: expected 1.0, got {version}"
                    )

                # Load entries
                entries_data = data.get("entries", [])
                self._history.clear()

                for entry_data in entries_data:
                    entry = ConversationEntry.from_dict(entry_data)
                    self._history[entry.tick] = entry

                self._unsaved_count = 0
                logging.info(
                    f"Loaded {len(self._history)} conversation entries from {source_path}"
                )
                return len(self._history)

            except json.JSONDecodeError as error:
                # Handle corrupted file
                backup_path = f"{source_path}.corrupted.{int(time.time())}"
                logging.error(
                    f"Corrupted conversation history file: {error}. "
                    f"Backing up to {backup_path}"
                )
                try:
                    os.rename(source_path, backup_path)
                except Exception as backup_error:
                    logging.error(f"Failed to backup corrupted file: {backup_error}")
                return 0

            except Exception as error:
                logging.error(f"Failed to load conversation history: {error}")
                return 0

    def get_history(self) -> Dict[int, ConversationEntry]:
        """
        Get a copy of the entire conversation history.

        Returns
        -------
        dict
            Dictionary mapping tick numbers to conversation entries.
        """
        with self._lock:
            return dict(self._history)

    def get_entry(self, tick: int) -> Optional[ConversationEntry]:
        """
        Get a specific conversation entry by tick number.

        Parameters
        ----------
        tick : int
            The tick number to retrieve.

        Returns
        -------
        ConversationEntry, optional
            The conversation entry if found, None otherwise.
        """
        with self._lock:
            return self._history.get(tick)

    def get_recent_entries(self, count: int = 10) -> List[ConversationEntry]:
        """
        Get the most recent conversation entries.

        Parameters
        ----------
        count : int, optional
            Number of recent entries to retrieve (default is 10).

        Returns
        -------
        list
            List of recent conversation entries, sorted by tick descending.
        """
        with self._lock:
            sorted_entries = sorted(
                self._history.values(), key=lambda e: e.tick, reverse=True
            )
            return sorted_entries[:count]

    def clear(self, delete_file: bool = False) -> None:
        """
        Clear all conversation history from memory.

        Parameters
        ----------
        delete_file : bool, optional
            If True, also delete the persistence file from disk (default is False).

        Returns
        -------
        None
        """
        with self._lock:
            self._history.clear()
            self._unsaved_count = 0

            if delete_file and os.path.exists(self._filepath):
                try:
                    os.remove(self._filepath)
                    logging.info(f"Deleted conversation history file: {self._filepath}")
                except Exception as error:
                    logging.error(
                        f"Failed to delete conversation history file: {error}"
                    )

    def get_stats(self) -> Dict:
        """
        Get statistics about the conversation history.

        Returns
        -------
        dict
            Dictionary containing statistics like total entries, unsaved count, etc.
        """
        with self._lock:
            complete_conversations = sum(
                1
                for entry in self._history.values()
                if entry.prompt is not None and entry.output is not None
            )

            return {
                "total_entries": len(self._history),
                "complete_conversations": complete_conversations,
                "unsaved_count": self._unsaved_count,
                "filepath": self._filepath,
                "auto_save_enabled": self._enable_auto_save,
                "auto_save_interval": self._auto_save_interval,
            }

    def force_save(self) -> bool:
        """
        Force an immediate save to disk, regardless of auto-save settings.

        Useful for graceful shutdown scenarios.

        Returns
        -------
        bool
            True if save was successful, False otherwise.
        """
        with self._lock:
            result = self._save_to_disk_unlocked()
            if result:
                self._unsaved_count = 0
            return result
