"""
Simple JSON persistence for conversation history.

Provides atomic save/load operations for chat history.
"""

import json
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional


def save_history(
    history: List[Dict[str, str]],
    filepath: str = "data/conversation_history.json",
) -> bool:
    """
    Save conversation history to a JSON file atomically.

    Uses temp file + rename strategy to prevent corruption.

    Parameters
    ----------
    history : List[Dict[str, str]]
        List of message dictionaries with "role" and "content" keys.
    filepath : str
        Path to the JSON file.

    Returns
    -------
    bool
        True if save was successful, False otherwise.
    """
    try:
        # Ensure directory exists
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        # Write to temp file first (atomic write)
        fd, temp_path = tempfile.mkstemp(
            suffix=".json",
            prefix="history_",
            dir=directory if directory else None,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)

            # Atomic rename
            os.replace(temp_path, filepath)
            logging.debug(f"Conversation history saved to {filepath}")
            return True

        except Exception:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    except Exception as e:
        logging.error(f"Failed to save conversation history: {e}")
        return False


def load_history(
    filepath: str = "data/conversation_history.json",
) -> List[Dict[str, str]]:
    """
    Load conversation history from a JSON file.

    Parameters
    ----------
    filepath : str
        Path to the JSON file.

    Returns
    -------
    List[Dict[str, str]]
        List of message dictionaries, or empty list if file doesn't exist
        or is corrupted.
    """
    if not os.path.exists(filepath):
        logging.debug(f"No history file found at {filepath}, starting fresh")
        return []

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Validate structure
        if not isinstance(data, list):
            logging.warning(f"Invalid history format in {filepath}, starting fresh")
            return []

        # Validate each message has required fields
        valid_history = []
        for msg in data:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                valid_history.append({"role": msg["role"], "content": msg["content"]})
            else:
                logging.warning(f"Skipping invalid message: {msg}")

        logging.info(f"Loaded {len(valid_history)} messages from {filepath}")
        return valid_history

    except json.JSONDecodeError as e:
        logging.warning(f"Corrupted history file {filepath}: {e}, starting fresh")
        return []
    except Exception as e:
        logging.error(f"Failed to load conversation history: {e}")
        return []


def clear_history(filepath: str = "data/conversation_history.json") -> bool:
    """
    Delete the conversation history file.

    Parameters
    ----------
    filepath : str
        Path to the JSON file.

    Returns
    -------
    bool
        True if file was deleted or didn't exist, False on error.
    """
    try:
        if os.path.exists(filepath):
            os.unlink(filepath)
            logging.info(f"Conversation history cleared: {filepath}")
        return True
    except Exception as e:
        logging.error(f"Failed to clear conversation history: {e}")
        return False
