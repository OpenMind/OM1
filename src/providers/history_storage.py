"""
Simple JSON storage utilities for conversation history.
Implements Issue #985 with minimal changes.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

# Configuration
HISTORY_FILE = Path("data/conversation_history.json")


def save_history(history_data: List[Dict[str, str]]) -> bool:
    """
    Save conversation history to disk as JSON.

    Parameters
    ----------
    history_data : List[Dict[str, str]]
        List of message dictionaries with 'role' and 'content' keys

    Returns
    -------
    bool
        True if save was successful, False otherwise
    """
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)

        logging.info(f"Saved conversation history to {HISTORY_FILE}")
        return True

    except (IOError, OSError) as e:
        logging.error(f"Failed to save conversation history: {e}")
        return False


def load_history() -> List[Dict[str, str]]:
    """
    Load conversation history from disk.

    Returns
    -------
    List[Dict[str, str]]
        List of message dictionaries, or empty list if file doesn't exist
    """
    if not HISTORY_FILE.exists():
        logging.debug(f"No existing conversation history found at {HISTORY_FILE}")
        return []

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        logging.info(f"Loaded {len(data)} messages from conversation history")
        return data

    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in conversation history file: {e}")
        return []
    except (IOError, OSError) as e:
        logging.error(f"Failed to load conversation history: {e}")
        return []


def history_exists() -> bool:
    """
    Check if conversation history file exists.

    Returns
    -------
    bool
        True if history file exists, False otherwise
    """
    return HISTORY_FILE.exists()
