import json
from pathlib import Path
from typing import Any, Dict, List


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_history(path: str) -> List[Dict[str, Any]]:
    """
    Load conversation history from a JSONL file.
    Each line is one JSON object.
    If the file doesn't exist, return an empty list.
    Corrupted lines are skipped safely.
    """
    p = Path(path)
    if not p.exists():
        return []

    messages: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                messages.append(json.loads(line))
            except Exception:
                continue
    return messages


def append_message(path: str, message: Dict[str, Any]) -> None:
    """
    Append a single message to a JSONL history file.
    """
    p = Path(path)
    _ensure_parent_dir(p)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(message, ensure_ascii=False) + "\n")


def default_history_path() -> str:
    """
    Default history file location:
    .om1/history/conversation_history.jsonl
    """
    return str(Path(".om1") / "history" / "conversation_history.jsonl")
