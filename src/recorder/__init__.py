import atexit
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

enable_recording = os.environ.get("OM1_ENABLE_RECORDING", "false").lower() == "true"
output_dir = "recordings"
current_date = None
file = None
generation = 0

if enable_recording:
    os.makedirs(output_dir, exist_ok=True)


def record(llm_input: str, llm_output: list[dict[str, Any]]) -> None:
    """Write one record as a JSONL line."""
    if not enable_recording:
        return
    try:
        data = {
            "generation": generation,
            "ts": datetime.now(timezone.utc).isoformat(),
            "llm_input": llm_input,
            "llm_output": llm_output,
        }
        _write(json.dumps(data, ensure_ascii=False))
    except Exception:
        logging.exception("recorder: failed to write record")


def stop() -> None:
    """Close the current file handle."""
    global file, current_date
    if file:
        try:
            file.close()
        except Exception:
            pass
        file = None
        current_date = None


def _write(line: str) -> None:
    global file, current_date
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if now != current_date:
        stop()
        filepath = os.path.join(output_dir, f"{now}.jsonl")
        file = open(filepath, "a", encoding="utf-8")
        current_date = now

    file.write(line + "\n")
    file.flush()


atexit.register(stop)
