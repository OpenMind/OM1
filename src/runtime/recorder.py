import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional


class RuntimeRecorder:
    """Records OM1 runtime data to JSONL files."""

    def __init__(self, output_dir: str = "recordings"):
        self.output_dir = output_dir
        self._current_date: Optional[str] = None
        self._file = None

        os.makedirs(self.output_dir, exist_ok=True)
        logging.info(f"RuntimeRecorder started: dir={self.output_dir}")

    def record_tick(
        self,
        tick_num: int,
        mode: str,
        asr_input: Optional[str],
        llm_input: str,
        llm_output: list[dict[str, Any]],
        actions_executed: list[dict[str, Any]],
    ) -> None:
        """Write one tick record as a JSONL line."""
        try:
            record = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "tick": tick_num,
                "mode": mode,
                "asr_input": asr_input,
                "llm_input": llm_input,
                "llm_output": llm_output,
                "actions_executed": actions_executed,
            }
            self._write_record(json.dumps(record, ensure_ascii=False))
        except Exception:
            logging.exception("RuntimeRecorder: failed to write tick record")

    def stop(self) -> None:
        """Close the current file handle."""
        if self._file:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None
            self._current_date = None

    def _write_record(self, line: str) -> None:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if now != self._current_date:
            self.stop()
            filepath = os.path.join(self.output_dir, f"{now}.jsonl")
            self._file = open(filepath, "a", encoding="utf-8")
            self._current_date = now

        self._file.write(line + "\n")
        self._file.flush()
