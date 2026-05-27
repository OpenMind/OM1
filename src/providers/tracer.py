import atexit
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

from providers.singleton import singleton


@singleton
class Tracer:
    """
    Tracer for LLM interactions. Writes one JSONL line per record, with a daily rotation.
    """

    def __init__(self, output_dir: str = "traces") -> None:
        """
        Initialize the Tracer.

        Parameters
        ----------
        output_dir : str
            Directory where traces will be saved (default: "traces").
        """
        self.output_dir = output_dir
        self._enabled = False
        self._current_date: Optional[str] = None
        self._file: Any = None
        self.generation: int = 0

        if self._enabled:
            os.makedirs(self.output_dir, exist_ok=True)

        atexit.register(self.stop)

    def enable(self) -> None:
        """
        Enable tracing and ensure the output directory exists.
        """
        self._enabled = True
        os.makedirs(self.output_dir, exist_ok=True)

    def disable(self) -> None:
        """
        Disable tracing and close any open file handle.
        """
        self._enabled = False
        self.stop()

    @property
    def enabled(self) -> bool:
        """
        Return whether tracing is currently enabled.
        """
        return self._enabled

    def gauge(self, llm_input: str, llm_output: list[dict[str, Any]]) -> None:
        """
        Record an LLM interaction with the given input and output.

        Parameters
        ----------
        llm_input : str
            The input prompt sent to the LLM.

        llm_output : list[dict[str, Any]]
            The output from the LLM, typically a list of message dictionaries.
        """
        if not self._enabled:
            return

        try:
            data = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "generation": self.generation,
                "llm_input": llm_input,
                "llm_output": llm_output,
            }
            self._write(json.dumps(data, ensure_ascii=False))
        except Exception:
            logging.exception("tracer: failed to write record")

    def stop(self) -> None:
        """
        Close the current file handle.
        """
        if self._file:
            try:
                self._file.close()
            except Exception:
                pass

            self._file = None
            self._current_date = None

    def _write(self, line: str) -> None:
        """
        Write a line to the current day's JSONL file, rotating if the date has changed.

        Parameters
        ----------
        line : str
            The JSON string to write as a line in the file.
        """
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        if now != self._current_date:
            self.stop()
            filepath = os.path.join(self.output_dir, f"{now}.jsonl")
            self._file = open(filepath, "a", encoding="utf-8")
            self._current_date = now

        self._file.write(line + "\n")
        self._file.flush()

    def set_generation(self, generation: int) -> None:
        """
        Set the current generation number for tracing.

        Parameters
        ----------
        generation : int
            The generation number to set.
        """
        self.generation = generation
