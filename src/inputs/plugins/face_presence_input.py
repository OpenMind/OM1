# src/inputs/face_presence_input.py
""" Inputs 
This module exposes a thin, consumer-oriented interface for reading the latest "who is present now"
signal produced by :class: 'FacePresenceProvider'. Here we simply transform the provider's buffer entries 
into a small value object that's convenient for downstream consumers (e.g,LLM/fuser) to use
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from providers.face_presence_provider import FacePresenceProvider, PresenceSnapshot


@dataclass
class FacePresenceReading:
    """Value object exposed to the rest of the system / LLM.
    Attributes
    ----------
    text : str
        Human-readable summary of the presence snapshot, e.g.
        ``"present: [alice, bob], unknown=0 @ 1728361234.567"``.
    snapshot : PresenceSnapshot
        The full structured snapshot (names, unknown count, timestamps, raw JSON)
        for downstream logic that needs machine-readable fields.
    """
    text: str
    snapshot: PresenceSnapshot


class FacePresenceInput:
    """
    Thin input facade for consumers:
      - get_latest(): fetch and clear backlog (return newest only)
      - peek(): read the newest without clearing
    """

    def __init__(self, provider: FacePresenceProvider) -> None:
        self._provider = provider

    def get_latest(self) -> Optional[FacePresenceReading]:
        """Return the newest presence reading and clear any backlog."""
        item = self._provider.buffer.drain_latest()
        if not item:
            return None
        txt = item.value.to_text()
        return FacePresenceReading(text=txt, snapshot=item.value)

    def peek(self) -> Optional[FacePresenceReading]:
        """Return the current newest presence reading without clearing the buffer."""
        item = self._provider.buffer.peek_latest()
        if not item:
            return None
        return FacePresenceReading(text=item.value.to_text(), snapshot=item.value)
