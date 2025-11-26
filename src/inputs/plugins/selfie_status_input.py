# src/inputs/plugins/selfie_status_input.py

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

from inputs.base import SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider


@dataclass
class Message:
    timestamp: float
    message: str


class SelfieStatus(FuserInput[str]):
    """
    Surfaces 'SelfieStatus' lines written by the connector as a single INPUT block
    when a NEW timestamp arrives. One-shot per status.

    Examples of connector values:
      ok id=wendy
      failed reason=multiple faces=2
      failed reason=none
      failed reason=service
      failed reason=bad_id
    """

    def __init__(self, config: SensorConfig = SensorConfig()):
        super().__init__(config)
        self.io_provider = IOProvider()
        self.messages: Deque[Message] = deque(maxlen=50)
        self._last_ts_seen: float = 0.0
        self.descriptor_for_LLM = "SelfieStatus"

    async def _poll(self) -> Optional[str]:
        await asyncio.sleep(0.1)
        rec = self.io_provider.inputs.get("SelfieStatus")
        if not rec:
            return None

        ts = float(rec.timestamp or 0.0)
        if ts <= self._last_ts_seen:
            return None

        self._last_ts_seen = ts
        return rec.input

    async def _raw_to_text(self, raw_input: str) -> Message:
        return Message(timestamp=time.time(), message=raw_input)

    async def raw_to_text(self, raw_input: Optional[str]):
        if raw_input is None:
            return
        self.messages.append(await self._raw_to_text(raw_input))

    def formatted_latest_buffer(self) -> Optional[str]:
        if not self.messages:
            return None
        latest = self.messages[-1]
        block = f"""INPUT: {self.descriptor_for_LLM}
// START
{latest.message}
// END"""
        self.messages.clear()
        return block
