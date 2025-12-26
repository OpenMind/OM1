# src/inputs/plugins/selfie_status_input.py

import asyncio
import time
from collections import deque
from typing import Deque, Optional

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider


class SelfieStatus(FuserInput[SensorConfig, Optional[str]]):
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

    def __init__(self, config: SensorConfig):
        super().__init__(config)
        self.io_provider = IOProvider()
        self.messages: Deque[Message] = deque(maxlen=50)
        self._last_ts_seen: float = 0.0
        self.descriptor_for_LLM = "SelfieStatus"

    async def _poll(self) -> Optional[str]:
        """
        Poll for new SelfieStatus messages from the IOProvider.

        Returns
        -------
        Optional[str]
            The next SelfieStatus message if a new timestamp is detected, None otherwise
        """
        await asyncio.sleep(0.1)
        rec = self.io_provider.inputs.get("SelfieStatus")
        if not rec:
            return None

        ts = float(rec.timestamp or 0.0)
        if ts <= self._last_ts_seen:
            return None

        self._last_ts_seen = ts
        return rec.input

    async def _raw_to_text(self, raw_input: Optional[str]) -> Optional[Message]:
        """
        Process raw input to generate a timestamped message.

        Parameters
        ----------
        raw_input : Optional[str]
            Raw input string to be processed

        Returns
        -------
        Optional[Message]
            A timestamped message containing the processed input
        """
        if raw_input is None:
            return None

        return Message(timestamp=time.time(), message=raw_input)

    async def raw_to_text(self, raw_input: Optional[str]):
        """
        Convert raw input to text and update message buffer.

        Parameters
        ----------
        raw_input : Optional[str]
            Raw input to be processed, or None if no input is available
        """
        if raw_input is None:
            return

        message = await self._raw_to_text(raw_input)
        if message is not None:
            self.messages.append(message)

    def formatted_latest_buffer(self) -> Optional[str]:
        """
        Format and clear the latest buffer contents.

        Returns
        -------
        Optional[str]
            Formatted string of buffer contents or None if buffer is empty
        """
        if not self.messages:
            return None
        latest = self.messages[-1]
        block = f"""INPUT: {self.descriptor_for_LLM}
// START
{latest.message}
// END"""
        self.messages.clear()
        return block
