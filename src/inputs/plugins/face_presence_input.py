import asyncio
import time
from dataclasses import dataclass
from queue import Queue
from typing import List, Optional

from inputs.base import SensorConfig
from inputs.base.loop import FuserInput
from providers.face_presence_provider import FacePresenceProvider
from providers.io_provider import IOProvider


@dataclass
class Message:
    """
    Container for timestamped messages.

    Parameters
    ----------
    timestamp : float
        Unix timestamp of the message
    message : str
        Content of the message
    """

    timestamp: float
    message: str


class FacePresenceInput(FuserInput[str]):
    """
    Async input facade (mirrors the style of dimo_tesla / vlm_local_yolo):

      - start()/stop(): manage an internal polling coroutine that samples the provider
        at a fixed interval (default 0.2s) and caches the latest reading.

      - get_latest(): return newest item and clear older provider entries.
      - peek(): non-destructive read of the newest item.
      - formatted_latest_buffer(): compact string (or multi-line) for LLM prompts.

    This class does *not* talk to HTTP directly; it consumes the provider’s buffer.
    """

    def __init__(self, config: SensorConfig = SensorConfig()) -> None:

        super().__init__(config)

        # Track IO
        self.io_provider = IOProvider()

        base_url = getattr(self.config, "base_url", "http://127.0.0.1:6793")

        self.face_presence_provider = FacePresenceProvider(base_url=base_url)
        self.face_presence_provider.start()

        self.messages: List[Message] = []

        self.message_buffer: Queue[str] = Queue()

        self.descriptor_for_LLM = "Face Presence Sensor"

    async def _poll(self) -> Optional[str]:
        """
        Poll for new messages in the buffer.

        Returns
        -------
        Optional[str]
            Message from the buffer if available, None otherwise
        """
        await asyncio.sleep(0.2)
        try:
            return self.face_presence_provider.peek_latest()
        except asyncio.QueueEmpty:
            return None

    async def _raw_to_text(self, raw_input: str) -> Message:
        """
        Process raw input to generate a timestamped message.

        Creates a Message object from the raw input string, adding
        the current timestamp.

        Parameters
        ----------
        raw_input : str
            Raw input string to be processed

        Returns
        -------
        Message
            A timestamped message containing the processed input
        """
        return Message(timestamp=time.time(), message=raw_input)

    async def raw_to_text(self, raw_input: Optional[str]):
        """
        Convert raw input to text and update message buffer.

        Processes the raw input if present and adds the resulting
        message to the internal message buffer.

        Parameters
        ----------
        raw_input : Optional[str]
            Raw input to be processed, or None if no input is available
        """
        if raw_input is None:
            return

        pending_message = await self._raw_to_text(raw_input)

        if pending_message is not None:
            self.messages.append(pending_message)

    def formatted_latest_buffer(self) -> Optional[str]:
        """
        Format and clear the latest buffer contents.

        Retrieves the most recent message from the buffer, formats it
        with timestamp and class name, adds it to the IO provider,
        and clears the buffer.

        Returns
        -------
        Optional[str]
            Formatted string containing the latest message and metadata,
            or None if the buffer is empty

        """
        if len(self.messages) == 0:
            return None

        latest_message = self.messages[-1]

        result = f"""
INPUT: {self.descriptor_for_LLM}
// START
{latest_message.message}
// END
"""

        self.io_provider.add_input(
            self.__class__.__name__, latest_message.message, latest_message.timestamp
        )
        self.messages = []

        return result
