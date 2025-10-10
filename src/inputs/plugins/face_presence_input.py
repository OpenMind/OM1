import asyncio
import time
from collections import deque
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Deque, Optional

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


class FacePresence(FuserInput[str]):
    """
    Async input that adapts the FacePresenceProvider to the fuser/LLM pipeline.

    Tasks
    ----------------
    - Subscribe to the provider's callbacks and enqueue received text lines.
    - Poll the queue periodically (non-blocking) in `_poll()`.
    - Convert raw text into `Message` objects in `_raw_to_text()`.
    - Keep a bounded in-memory history (`self.messages`, deque with maxlen=300).
    - Produce a compact, prompt-ready block via `formatted_latest_buffer()`.
    """

    def __init__(self, config: SensorConfig = SensorConfig()):
        """
        Initialize the face presence input
        """
        super().__init__(config)

        self.io_provider = IOProvider()

        self.messages: Deque[Message] = deque(maxlen=300)

        self.message_buffer: Queue[str] = Queue(maxsize=64)

        # Read config and construct the provider WITH required args
        base_url = getattr(self.config, "face_http_base_url", "http://127.0.0.1:6793")
        recent_sec = float(getattr(self.config, "face_recent_sec", 2.0))
        fps = float(getattr(self.config, "face_poll_fps", 5.0))

        self.provider: FacePresenceProvider = FacePresenceProvider(
            base_url=base_url, recent_sec=recent_sec, fps=fps, timeout_s=2.0
        )
        self._is_registered: bool = True

        self.provider.start()
        self.provider.register_message_callback(self._handle_face_message)

        self.descriptor_for_LLM = "Face Presence Sensor"

    def _handle_face_message(self, text_line: str) -> None:
        """
        Provider callback: push a new line into the bounded queue.

        Tasks
        --------
        - Tries a non-blocking enqueue into `self.message_buffer` (capacity=64).
        - If the queue is full, drops one oldest item and retries once.

        Parameters
        ----------
        text_line : str
            A single, already formatted line (e.g., "present=[alice], unknown=0, ts=...").
        """
        try:
            self.message_buffer.put_nowait(text_line)
        except Exception:
            try:
                _ = self.message_buffer.get_nowait()
            except Empty:
                pass
            try:
                self.message_buffer.put_nowait(text_line)
            except Exception:
                pass

    async def _poll(self) -> Optional[str]:
        """
        Poll for new messages from the face presence service.

        Checks the message buffer for new messages with a brief delay
        to prevent excessive CPU usage.

        Returns
        -------
        Optional[str]
            The next message from the buffer if available, None otherwise
        """
        await asyncio.sleep(0.5)
        try:
            return self.message_buffer.get_nowait()
        except Empty:
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

        self.messages.append(await self._raw_to_text(raw_input))

    def formatted_latest_buffer(self) -> Optional[str]:
        """
        Return the newest message as a compact prompt result and clear history.

        Returns
        -------
        str or None
            A formatted multi-line string ready for LLM consumption, or None if there
            are no messages.

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
        self.messages.clear()
        return result
