import asyncio
import logging
import time
from collections import deque
from queue import Empty, Queue
from typing import Deque, Optional

from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.face_presence_provider import FacePresenceProvider
from providers.io_provider import IOProvider


class FacePresenceConfig(SensorConfig):
    """
    Configuration for Face Presence Sensor.

    Parameters
    ----------
    face_http_base_url : str
        Base URL for the Face HTTP service.
    face_recent_sec : float
        Time window in seconds to consider a face present.
    face_poll_fps : float
        Polling frequency in frames per second.
    """

    face_http_base_url: str = Field(
        default="http://127.0.0.1:6793",
        description="Base URL for the Face HTTP service",
    )
    face_recent_sec: float = Field(default=1.0, description="Time window in seconds to consider a face present")
    face_poll_fps: float = Field(default=5.0, description="Polling frequency in frames per second")


class FacePresence(FuserInput[FacePresenceConfig, Optional[str]]):
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

    def __init__(self, config: FacePresenceConfig):
        """
        Initialize the face presence input.

        Parameters
        ----------
        config : FacePresenceConfig
            Configuration settings for the sensor input.
        """
        super().__init__(config)

        self.io_provider = IOProvider()

        self.messages: Deque[Message] = deque(maxlen=300)

        self.message_buffer: Queue[str] = Queue(maxsize=64)

        # Read config and construct the provider WITH required args
        base_url = self.config.face_http_base_url
        recent_sec = self.config.face_recent_sec
        fps = self.config.face_poll_fps

        self.provider: FacePresenceProvider = FacePresenceProvider(
            base_url=base_url, recent_sec=recent_sec, fps=fps, timeout_s=2.0
        )
        self._is_registered: bool = True

        self.provider.start()
        self.provider.register_message_callback(self._handle_face_message)

        self.descriptor_for_LLM = "Face Presence Sensor"

    def _handle_face_message(self, text_line: str) -> None:
        """
        Provider callback: push a new line into the bounded queue, clearing old messages.

        Parameters
        ----------
        text_line : str
            A single, already formatted line (e.g., "present=[alice], unknown=0, ts=...").
        """
        while not self.message_buffer.empty():
            try:
                _ = self.message_buffer.get_nowait()
            except Empty:
                break

        try:
            self.message_buffer.put_nowait(text_line)
        except Exception as e:
            logging.warning(f"Failed to enqueue face presence message: {e}")

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

    async def _raw_to_text(self, raw_input: Optional[str]) -> Optional[Message]:
        """
        Process raw input to generate a timestamped message.

        Creates a Message object from the raw input string, adding
        the current timestamp.

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

        Processes the raw input if present and adds the resulting
        message to the internal message buffer.

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
{self.descriptor_for_LLM}: "{latest_message.message}"
"""

        self.io_provider.add_input(self.__class__.__name__, latest_message.message, latest_message.timestamp)
        self.messages.clear()
        return result

    def stop(self):
        """
        Stop the provider and clean up resources.

        Unregisters callbacks, stops the provider, and ensures that
        all resources are released properly.
        """
        try:
            self.provider.unregister_message_callback(self._handle_face_message)
            logging.info("Unregistered face presence message callback")
        except Exception as e:
            logging.warning(f"Failed to unregister face presence callback: {e}")
