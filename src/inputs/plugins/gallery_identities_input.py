import asyncio
import time
from collections import deque
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Deque, Optional

from inputs.base import SensorConfig
from inputs.base.loop import FuserInput
from providers.gallery_identities_provider import GalleryIdentitiesProvider
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


class GalleryIdentities(FuserInput[str]):
    """
    Async input that adapts the GalleryIdentitiesProvider to the fuser/LLM pipeline.

    Tasks
    -----
    - Subscribe to the provider's callbacks and enqueue received text lines.
    - Poll the queue periodically (non-blocking) in `_poll()`.
    - Convert raw text into `Message` objects in `_raw_to_text()`.
    - Keep a bounded in-memory history (`self.messages`, deque with maxlen=300).
    - Produce a compact, prompt-ready block via `formatted_latest_buffer()`.
    """

    def __init__(self, config: SensorConfig = SensorConfig()):
        """Initialize the GalleryIdentities input adapter

        Subscribes to `GalleryIdentitiesProvider` and adapts its messages into
        a compact INPUT block for the LLM (“Gallery Identities …”). Uses a small
        in-memory queue and a bounded deque to hold the latest message.

        Parameters
        ----------
        config : SensorConfig, optional
            Runtime configuration. Supported (optional) fields:
            - face_http_base_url : str   Base URL of the face HTTP API (default "http://127.0.0.1:6793").
            - gallery_poll_fps   : float Polling rate in Hz (e.g., 0.5 → every 2 s).
            - http_timeout_sec   : float HTTP timeout per request (seconds).
            - descriptor_for_LLM : str   Input block label (default "Gallery Identities").

        """
        super().__init__(config)

        self.io_provider = IOProvider()

        self.messages: Deque[Message] = deque(maxlen=300)
        self.message_buffer: Queue[str] = Queue(maxsize=64)

        # Config mirrors FacePresence input naming where possible
        base_url = getattr(self.config, "face_http_base_url", "http://127.0.0.1:6793")
        fps = float(getattr(self.config, "gallery_poll_fps", 1.0))  # default 1 Hz

        self.provider: GalleryIdentitiesProvider = GalleryIdentitiesProvider(
            base_url=base_url, fps=fps, timeout_s=2.0
        )
        self._is_registered: bool = True

        self.provider.start()
        self.provider.register_message_callback(self._handle_gallery_message)

        self.descriptor_for_LLM = "Gallery Identities"

    def _handle_gallery_message(self, text_line: str) -> None:
        """
        Provider callback to enqueue one formatted gallery line.

        Parameters
        ----------
        text_line : str
            A single preformatted summary string (e.g., "total=3 ids=[alice, bob, wendy]").
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
