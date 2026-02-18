import asyncio
import time
from queue import Empty, Queue
from typing import Any, List, Optional, TypeVar

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider

C = TypeVar("C", bound=SensorConfig)


class BaseVLMFuserInput(FuserInput[C, Optional[str]]):
    """Base class for API-based VLM input plugins.

    Provides shared polling, message buffering, text conversion,
    and output formatting logic. Subclasses only need to implement
    provider-specific initialization and message parsing.
    """

    DESCRIPTOR_FOR_LLM: str = "Vision"

    def __init__(self, config: C):
        super().__init__(config)
        self.io_provider = IOProvider()
        self.messages: List[Message] = []
        self.message_buffer: Queue[str] = Queue()
        self.descriptor_for_LLM = self.DESCRIPTOR_FOR_LLM
        self.vlm: Any = None

    async def _poll(self) -> Optional[str]:
        """Poll the message buffer for new VLM responses."""
        await asyncio.sleep(0.5)
        try:
            message = self.message_buffer.get_nowait()
            return message
        except Empty:
            return None

    async def _raw_to_text(self, raw_input: Optional[str]) -> Optional[Message]:
        """Convert a raw string into a timestamped Message."""
        if raw_input is None:
            return None
        return Message(timestamp=time.time(), message=raw_input)

    async def raw_to_text(self, raw_input: Optional[str]):
        """Convert raw input to text and append to message buffer."""
        if raw_input is None:
            return
        pending_message = await self._raw_to_text(raw_input)
        if pending_message is not None:
            self.messages.append(pending_message)

    def formatted_latest_buffer(self) -> Optional[str]:
        """Format and clear the latest buffer contents."""
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

    def stop(self) -> None:
        """Stop the VLM provider."""
        if self.vlm:
            self.vlm.stop()
