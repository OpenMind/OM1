import asyncio
import random
import time
from typing import Optional

from PIL import Image

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider


class DummyVLMLocal(FuserInput[SensorConfig, Image.Image]):
    """
    Vision Language Model input handler.

    Simulates processing of image inputs and generates dummy text descriptions.
    Maintains a buffer of processed messages.
    """

    def __init__(self, config: SensorConfig):
        """
        Initialize VLM input handler with empty message buffer.
        """
        super().__init__(config)

        # Track IO
        self.io_provider = IOProvider()

        # Messages buffer
        self.messages: list[Message] = []

        self.descriptor_for_LLM = "Vision"

    async def _poll(self) -> Image.Image:
        """
        Poll for new image input.

        Currently generates random colored images for testing.
        In production, this would interface with camera or sensor.

        Returns
        -------
        Image.Image
            Generated or captured image
        """
        await asyncio.sleep(0.5)

        # this could be a camera or other sensor
        img = Image.new(
            "RGB",
            (100, 100),
            (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
        )
        return img

    async def _raw_to_text(self, raw_input: Image.Image) -> Message:
        """
        Process raw image input to generate text description.

        Parameters
        ----------
        raw_input : Image.Image
            Input image to process

        Returns
        -------
        Message
            Timestamped message containing description
        """
        # You can use the `raw_input` variable for something, it is of Type Image
        # But for simplicity let's just create a string that changes
        num = random.randint(0, 100)
        message = f"DUMMY VLM - FAKE DATA - I see {num} people. Also, I see a rocket."

        return Message(timestamp=time.time(), message=message)

    async def raw_to_text(self, raw_input: Image.Image):
        """
        Convert raw image to text and update message buffer.

        Parameters
        ----------
        raw_input : Image.Image
            Raw image to be processed
        """
        pending_message = await self._raw_to_text(raw_input)

        if pending_message is not None:
            self.messages.append(pending_message)

    def formatted_latest_buffer(self) -> Optional[str]:
        """
        Format and clear the latest buffer contents.

        Formats the most recent message with timestamp and class name,
        adds it to the IO provider, then clears the buffer.

        Returns
        -------
        Optional[str]
            Formatted string of buffer contents or None if buffer is empty
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
            self.descriptor_for_LLM, latest_message.message, latest_message.timestamp
        )
        self.messages = []

        return result
