import asyncio
import logging
import time
from queue import Empty
from typing import Optional

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider
from providers.rtk_provider import RtkProvider


class Rtk(FuserInput[SensorConfig, Optional[dict]]):
    """
    Reads RTK data from RTK provider.
    """

    def __init__(self, config: SensorConfig):
        super().__init__(config)

        self.rtk = RtkProvider()
        self.io_provider = IOProvider()
        self.messages: list[Message] = []
        self.descriptor_for_LLM = "Precision Location"

    async def _poll(self) -> Optional[dict]:
        """
        Poll for new messages from the RTK Provider.

        Checks the message buffer for new messages with a brief delay
        to prevent excessive CPU usage.

        Returns
        -------
        Optional[dict]
            The next message from the buffer if available, None otherwise
        """
        await asyncio.sleep(0.5)

        try:
            return self.rtk.data
        except Empty:
            return None

    async def _raw_to_text(self, raw_input: Optional[dict]) -> Optional[Message]:
        """
        Process raw input to generate a timestamped message.

        Creates a Message object from the raw input, adding
        the current timestamp.

        Parameters
        ----------
        raw_input : Optional[dict]
            Raw input to be processed

        Returns
        -------
        Message
            A timestamped message containing the processed input
        """
        logging.debug(f"rtk: {raw_input}")

        r = raw_input
        if r:
            logging.debug(f"RTK Provider: {r}")
            lat = r["rtk_lat"]
            lon = r["rtk_lon"]
            alt = r["rtk_alt"]
            qua = r["rtk_qua"]

            lat_string = "South"
            if lat > 0:
                lat_string = "North"
            else:
                lat *= -1.0

            lon_string = "West"
            if lon > 0:
                lon_string = "East"
            else:
                lon *= -1.0

            if qua > 0:
                msg = f"Your precise location is {lat} {lat_string}, {lon} {lon_string} at {alt}m altitude. "
                return Message(timestamp=time.time(), message=msg)
            else:
                return None
        else:
            return None

    async def raw_to_text(self, raw_input: Optional[dict]):
        """
        Update message buffer.
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

        result = (
            f"\nINPUT: {self.descriptor_for_LLM}\n// START\n"
            f"{latest_message.message}\n// END\n"
        )

        self.io_provider.add_input(
            self.__class__.__name__, latest_message.message, latest_message.timestamp
        )
        self.messages = []

        return result
