import asyncio
import json
import logging
import time
from queue import Empty, Queue
from typing import List, Optional

import zenoh

from inputs.base import SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider
from providers.sleep_ticker_provider import SleepTickerProvider
from providers.zenoh_listener_provider import ZenohListenerProvider


class ZenohListener(FuserInput[str]):
    """
    Zenoh listener handler.

    This class manages the input stream from Zenoh, buffering messages
    and providing text conversion capabilities.
    """

    def __init__(self, config: SensorConfig = SensorConfig()):
        """
        Initialize the ZenohListener instance.
        """
        super().__init__(config)

        # Buffer for storing the final output
        self.messages: List[str] = []

        # Set IO Provider
        self.descriptor_for_message = "Message"
        self.io_provider = IOProvider()

        # Buffer for storing messages
        self.message_buffer: Queue[str] = Queue()

        # Initialize ZenohListenerProvider provider
        listen_topic = getattr(self.config, "listen_topic", None)
        if listen_topic is None:
            listen_topic = "speech"
            # Log the listen_topic being used
            logging.info(f"Listen topic not provided. Using default topic: {listen_topic}")

        self.listener: ZenohListenerProvider = ZenohListenerProvider(
            topic=listen_topic,
        )
        self.listener.start(self._handle_zenoh_message)

        # Initialize sleep ticker provider
        self.global_sleep_ticker_provider = SleepTickerProvider()

    def _handle_zenoh_message(self, zenoh_input: zenoh.Sample):
        """
        Process an incoming Zenoh message.

        Parameters
        ----------
        zenoh_input : object
            The Zenoh sample received, which should have a 'payload' attribute.
        """
        try:
            zenoh_message = json.loads(zenoh_input.payload.to_string())
            if "message" in zenoh_message:
                heard_message = zenoh_message["message"]
                self.message_buffer.put(heard_message)
                logging.info("Heard Zenoh message: %s", heard_message)
            else:
                logging.error("Deserialized payload does not have message")
        except Exception as e:
            logging.error(f"Error hearing: {e}")

    async def _poll(self) -> Optional[str]:
        """
        Poll for new messages in the buffer.

        Returns
        -------
        Optional[str]
            Message from the buffer if available, None otherwise
        """
        await asyncio.sleep(0.1)
        try:
            message = self.message_buffer.get_nowait()
            return message
        except Empty:
            return None

    async def _raw_to_text(self, raw_input: str) -> str:
        """
        Convert raw input to text format.

        Parameters
        ----------
        raw_input : str
            Raw input string to be converted

        Returns
        -------
        Optional[str]
            Converted text or None if conversion fails
        """
        return raw_input

    async def raw_to_text(self, raw_input: str):
        """
        Convert raw input to processed text and manage buffer.

        Parameters
        ----------
        raw_input : Optional[str]
            Raw input to be processed
        """
        pending_message = await self._raw_to_text(raw_input)
        if pending_message is None:
            if len(self.messages) != 0:
                # Skip sleep if there's already a message in the messages buffer
                self.global_sleep_ticker_provider.skip_sleep = True

        if pending_message is not None:
            if len(self.messages) == 0:
                self.messages.append(pending_message)
            else:
                self.messages[-1] = f"{self.messages[-1]} {pending_message}"

    def formatted_latest_buffer(self) -> Optional[str]:
        """
        Format and clear the latest buffer contents.

        Returns
        -------
        Optional[str]
            Formatted string of buffer contents or None if buffer is empty
        """
        if len(self.messages) == 0:
            return None

        result = f"""
{self.descriptor_for_message} INPUT
// START
{self.messages[-1]}
// END
"""
        self.io_provider.add_input(self.descriptor_for_message, self.messages[-1], time.time())
        self.messages = []
        return result
