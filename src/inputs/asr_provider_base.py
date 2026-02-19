import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, TypeVar
from uuid import uuid4

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider
from providers.sleep_ticker_provider import SleepTickerProvider
from providers.teleops_conversation_provider import TeleopsConversationProvider
from zenoh_msgs import ASRText, open_zenoh_session, prepare_header

C = TypeVar("C", bound=SensorConfig)


class BaseASRFuserInput(FuserInput[C, Optional[str]]):
    """Base class for API-based ASR input plugins.

    Provides shared message handling, polling, text conversion, buffer
    formatting, and Zenoh publishing logic used by all API-based ASR
    input plugins (Google ASR, Google ASR RTSP, Riva ASR, Riva ASR RTSP).

    Subclasses only need to define their Config class and set up the
    provider-specific ASR instance in ``__init__``.
    """

    DESCRIPTOR_FOR_LLM: str = "Voice"

    def __init__(self, config: C):
        super().__init__(config)

        self.messages: List[str] = []
        self.descriptor_for_LLM = self.DESCRIPTOR_FOR_LLM
        self.io_provider = IOProvider()
        self.message_buffer: asyncio.Queue[str] = asyncio.Queue()
        self.asr: Any = None

        self.global_sleep_ticker_provider = SleepTickerProvider()

        api_key = getattr(config, "api_key", None)
        self.conversation_provider = TeleopsConversationProvider(api_key=api_key)

        self.asr_topic = "om/asr/text"
        self.session = None
        self.asr_publisher = None

        try:
            self.session = open_zenoh_session()
            self.asr_publisher = self.session.declare_publisher(self.asr_topic)
            logging.info("Zenoh ASR publisher initialized on topic 'om/asr/text'")
        except Exception as e:
            logging.warning(f"Could not initialize Zenoh for ASR broadcast: {e}")
            self.session = None
            self.asr_publisher = None

    def _handle_asr_message(self, raw_message: str):
        """Process incoming ASR messages.

        Parameters
        ----------
        raw_message : str
            Raw JSON message received from ASR service
        """
        try:
            json_message: Dict = json.loads(raw_message)
            if "asr_reply" in json_message:
                asr_reply = json_message["asr_reply"]
                if len(asr_reply.split()) > 1:
                    self.message_buffer.put_nowait(asr_reply)
                    logging.info("Detected ASR message: %s", asr_reply)
        except json.JSONDecodeError:
            pass

    async def _poll(self) -> Optional[str]:
        """Poll for new messages in the buffer.

        Returns
        -------
        Optional[str]
            Message from the buffer if available, None otherwise
        """
        try:
            message = self.message_buffer.get_nowait()
            return message
        except asyncio.QueueEmpty:
            await asyncio.sleep(0.01)
            return None

    async def _raw_to_text(self, raw_input: Optional[str]) -> Optional[Message]:
        """Convert raw input to text format.

        Parameters
        ----------
        raw_input : Optional[str]
            Raw input to be processed

        Returns
        -------
        Optional[Message]
            Processed message or None if input is None
        """
        if raw_input is None:
            return None

        return Message(timestamp=time.time(), message=raw_input)

    async def raw_to_text(self, raw_input: Optional[str]):
        """Convert raw input to processed text and manage buffer.

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
                self.messages.append(pending_message.message)
            else:
                self.messages[-1] = f"{self.messages[-1]} {pending_message.message}"

    def formatted_latest_buffer(self) -> Optional[str]:
        """Format and clear the latest buffer contents.

        Returns
        -------
        Optional[str]
            Formatted string of buffer contents or None if buffer is empty
        """
        if len(self.messages) == 0:
            return None

        result = f"""
INPUT: {self.descriptor_for_LLM}
// START
{self.messages[-1]}
// END
"""
        # Add to IO provider and conversation provider
        self.io_provider.add_input(
            self.descriptor_for_LLM, self.messages[-1], time.time()
        )
        self.io_provider.add_mode_transition_input(self.messages[-1])
        self.conversation_provider.store_user_message(self.messages[-1])

        # Publish to Zenoh
        if self.asr_publisher:
            try:
                asr_msg = ASRText(
                    header=prepare_header(str(uuid4())),
                    text=self.messages[-1],
                )
                self.asr_publisher.put(asr_msg.serialize())
                logging.info(f"Published ASR to Zenoh: {self.messages[-1]}")
            except Exception as e:
                logging.warning(f"Failed to publish ASR to Zenoh: {e}")

        # Reset messages buffer
        self.messages = []
        return result

    def stop(self):
        """Stop the ASR input and clean up resources."""
        if self.asr:
            self.asr.stop()

        if self.session:
            self.session.close()
            logging.info("Zenoh ASR session closed")
