import asyncio
import json
import logging
import time
from queue import Empty, Queue
from typing import Dict, List, Optional

from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.asr_provider import ASRProvider
from providers.io_provider import IOProvider
from providers.sleep_ticker_provider import SleepTickerProvider


class RivaASRSensorConfig(SensorConfig):
    """
    Configuration for Riva ASR Sensor.

    Parameters
    ----------
    api_key : Optional[str]
        API Key.
    rate : int
        Sampling rate.
    chunk : int
        Chunk size.
    base_url : str
        Base URL for the ASR service.
    stream_base_url : Optional[str]
        Stream Base URL.
    microphone_device_id : Optional[str]
        Microphone Device ID.
    microphone_name : Optional[str]
        Microphone Name.
    remote_input : bool
        Whether to use remote input.
    """

    api_key: Optional[str] = Field(default=None, description="API Key")
    rate: int = Field(default=48000, description="Sampling rate")
    chunk: int = Field(default=12144, description="Chunk size")
    base_url: str = Field(
        default="wss://api-asr.openmind.org", description="Base URL for the ASR service"
    )
    stream_base_url: Optional[str] = Field(default=None, description="Stream Base URL")
    microphone_device_id: Optional[int] = Field(
        default=None, description="Microphone Device ID"
    )
    microphone_name: Optional[str] = Field(default=None, description="Microphone Name")
    remote_input: bool = Field(default=False, description="Whether to use remote input")
    enable_tts_interrupt: bool = Field(
        default=False,
        description="Enable TTS interrupt (does not mute mic during TTS playback)",
    )


class RivaASRInput(FuserInput[RivaASRSensorConfig, Optional[str]]):
    """
    Automatic Speech Recognition (ASR) input handler.

    This class manages the input stream from an ASR service, buffering messages
    and providing text conversion capabilities.
    """

    def __init__(self, config: RivaASRSensorConfig):
        """
        Initialize ASRInput instance.
        """
        super().__init__(config)

        # Buffer for storing the final output
        self.messages: List[str] = []

        # Set IO Provider
        self.descriptor_for_LLM = "Voice"
        self.io_provider = IOProvider()

        # Buffer for storing messages
        self.message_buffer: Queue[str] = Queue()

        # Initialize ASR provider
        api_key = self.config.api_key
        rate = self.config.rate
        chunk = self.config.chunk
        base_url = self.config.base_url
        stream_base_url = (
            self.config.stream_base_url
            or f"wss://api.openmind.org/api/core/teleops/stream/audio?api_key={api_key}"
        )
        microphone_device_id = self.config.microphone_device_id
        microphone_name = self.config.microphone_name
        remote_input = self.config.remote_input
        enable_tts_interrupt = self.config.enable_tts_interrupt

        self.asr: ASRProvider = ASRProvider(
            rate=rate,
            chunk=chunk,
            ws_url=base_url,
            stream_url=stream_base_url,
            device_id=microphone_device_id,
            microphone_name=microphone_name,
            remote_input=remote_input,
            enable_tts_interrupt=enable_tts_interrupt,
        )
        self.asr.start()
        self.asr.register_message_callback(self._handle_asr_message)

        # Initialize sleep ticker provider
        self.global_sleep_ticker_provider = SleepTickerProvider()

    def _handle_asr_message(self, raw_message: str):
        """
        Process incoming ASR messages.

        Parameters
        ----------
        raw_message : str
            Raw message received from ASR service
        """
        try:
            json_message: Dict = json.loads(raw_message)
            if "asr_reply" in json_message:
                asr_reply = json_message["asr_reply"]
                if len(asr_reply.split()) > 1:
                    self.message_buffer.put(asr_reply)
                    logging.info("Detected ASR message: %s", asr_reply)
        except json.JSONDecodeError:
            pass

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

    async def _raw_to_text(self, raw_input: Optional[str]) -> Optional[Message]:
        """
        Convert raw input to text format.

        Parameters
        ----------
        raw_input : Optional[str]
            Raw input to be processed

        Returns
        -------
        Optional[str]
            Processed text message or None if input is None
        """
        if raw_input is None:
            return None

        return Message(timestamp=time.time(), message=raw_input)

    async def raw_to_text(self, raw_input: Optional[str]):
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
                self.messages.append(pending_message.message)
            else:
                self.messages[-1] = f"{self.messages[-1]} {pending_message.message}"

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
INPUT: {self.descriptor_for_LLM}
// START
{self.messages[-1]}
// END
"""
        self.io_provider.add_input(
            self.descriptor_for_LLM, self.messages[-1], time.time()
        )
        self.messages = []
        return result
