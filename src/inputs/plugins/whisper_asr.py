import asyncio
import json
import logging
import time
from queue import Empty, Queue
from typing import Dict, List, Optional
from uuid import uuid4

from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider
from providers.sleep_ticker_provider import SleepTickerProvider
from providers.teleops_conversation_provider import TeleopsConversationProvider
from providers.whisper_asr_provider import WhisperASRProvider
from zenoh_msgs import ASRText, open_zenoh_session, prepare_header


class WhisperASRSensorConfig(SensorConfig):
    """
    Configuration for Whisper ASR Sensor.

    Parameters
    ----------
    model_size : str
        Whisper model size to use.
    device : str
        Compute device for inference.
    compute_type : str
        Compute precision type.
    language : Optional[str]
        Language code for transcription. None for auto-detection.
    microphone_device_id : Optional[int]
        Microphone device ID.
    microphone_name : Optional[str]
        Microphone device name.
    rate : int
        Audio sampling rate.
    chunk : int
        Audio chunk size.
    enable_tts_interrupt : bool
        Whether to enable TTS interrupt.
    api_key : Optional[str]
        API key for TeleopsConversationProvider.
    """

    model_size: str = Field(
        default="turbo",
        description="Whisper model: tiny, base, small, medium, large-v3, turbo",
    )
    device: str = Field(default="auto", description="Compute device: cuda, cpu, auto")
    compute_type: str = Field(
        default="auto",
        description="Compute precision: float16, int8, float32, auto",
    )
    language: Optional[str] = Field(
        default=None,
        description="Language code (e.g. en, tr, ja). None for auto-detection",
    )
    microphone_device_id: Optional[int] = Field(
        default=None, description="Microphone Device ID"
    )
    microphone_name: Optional[str] = Field(default=None, description="Microphone Name")
    rate: int = Field(default=16000, description="Audio sampling rate")
    chunk: int = Field(default=3200, description="Audio chunk size")
    enable_tts_interrupt: bool = Field(
        default=False,
        description="Enable TTS interrupt (does not mute mic during TTS playback)",
    )
    api_key: Optional[str] = Field(default=None, description="API Key")


class WhisperASRInput(FuserInput[WhisperASRSensorConfig, Optional[str]]):
    """
    Local Whisper Automatic Speech Recognition (ASR) input handler.

    This class manages the input stream from a local Whisper model,
    buffering messages and providing text conversion capabilities.
    No cloud API dependency required.
    """

    def __init__(self, config: WhisperASRSensorConfig):
        """
        Initialize WhisperASRInput instance.

        Parameters
        ----------
        config : WhisperASRSensorConfig
            Configuration for the Whisper ASR input.
        """
        super().__init__(config)

        # Buffer for storing the final output
        self.messages: List[str] = []

        # Set IO Provider
        self.descriptor_for_LLM = "Voice"
        self.io_provider = IOProvider()

        # Buffer for storing messages
        self.message_buffer: Queue[str] = Queue()

        # Initialize Whisper ASR provider
        self.asr: WhisperASRProvider = WhisperASRProvider(
            model_size=self.config.model_size,
            device=self.config.device,
            compute_type=self.config.compute_type,
            language=self.config.language,
            device_id=self.config.microphone_device_id,
            microphone_name=self.config.microphone_name,
            rate=self.config.rate,
            chunk=self.config.chunk,
            enable_tts_interrupt=self.config.enable_tts_interrupt,
        )
        self.asr.start()
        self.asr.register_message_callback(self._handle_asr_message)

        # Initialize sleep ticker provider
        self.global_sleep_ticker_provider = SleepTickerProvider()

        # Initialize conversation provider
        api_key = self.config.api_key
        self.conversation_provider = TeleopsConversationProvider(api_key=api_key)

        # Initialize Zenoh session
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

    def _handle_asr_message(self, raw_message: str) -> None:
        """
        Process incoming ASR messages from the Whisper provider.

        Parameters
        ----------
        raw_message : str
            Raw JSON message from the Whisper ASR provider.
        """
        try:
            json_message: Dict = json.loads(raw_message)
            if "asr_reply" in json_message:
                asr_reply = json_message["asr_reply"]
                if len(asr_reply.split()) > 1:
                    self.message_buffer.put(asr_reply)
                    logging.info("Detected Whisper ASR message: %s", asr_reply)
        except json.JSONDecodeError:
            pass

    async def _poll(self) -> Optional[str]:
        """
        Poll for new messages in the buffer.

        Returns
        -------
        Optional[str]
            Message from the buffer if available, None otherwise.
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
            Raw input to be processed.

        Returns
        -------
        Optional[Message]
            Processed message or None if input is None.
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
            Raw input to be processed.
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
            Formatted string of buffer contents or None if buffer is empty.
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

    def stop(self) -> None:
        """Stop the Whisper ASR input."""
        if self.asr:
            self.asr.stop()

        if self.session:
            self.session.close()
            logging.info("Zenoh ASR session closed")
