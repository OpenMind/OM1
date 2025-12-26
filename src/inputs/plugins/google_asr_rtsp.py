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
from providers.asr_rtsp_provider import ASRRTSPProvider
from providers.io_provider import IOProvider
from providers.sleep_ticker_provider import SleepTickerProvider
from providers.teleops_conversation_provider import TeleopsConversationProvider
from zenoh_msgs import ASRText, open_zenoh_session, prepare_header

LANGUAGE_CODE_MAP: dict = {
    "english": "en-US",
    "chinese": "cmn-Hans-CN",
    "german": "de-DE",
    "french": "fr-FR",
    "japanese": "ja-JP",
    "korean": "ko-KR",
    "spanish": "es-ES",
    "italian": "it-IT",
    "portuguese": "pt-BR",
    "russian": "ru-RU",
    "arabic": "ar-SA",
}


class GoogleASRRTSPSensorConfig(SensorConfig):
    """
    Configuration for Google ASR RTSP Sensor.

    Parameters
    ----------
    api_key : Optional[str]
        API Key.
    rtsp_url : str
        RTSP URL for the audio stream.
    rate : int
        Audio sampling rate.
    base_url : Optional[str]
        Base URL for the ASR service.
    language : str
        Language for speech recognition.
    """

    api_key: Optional[str] = Field(default=None, description="API Key")
    rtsp_url: str = Field(
        default="rtsp://localhost:8554/audio",
        description="RTSP URL for the audio stream",
    )
    rate: int = Field(default=16000, description="Audio sampling rate")
    base_url: Optional[str] = Field(
        default=None, description="Base URL for the ASR service"
    )
    language: str = Field(
        default="english", description="Language for speech recognition"
    )
    enable_tts_interrupt: bool = Field(
        default=False,
        description="Enable TTS interrupt (does not mute mic during TTS playback)",
    )


class GoogleASRRTSPInput(FuserInput[GoogleASRRTSPSensorConfig, Optional[str]]):
    """
    Automatic Speech Recognition (ASR) input handler.

    This class manages the RTSP input stream from an ASR service, buffering messages
    and providing text conversion capabilities.
    """

    def __init__(self, config: GoogleASRRTSPSensorConfig):
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
        rtsp_url = self.config.rtsp_url
        rate = self.config.rate
        base_url = (
            self.config.base_url
            or f"wss://api.openmind.org/api/core/google/asr?api_key={api_key}"
        )

        language = self.config.language.strip().lower()

        if language not in LANGUAGE_CODE_MAP:
            logging.error(
                f"Language {language} not supported. Current supported languages are : {list(LANGUAGE_CODE_MAP.keys())}. Defaulting to English"
            )
            language = "english"

        language_code = LANGUAGE_CODE_MAP.get(language, "en-US")
        logging.info(f"Using language code {language_code} for Google ASR")

        enable_tts_interrupt = self.config.enable_tts_interrupt

        self.asr: ASRRTSPProvider = ASRRTSPProvider(
            rtsp_url=rtsp_url,
            rate=rate,
            ws_url=base_url,
            language_code=language_code,
            enable_tts_interrupt=enable_tts_interrupt,
        )
        self.asr.start()
        self.asr.register_message_callback(self._handle_asr_message)

        # Initialize sleep ticker provider
        self.global_sleep_ticker_provider = SleepTickerProvider()

        # Initialize conversation provider
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
            Raw input string to be processed

        Returns
        -------
        Optional[Message]
            Timestamped message containing the processed input
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
        """
        Stop the ASR input.
        """
        if self.asr:
            self.asr.stop()

        if self.session:
            self.session.close()
            logging.info("Zenoh ASR session closed")
