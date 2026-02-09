import asyncio
import json
import logging
import time
from queue import Empty, Queue
from typing import List, Optional, Union
from uuid import uuid4

import pyaudio
from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.asr_provider import ASRProvider
from providers.io_provider import IOProvider
from providers.sleep_ticker_provider import SleepTickerProvider
from providers.teleops_conversation_provider import TeleopsConversationProvider
from zenoh_msgs import ASRText, open_zenoh_session, prepare_header

LANGUAGE_CODE_MAP = {
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


class GoogleASRSensorConfig(SensorConfig):
    """
    Configuration for Google ASR Sensor.
    """

    api_key: Optional[str] = Field(default=None, description="API Key")
    rate: int = Field(default=48000, description="Sampling rate")
    chunk: int = Field(default=12144, description="Chunk size")
    base_url: Optional[str] = Field(
        default=None, description="Base URL for the ASR service"
    )
    microphone_device_id: Optional[Union[int, str]] = Field(
        default=None, description="Microphone Device ID or name (e.g., 'hw:1,0')"
    )
    microphone_name: Optional[str] = Field(default=None, description="Microphone Name")
    language: str = Field(
        default="english", description="Language for speech recognition"
    )
    remote_input: bool = Field(default=False, description="Whether to use remote input")
    enable_tts_interrupt: bool = Field(
        default=False,
        description="Enable TTS interrupt (does not mute mic during TTS playback)",
    )


class GoogleASRInput(FuserInput[GoogleASRSensorConfig, Optional[str]]):
    """
    Google Automatic Speech Recognition (ASR) input handler.
    """

    def __init__(self, config: GoogleASRSensorConfig):
        super().__init__(config)

        self.messages: List[str] = []
        self.descriptor_for_LLM = "Voice"
        self.io_provider = IOProvider()
        self.message_buffer: Queue[str] = Queue()

        resolved_device_id = self._resolve_audio_device(config.microphone_device_id)
        resolved_microphone_name = config.microphone_name

        api_key = config.api_key
        rate = config.rate
        chunk = config.chunk
        base_url = (
            config.base_url
            or f"wss://api.openmind.org/api/core/google/asr?api_key={api_key}"
        )

        language = config.language.strip().lower()
        if language not in LANGUAGE_CODE_MAP:
            logging.error(
                f"Language {language} not supported. Current supported languages are : {list(LANGUAGE_CODE_MAP.keys())}. Defaulting to English"
            )
            language = "english"

        language_code = LANGUAGE_CODE_MAP.get(language, "en-US")
        logging.info(f"Using language code {language_code} for Google ASR")

        remote_input = config.remote_input
        enable_tts_interrupt = config.enable_tts_interrupt

        # Pass only int or None to ASRProvider, since it only accepts int | None
        device_id_for_asr = (
            resolved_device_id if isinstance(resolved_device_id, int) else None
        )

        self.asr: ASRProvider = ASRProvider(
            rate=rate,
            chunk=chunk,
            ws_url=base_url,
            device_id=device_id_for_asr,  # Only pass int or None
            microphone_name=resolved_microphone_name,
            language_code=language_code,
            remote_input=remote_input,
            enable_tts_interrupt=enable_tts_interrupt,
        )
        self.asr.start()
        self.asr.register_message_callback(self._handle_asr_message)

        self.global_sleep_ticker_provider = SleepTickerProvider()
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

    def _resolve_audio_device(
        self, requested_device: Optional[Union[int, str]]
    ) -> Optional[Union[int, str]]:
        if requested_device is not None:
            logging.info(
                f"Using explicitly configured microphone device: {requested_device}"
            )
            return requested_device

        try:
            p = pyaudio.PyAudio()
            default_info = p.get_default_input_device_info()
            p.terminate()

            # Convert to int as expected by ASRProvider
            device_id = int(default_info["index"])
            device_name = default_info["name"]

            logging.info(
                f"Auto-detected default microphone: {device_name} (ID: {device_id})"
            )
            return device_id
        except Exception as e:
            logging.warning(
                f"Could not auto-detect default microphone: {e}. Using None (system default)."
            )
            return None

    def _handle_asr_message(self, raw_message: str):
        try:
            json_message = json.loads(raw_message)
            if "asr_reply" in json_message:
                asr_reply = json_message["asr_reply"]
                if len(asr_reply.split()) > 1:
                    self.message_buffer.put(asr_reply)
                    logging.info("Detected ASR message: %s", asr_reply)
        except json.JSONDecodeError:
            pass

    async def _poll(self) -> Optional[str]:
        await asyncio.sleep(0.1)
        try:
            message = self.message_buffer.get_nowait()
            return message
        except Empty:
            return None

    async def _raw_to_text(self, raw_input: Optional[str]) -> Optional[Message]:
        if raw_input is None:
            return None
        return Message(timestamp=time.time(), message=raw_input)

    async def raw_to_text(self, raw_input: Optional[str]):
        """Convert raw input to processed text and manage buffer."""
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
        """Format and clear the latest buffer contents."""
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
        self.io_provider.add_mode_transition_input(self.messages[-1])
        self.conversation_provider.store_user_message(self.messages[-1])

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

        self.messages = []
        return result

    def stop(self):
        """Stop the ASR input."""
        if self.asr:
            self.asr.stop()

        if self.session:
            self.session.close()
            logging.info("Zenoh ASR session closed")
