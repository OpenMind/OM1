import asyncio
import json
import logging
import time
from queue import Empty, Queue
from typing import Dict, List, Optional

from inputs.base import SensorConfig
from inputs.base.loop import FuserInput
from providers.asr_rtsp_provider import ASRRTSPProvider
from providers.io_provider import IOProvider
from providers.sleep_ticker_provider import SleepTickerProvider
from providers.teleops_conversation_provider import TeleopsConversationProvider

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


class GoogleASRRTSPInput(FuserInput[str]):
    """
    Automatic Speech Recognition (ASR) input handler.

    This class manages the RTSP input stream from an ASR service, buffering messages
    and providing text conversion capabilities.
    """

    def __init__(self, config: SensorConfig = SensorConfig()):
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
        api_key = getattr(self.config, "api_key", None)
        rtsp_url = getattr(self.config, "rtsp_url", "rtsp://localhost:8554/audio")
        rate = getattr(self.config, "rate", 16000)
        base_url = getattr(
            self.config,
            "base_url",
            f"wss://api.openmind.org/api/core/google/asr?api_key={api_key}",
        )

        language = getattr(self.config, "language", "english").strip().lower()

        if language not in LANGUAGE_CODE_MAP:
            logging.error(
                f"Language {language} not supported. Current supported languages are : {list(LANGUAGE_CODE_MAP.keys())}. Defaulting to English"
            )
            language = "english"

        language_code = LANGUAGE_CODE_MAP.get(language, "en-US")
        logging.info(f"Using language code {language_code} for Google ASR")

        self.asr: ASRRTSPProvider = ASRRTSPProvider(
            rtsp_url=rtsp_url,
            rate=rate,
            ws_url=base_url,
            language_code=language_code,
        )
        self.asr.start()
        self.asr.register_message_callback(self._handle_asr_message)

        # Initialize sleep ticker provider
        self.global_sleep_ticker_provider = SleepTickerProvider()

        # Initialize conversation provider
        self.conversation_provider = TeleopsConversationProvider(api_key=api_key)

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

        # Reset messages buffer
        self.messages = []
        return result
