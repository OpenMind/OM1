import asyncio
import logging
import time
from typing import List, Optional

from inputs.base import Message
from inputs.base.loop import FuserInput
from inputs.plugins.google_asr import GoogleASRInput, GoogleASRSensorConfig
from providers.io_provider import IOProvider
from tests.integration.mock_inputs.data_providers.mock_text_provider import (
    get_next_text,
)


class _MockConversationProvider:
    """Lightweight stand-in for TeleopsConversationProvider.

    The real provider needs an API key and network access.
    This mock silently accepts store_user_message calls so that
    MockGoogleASR.formatted_latest_buffer matches upstream behavior.
    """

    def store_user_message(self, message: str) -> None:
        logging.debug(
            f"MockConversationProvider: stored message ({len(message)} chars)"
        )


class MockGoogleASR(GoogleASRInput):
    """
    Mock implementation of GoogleASRInput that uses the central text provider.

    This class bypasses the real ASR provider and audio hardware to inject
    mock text data for integration testing.
    """

    def __init__(self, config: GoogleASRSensorConfig = GoogleASRSensorConfig()):
        """
        Initialize with mock text provider, bypassing real ASR hardware.

        Parameters
        ----------
        config : GoogleASRSensorConfig, optional
            Configuration for the sensor
        """
        # Skip GoogleASRInput.__init__ to avoid ASRProvider/Zenoh/microphone setup
        # Initialize FuserInput base class directly
        FuserInput.__init__(self, config)

        self.messages: List[str] = []
        self.descriptor_for_LLM = "MOCK Voice INPUT (Integration Test)"
        self.io_provider = IOProvider()
        self.message_buffer: asyncio.Queue[str] = asyncio.Queue()

        self.conversation_provider = _MockConversationProvider()
        self.running = True
        self.texts_processed = False

        logging.info("MockGoogleASR initialized - using mock text provider")

    async def _poll(self) -> Optional[str]:
        """
        Poll for mock text data from the text provider.

        Returns
        -------
        Optional[str]
            Next text from the mock provider, or None if no more texts
        """
        text = get_next_text()
        if text is not None:
            logging.info(f"MockGoogleASR: Providing text: {text}")
            return text

        if not self.texts_processed:
            logging.info("MockGoogleASR: No more texts to process")
            self.texts_processed = True

        return None

    async def _raw_to_text(self, raw_input: Optional[str]) -> Optional[Message]:
        """Convert raw text input to Message format."""
        if raw_input is None:
            return None
        return Message(timestamp=time.time(), message=raw_input)

    async def raw_to_text(self, raw_input: Optional[str]):
        """Convert raw input to text and update message buffer."""
        pending_message = await self._raw_to_text(raw_input)
        if pending_message is not None:
            if len(self.messages) == 0:
                self.messages.append(pending_message.message)
            else:
                self.messages[-1] = f"{self.messages[-1]} {pending_message.message}"

    def formatted_latest_buffer(self) -> Optional[str]:
        """Format and clear the latest buffer contents."""
        if len(self.messages) == 0:
            return None

        result = f"\nINPUT: {self.descriptor_for_LLM}\n// START\n{self.messages[-1]}\n// END\n"

        self.io_provider.add_input(
            self.descriptor_for_LLM, self.messages[-1], time.time()
        )
        self.io_provider.add_mode_transition_input(self.messages[-1])
        self.conversation_provider.store_user_message(self.messages[-1])

        self.messages = []
        return result

    def stop(self):
        """Stop the mock ASR input."""
        self.running = False
        logging.info("MockGoogleASR: Stopped")

    def cleanup(self):
        """Clean up resources."""
        self.running = False
        logging.info("MockGoogleASR: Cleanup completed")

    def __del__(self):
        """Clean up resources when the object is destroyed."""
        self.cleanup()
