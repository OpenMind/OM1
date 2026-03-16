import asyncio
import logging
import time
from collections import deque
from typing import Deque, Optional

from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider


class ConversationHistoryConfig(SensorConfig):
    """
    Configuration for Conversation History Input.

    Parameters
    ----------
    max_rounds : int
        Maximum number of voice inputs to keep in history.
    """

    max_rounds: int = Field(
        default=3,
        description="Maximum number of voice inputs to keep in history",
    )


class ConversationHistoryInput(FuserInput[ConversationHistoryConfig, Optional[str]]):
    """
    Async input that polls IOProvider for voice inputs and maintains
    a sliding window of conversation history for the LLM prompt.
    """

    def __init__(self, config: ConversationHistoryConfig):
        super().__init__(config)

        self.io_provider = IOProvider()
        self.messages: Deque[Message] = deque(maxlen=config.max_rounds)
        self._last_recorded_tick: int = -1
        self.descriptor_for_LLM = "Conversation History"

        # Guard flag: when True, this instance ignores incoming voice inputs
        self._stopped = False

    async def _poll(self) -> Optional[str]:
        """
        Check IOProvider for new voice input this tick.

        Returns
        -------
        Optional[str]
            The voice input text if new, None otherwise.
        """
        await asyncio.sleep(0.5)

        if self._stopped:
            return

        current_tick = self.io_provider.tick_counter
        if current_tick <= self._last_recorded_tick:
            return None

        voice_input = self.io_provider.get_input("Voice")
        if voice_input and voice_input.input and voice_input.tick == current_tick:
            text = voice_input.input.strip()
            if text:
                self._last_recorded_tick = current_tick
                return text

        return None

    async def _raw_to_text(self, raw_input: Optional[str]) -> Optional[Message]:
        """
        Process raw input to generate a timestamped message.

        Parameters
        ----------
        raw_input : Optional[str]
            Raw input string to be processed.

        Returns
        -------
        Optional[Message]
            A timestamped message containing the processed input.
        """
        if raw_input is None:
            return None
        return Message(timestamp=time.time(), message=raw_input)

    async def raw_to_text(self, raw_input: Optional[str]):
        """
        Convert raw input to text and update message buffer.

        Parameters
        ----------
        raw_input : Optional[str]
            Raw input to be processed, or None if no input is available.
        """
        if raw_input is None:
            return

        message = await self._raw_to_text(raw_input)
        if message is not None:
            self.messages.append(message)

    def formatted_latest_buffer(self) -> Optional[str]:
        """
        Return all recorded voice inputs as a conversation history block.

        Returns
        -------
        Optional[str]
            A formatted string of the conversation history for LLM input, or None if no history exists.
        """
        if len(self.messages) == 0:
            return None

        lines = [f"User: {msg.message}" for msg in self.messages]
        result = f'{self.descriptor_for_LLM}: "{"; ".join(lines)}"'

        return result

    def stop(self):
        """
        Clear message history and reset state when stopping the input.
        """
        logging.info("Stopping ConversationHistoryInput")

        self._stopped = True

        self.messages.clear()
