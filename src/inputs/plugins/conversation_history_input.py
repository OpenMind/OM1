"""
Conversation history input plugin.

Follows the same pattern as FacePresenceInput:
- Subscribes to ConversationHistoryProvider callbacks
- Enqueues received voice input lines
- Polls the queue in _poll()
- Converts to Messages in raw_to_text()
- Formats for LLM in formatted_latest_buffer()

Place this file at: src/inputs/plugins/conversation_history_input.py

Add to greeting_local.json5 agent_inputs:
    { type: "ConversationHistoryInput", config: { max_rounds: 3 } }
"""

import asyncio
import logging
import time
from collections import deque
from queue import Empty, Queue
from typing import Deque, Optional

from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.conversation_history_provider import ConversationHistoryProvider
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
    Async input that adapts ConversationHistoryProvider to the fuser/LLM pipeline.

    Tasks
    -----
    - Subscribe to the provider's callbacks and enqueue received voice lines.
    - Poll the queue periodically (non-blocking) in _poll().
    - Convert raw text into Message objects in raw_to_text().
    - Keep a bounded in-memory history (self.messages, deque with maxlen).
    - Produce a prompt-ready block via formatted_latest_buffer().
    """

    def __init__(self, config: ConversationHistoryConfig):
        super().__init__(config)

        self.io_provider = IOProvider()

        self.messages: Deque[Message] = deque(maxlen=config.max_rounds)

        self.message_buffer: Queue[str] = Queue(maxsize=64)

        self.provider: ConversationHistoryProvider = ConversationHistoryProvider(
            max_rounds=config.max_rounds,
        )
        self._is_registered: bool = True

        self.provider.start()
        self.provider.register_message_callback(self._handle_voice_message)

        self.descriptor_for_LLM = "Conversation History"

    def _handle_voice_message(self, text_line: str) -> None:
        """
        Provider callback: push a new line into the bounded queue.

        Parameters
        ----------
        text_line : str
            A user voice input string.
        """
        try:
            self.message_buffer.put_nowait(text_line)
        except Exception:
            logging.debug("ConversationHistory queue full; dropping oldest to enqueue")
            try:
                _ = self.message_buffer.get_nowait()
            except Empty:
                pass
            try:
                self.message_buffer.put_nowait(text_line)
            except Exception:
                logging.warning("ConversationHistory queue still full; dropping latest")
                pass

    async def _poll(self) -> Optional[str]:
        """
        Poll for new messages from the conversation history provider.

        Returns
        -------
        Optional[str]
            The next message from the buffer if available, None otherwise.
        """
        await asyncio.sleep(0.5)
        try:
            return self.message_buffer.get_nowait()
        except Empty:
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

        Unlike FacePresence which only returns the latest, this returns
        all messages to give the LLM full conversation context.

        Returns
        -------
        str or None
            Formatted conversation history for LLM, or None if empty.
        """
        if len(self.messages) == 0:
            return None

        lines = [f"User: {msg.message}" for msg in self.messages]
        result = f"""
INPUT: {self.descriptor_for_LLM}
// START
{chr(10).join(lines)}
// END
"""

        self.io_provider.add_input(
            self.__class__.__name__,
            "\n".join(lines),
            self.messages[-1].timestamp,
        )

        # Don't clear — keep history for sliding window
        return result
