import asyncio
import sys
import threading
import time
import logging
from dataclasses import dataclass
from typing import List, Optional
from queue import Queue, Empty

from inputs.base import SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider


@dataclass
class Message:
    timestamp: float
    message: str


class ConsoleInput(FuserInput[str]):
    """
    Captures text input from the console using a thread to allow blocking input() with prompt.
    """

    def __init__(self, config: SensorConfig):
        super().__init__(config)
        self.config = config
        self.messages: List[Message] = []
        self.descriptor_for_LLM = getattr(self.config, "input_name", "Console Input")
        self.io_provider = IOProvider()
        
        self.input_queue: Queue[str] = Queue()
        self.running = True
        
        # Start input thread
        self.thread = threading.Thread(target=self._input_loop, daemon=True)
        self.thread.start()

        logging.info("ConsoleInput: Initialized")
        print("\n" + "="*50)
        print("⌨️  CONSOLE INPUT ACTIVE")
        print("Type your command below and press ENTER")
        print("="*50 + "\n")

    def _input_loop(self):
        """Blocking input loop in separate thread"""
        time.sleep(4) # Wait for initial logs to settle
        while self.running:
            try:
                # Use standard input with prompt
                text = input("\n👤 YOU > ")
                if text.strip():
                    self.input_queue.put(text)
            except EOFError:
                break
            except Exception as e:
                logging.error(f"Input thread error: {e}")
                break

    async def _poll(self) -> Optional[str]:
        """
        Polls queue for input asynchronously.
        """
        await asyncio.sleep(0.1)
        try:
            return self.input_queue.get_nowait()
        except Empty:
            return None

    async def _raw_to_text(self, raw_input: str) -> Message:
        """
        Converts raw text input to a Message object.
        """
        logging.info(f"ConsoleInput received: {raw_input}")
        return Message(timestamp=time.time(), message=raw_input)

    async def raw_to_text(self, raw_input: Optional[str]):
        """
        Process raw input and update buffer.
        """
        if raw_input is None:
            return

        print(f"\n✅ Processing...\n")
        pending_message = await self._raw_to_text(raw_input)
        self.messages.append(pending_message)

    def formatted_latest_buffer(self) -> Optional[str]:
        """
        Format and clear buffer for LLM consumption.
        """
        if len(self.messages) == 0:
            return None

        latest_msg = self.messages[-1]
        
        result = f"""
INPUT: {self.descriptor_for_LLM}
// START
{latest_msg.message}
// END
"""
        self.io_provider.add_input(
            self.descriptor_for_LLM, latest_msg.message, latest_msg.timestamp
        )
        self.messages = []
        return result
