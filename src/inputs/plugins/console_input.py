"""
Console Input Plugin for OpenMind.

This plugin enables interaction with the robot via the standard console (stdin/stdout).
It is particularly useful for headless environments or cloud deployments where
physical audio/visual hardware is not available.

It also supports simulated vision by reading a local image file when specific
keywords are detected in the text input.
"""

import asyncio
import logging
import threading
import time
import base64
import os
import sys
import io
from queue import Empty, Queue
from typing import List, Optional

# Optional import for Pillow
try:
    from PIL import Image
except ImportError:
    Image = None

from pydantic import Field
from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider

# Initialize logger with standard naming
logger = logging.getLogger(__name__)

class ConsoleInputConfig(SensorConfig):
    """
    Configuration for the ConsoleInput plugin.
    """
    input_name: str = Field(default="User", description="Name of the input source")
    prompt: str = Field(default="You", description="Prompt string displayed in terminal") 
    image_path: str = Field(default="view.jpg", description="Path to the local image file for simulated vision")

class ConsoleInput(FuserInput[ConsoleInputConfig, Optional[str]]):
    """
    A plugin that reads text input from the console (stdin) and simulates visual input.
    """

    def __init__(self, config: ConsoleInputConfig):
        super().__init__(config)
        self.messages: List[Message] = []
        self.io_provider = IOProvider()
        self.descriptor_for_LLM = self.config.input_name
        self.input_queue: Queue[str] = Queue()
        self.running = True
        
        # Start background thread
        self.input_thread = threading.Thread(target=self._read_stdin_loop, daemon=True)
        self.input_thread.start()
        
        logger.info(f"ConsoleInput loaded. Vision source: {self.config.image_path}")
        if Image is None:
            logger.warning("Pillow not installed. Image input may fail.")

    def _read_stdin_loop(self):
        """
        Blocking read loop. 
        Uses standard print for prompt (allowed exception) and logging for captured input.
        """
        while self.running:
            try:
                # Standard prompt without color codes
                print(f"\n>>> {self.config.prompt}:", end=' ', flush=True)
                
                text = input()
                if text.strip():
                    # Log with a specific tag so external tools can parse it
                    logger.info(f"[USER SAID]: {text.strip()}")
                    self.input_queue.put(text.strip())
            except EOFError:
                logger.info("ConsoleInput: EOF detected.")
                break
            except Exception as e:
                logger.error(f"ConsoleInput Read Error: {e}")
                time.sleep(0.1)

    def _encode_image(self) -> Optional[str]:
        """Read and compress local image."""
        if not os.path.exists(self.config.image_path):
            logger.error(f"ConsoleInput: Image {self.config.image_path} not found.")
            return None
            
        if Image is None: return None

        try:
            with Image.open(self.config.image_path) as img:
                if img.mode != 'RGB': img = img.convert('RGB')
                img.thumbnail((512, 512))
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=60)
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            logger.error(f"ConsoleInput: Image error: {e}")
            return None

    async def _poll(self) -> Optional[str]:
        await asyncio.sleep(0.1)
        try:
            text = self.input_queue.get_nowait()
            if not text: return None

            keywords = ["see", "look", "view", "what is this", "picture", "image"]
            if any(k in text.lower() for k in keywords):
                logger.info(f"ConsoleInput: Vision keyword detected in '{text}'")
                img_data = self._encode_image()
                if img_data:
                    return f"{text}\n\n[System: User showed a compressed image]\ndata:image/jpeg;base64,{img_data}"
            
            return text
        except Empty:
            return None

    async def raw_to_text(self, raw_input: Optional[str]):
        if raw_input: 
            self.messages.append(Message(timestamp=time.time(), message=raw_input))

    def formatted_latest_buffer(self) -> Optional[str]:
        if len(self.messages) == 0: return None
        content = self.messages[-1].message
        self.messages = []
        
        disp = content[:100] + "... [Image Data]" if "base64" in content else content
        self.io_provider.add_input(self.descriptor_for_LLM, disp, time.time())

        return f"INPUT: {self.descriptor_for_LLM}\n// START\n{content}\n// END"
