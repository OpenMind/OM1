"""
ConsoleInput Plugin
===================
A plugin designed for headless or cloud environments where physical microphones
and cameras are unavailable.

Features:
- **Terminal Chat**: Allows text interaction via stdin/stdout.
- **Simulated Vision**: Reads a local image file and sends it to the VLM when keywords
  (e.g., "see", "look") are detected.
- **Token Optimization**: Auto-compresses images to avoid LLM context overflow.

Dependencies:
- Pillow (Optional, but recommended for image compression)
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

class ConsoleInputConfig(SensorConfig):
    input_name: str = Field(default="User", description="Name of the input source")
    prompt: str = Field(default="You", description="Prompt string") 
    image_path: str = Field(default="view.jpg", description="Path to local image for vision")

class ConsoleInput(FuserInput[ConsoleInputConfig, Optional[str]]):
    def __init__(self, config: ConsoleInputConfig):
        super().__init__(config)
        self.messages: List[Message] = []
        self.io_provider = IOProvider()
        self.descriptor_for_LLM = self.config.input_name
        self.input_queue: Queue[str] = Queue()
        self.running = True
        
        # Start keyboard listener
        self.input_thread = threading.Thread(target=self._read_stdin_loop, daemon=True)
        self.input_thread.start()
        
        print(f"\n[SYSTEM] ConsoleInput loaded. Vision source: '{self.config.image_path}'\n")

    def _read_stdin_loop(self):
        """Blocking read from stdin."""
        while self.running:
            try:
                print(f"\n>>> {self.config.prompt}:", end=' ', flush=True)
                text = input()
                if text.strip():
                    print(f"   [Captured]: {text}") 
                    self.input_queue.put(text.strip())
            except (EOFError, KeyboardInterrupt):
                break
            except Exception:
                time.sleep(0.1)

    def _encode_image(self) -> Optional[str]:
        """Read and compress image."""
        if not os.path.exists(self.config.image_path):
            logging.error(f"Image {self.config.image_path} not found.")
            return None
        
        if Image is None:
            logging.warning("Pillow not installed. Skipping image to prevent errors.")
            return None

        try:
            with Image.open(self.config.image_path) as img:
                if img.mode != 'RGB': img = img.convert('RGB')
                img.thumbnail((512, 512)) # Max 512px
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=60)
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            logging.error(f"Image error: {e}")
            return None

    async def _poll(self) -> Optional[str]:
        await asyncio.sleep(0.1)
        try:
            text = self.input_queue.get_nowait()
            if not text: return None

            keywords = ["see", "look", "view", "what is this", "picture", "image"]
            if any(k in text.lower() for k in keywords):
                logging.info(f"Vision keyword detected in '{text}'")
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
