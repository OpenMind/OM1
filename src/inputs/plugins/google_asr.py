import asyncio
import json
import logging
from queue import Empty, Queue
from typing import Optional
from inputs.base import SensorConfig
from providers.asr_provider import ASRProvider

class GoogleASRSensorConfig(SensorConfig):
    api_key: Optional[str] = None
    language: str = "english"
    rate: int = 48000
    chunk: int = 12144

class GoogleASRInput:
    def __init__(self, config: GoogleASRSensorConfig):
        self.config = config
        self.message_buffer = Queue()
        
        # Eğer dil 'auto' ise çok dilli modda başlat
        is_multi = (self.config.language.lower() == "auto")
        
        self.asr = ASRProvider(
            api_key=self.config.api_key,
            language=self.config.language,
            is_multi_lang=is_multi
        )
        self.asr.start()
        self.asr.register_message_callback(self._handle_asr_message)

    def _handle_asr_message(self, raw_message: str):
        try:
            data = json.loads(raw_message)
            if "asr_reply" in data:
                self.message_buffer.put(data["asr_reply"])
        except: pass

    async def _poll(self) -> Optional[str]:
        await asyncio.sleep(0.1)
        try: return self.message_buffer.get_nowait()
        except Empty: return None

    def stop(self):
        self.asr.stop()
