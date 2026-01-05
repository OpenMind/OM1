import asyncio
import logging
import time
from typing import Callable, Optional, Any

from om1_utils import ws
from om1_vlm import VideoStream
from openai import AsyncOpenAI

from .singleton import singleton


@singleton
class VLMOpenAIProvider:
    """
    VLM Provider that handles video streaming and OpenAI API communication.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        fps: int = 10,
        stream_url: Optional[str] = None,
        camera_index: int = 0,
        model: str = "gpt-4o-mini",
        prompt: str = "What is the most interesting aspect in this series of images?",
        max_tokens: int = 300,
    ):
        self.running: bool = False
        self.model = model
        self.prompt = prompt
        self.max_tokens = max_tokens

        self.api_client: AsyncOpenAI = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        self.stream_ws_client: Optional[ws.Client] = (
            ws.Client(url=stream_url) if stream_url else None
        )

        self._lock = asyncio.Lock()

        self.video_stream: VideoStream = VideoStream(
            frame_callback=self._process_frame,
            fps=fps,
            device_index=camera_index,  # type: ignore
        )

        self.message_callback: Optional[Callable[[Any], None]] = None

    async def _process_frame(self, frame: str):
        if not self.running:
            return

        async with self._lock:
            if not self.running:
                return

            processing_start = time.perf_counter()

            try:
                response = await self.api_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": self.prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{frame}",
                                        "detail": "low",
                                    },
                                },
                            ],
                        }
                    ],
                    max_tokens=self.max_tokens,
                )

                latency = time.perf_counter() - processing_start
                logging.debug("VLM processing latency: %.3f seconds", latency)

                if self.message_callback:
                    self.message_callback(response)

            except asyncio.CancelledError:
                logging.info("VLM frame processing cancelled")
                raise

            except Exception as e:
                logging.exception("Error while processing VLM frame: %s", e)

    def register_message_callback(
        self, message_callback: Optional[Callable[[Any], None]]
    ):
        self.message_callback = message_callback

    def start(self):
        if self.running:
            logging.warning("VLM provider is already running")
            return

        self.running = True
        self.video_stream.start()

        if self.stream_ws_client:
            self.stream_ws_client.start()
            self.video_stream.register_frame_callback(
                self.stream_ws_client.send_message
            )

        logging.info("OpenAI VLM provider started")

    def stop(self):
        if not self.running:
            return

        self.running = False
        self.video_stream.stop()

        if self.stream_ws_client:
            self.stream_ws_client.stop()

        logging.info("OpenAI VLM provider stopped")
