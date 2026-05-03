import asyncio
import json
import logging
import threading
import time
from typing import Callable, Optional

from om1_vlm import VideoZenohStream
from openai import AsyncOpenAI

from .singleton import singleton


@singleton
class VLMGeminiZenohProvider:
    """
    Gemini VLM provider that ingests frames from a Zenoh topic instead of
    a local camera. Mirrors VLMGeminiProvider's HTTP behavior; the only
    difference is the frame source.

    Use this for cloud_sim, where the camera lives on the GPU EC2 / robot
    and is bridged to Zenoh by zenoh-bridge-ros2dds.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        topic: str = "rgb_image",
        decode_format: str = "RAW",
        model: str = "gemini-2.5-flash",
        max_tokens: int = 1024,
        prompt: str = (
            "In one concise sentence, describe what you see in this image. "
            "Just the description — no explanation of your reasoning."
        ),
    ):
        self.running: bool = False
        # AsyncOpenAI for auth/connection pooling; we bypass its response
        # parser via `with_raw_response.create()` because openai==1.60.1
        # rejects Gemini's `extra_content.google.thought_signature` field.
        self.api_client: AsyncOpenAI = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model: str = model
        self.max_tokens: int = max_tokens
        self.prompt: str = prompt
        self.message_callback: Optional[Callable] = None
        self._inflight: int = 0
        self._max_inflight: int = 2  # back-pressure: at most N concurrent VLM calls

        # VideoZenohStream's own self.loop is never started — its zenoh
        # callback would enqueue our coroutine onto a dead loop. So we run
        # our own loop in a dedicated thread and pass a synchronous
        # frame_callback that schedules onto it.
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._loop_thread: threading.Thread = threading.Thread(
            target=self._run_loop, daemon=True, name="VLMGeminiZenohLoop"
        )
        self._loop_thread.start()

        # VideoZenohStream subscribes via open_zenoh_session(); when
        # OPENMIND_CLOUD_URL is set it uses the broker shim, so frames
        # come through the same broker the rest of cloud_sim uses.
        self.video_stream: VideoZenohStream = VideoZenohStream(
            topic=topic,
            decode_format=decode_format,
            frame_callback=self._dispatch_frame,
        )

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _dispatch_frame(self, frame: str):
        """Sync entrypoint called from the zenoh callback thread. Drops
        the frame if too many requests are already in flight (Gemini latency
        is ~1-3s; at 12fps frames pile up otherwise).
        """
        if self._inflight >= self._max_inflight:
            return
        self._inflight += 1
        asyncio.run_coroutine_threadsafe(self._process_frame(frame), self._loop)

    async def _process_frame(self, frame: str):
        """Frame callback. ``frame`` is `{"timestamp":..., "frame":"<b64>"}` —
        unwrap to bare base64 for the data: URL Gemini expects.
        """
        try:
            envelope = json.loads(frame)
            b64 = envelope["frame"]
        except (json.JSONDecodeError, KeyError, TypeError):
            b64 = frame

        processing_start = time.perf_counter()
        try:
            raw = await self.api_client.chat.completions.with_raw_response.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
                            },
                        ],
                    }
                ],
                max_tokens=self.max_tokens,
            )
            data = json.loads(raw.text)
            content = data["choices"][0]["message"]["content"]
            logging.debug(
                "Gemini VLM Zenoh latency=%.3fs content=%r",
                time.perf_counter() - processing_start,
                content[:200] if content else None,
            )
            if self.message_callback and content is not None:
                self.message_callback(content)
        except Exception as e:
            logging.error("Error processing frame: %s", e)
        finally:
            self._inflight = max(0, self._inflight - 1)

    def register_message_callback(self, message_callback: Optional[Callable]):
        """Register a callback invoked with each VLM response string."""
        self.message_callback = message_callback

    def start(self):
        """Start the underlying video stream (idempotent)."""
        if self.running:
            logging.warning("Gemini VLM Zenoh provider is already running")
            return
        self.running = True
        self.video_stream.start()
        logging.info("Gemini VLM Zenoh provider started")

    def stop(self):
        """Stop the video stream and shut down the internal asyncio loop."""
        self.running = False
        self.video_stream.stop()
        self._loop.call_soon_threadsafe(self._loop.stop)
