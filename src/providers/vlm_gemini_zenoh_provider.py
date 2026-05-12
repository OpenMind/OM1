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
        """
        Initialize the VLM Provider.

        Parameters
        ----------
        base_url : str
            The base URL for the OM Gemini proxy.
        api_key : str
            The API key.
        topic : str
            The Zenoh topic to subscribe to for incoming frames.
        decode_format : str
            A hint for the incoming frame format; currently unused.
        model : str
            Gemini model id (e.g. "gemini-2.5-flash", "gemini-3.1-pro-preview").
        max_tokens : int
            The token budget for each VLM call.
        prompt : str
            The prompt to send with each frame to the Gemini model.
        """
        self.running: bool = False

        self.api_client: AsyncOpenAI = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model: str = model
        self.max_tokens: int = max_tokens
        self.prompt: str = prompt
        self.message_callback: Optional[Callable] = None
        self._inflight: int = 0
        self._max_inflight: int = 2  # back-pressure: at most N concurrent VLM calls

        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._loop_thread: threading.Thread = threading.Thread(
            target=self._run_loop, daemon=True, name="VLMGeminiZenohLoop"
        )
        self._loop_thread.start()

        self.video_stream: VideoZenohStream = VideoZenohStream(
            topic=topic,
            decode_format=decode_format,
            frame_callback=self._dispatch_frame,
        )

    def _run_loop(self):
        """
        Run the internal asyncio event loop in a dedicated thread.
        """
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _dispatch_frame(self, frame: str):
        """
        Dispatch a new frame from the Zenoh topic to the asyncio processing method.

        Parameters
        ----------
        frame : str
            The incoming frame data as a string, expected to be a JSON-encoded object containing a base64-encoded image.
        """
        if self._inflight >= self._max_inflight:
            return

        self._inflight += 1
        asyncio.run_coroutine_threadsafe(self._process_frame(frame), self._loop)

    async def _process_frame(self, frame: str):
        """
        Process a video frame via the OM Gemini proxy.

        Parameters
        ----------
        frame : str
            JSON string emitted by `om1_vlm.VideoZenohStream`, shape:
            `{"timestamp": <float>, "frame": "<base64-jpeg>"}`.
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
        """
        Register a callback for processing Gemini results.

        Parameters
        ----------
        message_callback : Optional[Callable]
            The callback function to process Gemini results.
        """
        self.message_callback = message_callback

    def deregister_message_callback(self):
        """
        Deregister the message callback.
        """
        self.message_callback = None

    def start(self):
        """
        Start the Gemini provider.

        Initializes and starts the video stream and processing thread
        if not already running.
        """
        if self.running:
            logging.warning("Gemini VLM Zenoh provider is already running")
            return

        self.running = True
        self.video_stream.start()

        logging.info("Gemini VLM Zenoh provider started")

    def stop(self):
        """
        Stop the Gemini provider.

        Stops the video stream and processing thread.
        """
        self.running = False
        self.video_stream.stop()
        self._loop.call_soon_threadsafe(self._loop.stop)
