import json
import logging
import time
from typing import Callable, Optional

from om1_utils import ws
from om1_vlm import VideoStream
from openai import AsyncOpenAI

from .singleton import singleton


@singleton
class VLMGeminiProvider:
    """
    VLM Provider that handles video streaming and Gemini API communication.

    This class implements a singleton pattern to manage video input streaming and API
    communication for vlm services. It runs in a separate thread to handle
    continuous vlm processing.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        fps: int = 10,
        stream_url: Optional[str] = None,
        camera_index: int = 0,
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
        fps : int
            The frames per second for the video stream.
        stream_url : str, optional
            The URL for the teleops video stream.
        camera_index : int
            The camera index for the video stream device. Defaults to 0.
        model : str
            Gemini model id (e.g. "gemini-2.5-flash", "gemini-3.1-pro-preview").
        """
        self.running: bool = False
        self.api_client: AsyncOpenAI = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model: str = model
        self.max_tokens: int = max_tokens
        self.prompt: str = prompt
        self.stream_ws_client: Optional[ws.Client] = ws.Client(url=stream_url) if stream_url else None
        self.video_stream: VideoStream = VideoStream(
            frame_callback=self._process_frame, fps=fps, device_index=camera_index  # type: ignore
        )
        self.message_callback: Optional[Callable] = None

    async def _process_frame(self, frame: str):
        """
        Process a video frame via the OM Gemini proxy.

        Parameters
        ----------
        frame : str
            JSON string emitted by `om1_vlm.VideoStream`, shape:
            `{"timestamp": <float>, "frame": "<base64-jpeg>"}`.
        """
        try:
            envelope = json.loads(frame)
            base64_image = envelope["frame"]
        except (json.JSONDecodeError, KeyError, TypeError):
            base64_image = frame

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
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "low"},
                            },
                        ],
                    }
                ],
                max_tokens=self.max_tokens,
            )

            data = json.loads(raw.text)
            content = data["choices"][0]["message"]["content"]
            processing_latency = time.perf_counter() - processing_start

            logging.debug(f"Processing latency: {processing_latency:.3f} seconds")
            logging.debug(f"Gemini VLM content: {content[:200]!r}")

            if self.message_callback and content is not None:
                self.message_callback(content)

        except Exception as e:
            body = None
            resp = getattr(e, "response", None)
            if resp is not None:
                try:
                    body = resp.text
                except Exception:
                    pass
            logging.error("Error processing frame: %s | body=%s", e, body)

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
            logging.warning("Gemini VLM provider is already running")
            return

        self.running = True
        self.video_stream.start()

        if self.stream_ws_client:
            self.stream_ws_client.start()
            self.video_stream.register_frame_callback(self.stream_ws_client.send_message)

        logging.info("Gemini VLM provider started")

    def stop(self):
        """
        Stop the Gemini provider.

        Stops the video stream and processing thread.
        """
        self.running = False
        self.video_stream.stop()

        if self.stream_ws_client:
            self.stream_ws_client.stop()
