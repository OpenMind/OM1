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
    ):
        """
        Initialize the VLM Provider.

        Parameters
        ----------
        base_url : str
            The base URL for the OM API.
        api_key : str
            The API key for the OM API.
        fps : int
            The frames per second for the video stream.
        stream_url : str, optional
            The URL for the video stream. If not provided, defaults to None.
        camera_index : int
            The camera index for the video stream device. Defaults to 0.
        """
        self.running: bool = False
        self.api_client: AsyncOpenAI = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.stream_ws_client: Optional[ws.Client] = (
            ws.Client(url=stream_url) if stream_url else None
        )
        self.video_stream: VideoStream = VideoStream(
            frame_callback=self._process_frame, fps=fps, device_index=camera_index  # type: ignore
        )
        self.message_callback: Optional[Callable] = None

    async def _process_frame(self, frame: str):
        """
        Process a video frame using the Gemini API.

        Parameters
        ----------
        frame : str
            The base64 encoded video frame to process.
        """
        processing_start = time.perf_counter()
        try:
            response = await self.api_client.chat.completions.create(
                model="gemini-2.0-flash-exp",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "What is the most interesting aspect in this series of images?",
                            },
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
                max_tokens=300,
            )
            processing_latency = time.perf_counter() - processing_start
            logging.debug(f"Processing latency: {processing_latency:.3f} seconds")
            logging.debug(f"Gemini LLM VLM Response: {response}")
            if self.message_callback:
                self.message_callback(response)
        except Exception as e:
            logging.error(f"Error processing frame: {e}")

    def register_message_callback(self, message_callback: Optional[Callable]):
        """
        Register a callback for processing Gemini results.

        Parameters
        ----------
        message_callback : Optional[Callable]
            The callback function to process Gemini results.
        """
        self.message_callback = message_callback

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
            self.video_stream.register_frame_callback(
                self.stream_ws_client.send_message
            )

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
