import logging
import time
from typing import Callable, Optional

from om1_vlm import VideoRTSPStream
from openai import AsyncOpenAI

from .singleton import singleton


@singleton
class VLMOpenAIRTSPProvider:
    """
    VLM Provider that handles audio streaming and websocket communication.

    This class implements a singleton pattern to manage video stream from RTSP and websocket
    communication for vlm services. It runs in a separate thread to handle
    continuous vlm processing.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        rtsp_url: str = "rtsp://localhost:8554/top_camera",
        decode_format: str = "H264",
        prompt: str = "What is the most interesting aspect in this series of images?",
        fps: int = 30,
    ):
        """
        Initialize the VLM Provider.

        Parameters
        ----------
        base_url : str
            The base URL for the OM API.
        api_key : str
            The API key for the OM API.
        rtsp_url : str
            The RTSP URL for the video stream. Defaults to "rtsp://localhost:8554/top_camera".
        decode_format : str
            The decode format for the video stream. Defaults to "H264".
        fps : int
            The fps for the VLM service connection.
        """
        self.running: bool = False
        self.api_client: AsyncOpenAI = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.video_stream: VideoRTSPStream = VideoRTSPStream(
            rtsp_url,
            decode_format,
            frame_callback=self._process_frame,  # type: ignore
            fps=fps,
        )
        self.message_callback: Optional[Callable] = None
        self.prompt = prompt

    async def _process_frame(self, frame: str):
        """
        Process a video frame using the LLM API.

        Parameters
        ----------
        frame : str
            The base64 encoded video frame to process.
        """
        processing_start = time.perf_counter()
        try:
            response = await self.api_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self.prompt,
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
            logging.debug(f"OpenAI LLM VLM Response: {response}")
            if self.message_callback:
                self.message_callback(response)
        except Exception as e:
            logging.error(f"Error processing frame: {e}")

    def register_message_callback(self, message_callback: Optional[Callable]):
        """
        Register a callback for processing VLM results.

        Parameters
        ----------
        callback : callable
            The callback function to process VLM results.
        """
        self.message_callback = message_callback

    def start(self):
        """
        Start the VLM RTSP provider.

        Initializes and starts the websocket client, video stream, and processing thread
        if not already running.
        """
        if self.running:
            logging.warning("VLM RTSP provider is already running")
            return

        self.running = True
        self.video_stream.start()

        logging.info("OpenAI VLM RTSP provider started")

    def stop(self):
        """
        Stop the VLM RTSP provider.

        Stops the websocket client, video stream, and processing thread.
        """
        self.running = False

        self.video_stream.stop()
