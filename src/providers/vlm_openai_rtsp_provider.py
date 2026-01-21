import asyncio
import json
import logging
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional

from om1_vlm import VideoRTSPStream
from openai import AsyncOpenAI

from .singleton import singleton


@singleton
class VLMOpenAIRTSPProvider:
    """
    VLM Provider that handles video streaming and websocket communication.

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
        batch_size: int = 5,
        batch_interval: float = 0.5,
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
        prompt : str
            The prompt for the VLM analysis. Defaults to "What is the most interesting aspect in this series of images?".
        fps : int
            The fps for the VLM service connection.
        batch_size : int
            Number of frames to collect before sending to OpenAI. Defaults to 5.
        batch_interval : float
            Time interval in seconds between batch processing. Defaults to 0.5.
        """
        self.running: bool = False
        self.api_client: AsyncOpenAI = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.video_stream: VideoRTSPStream = VideoRTSPStream(
            rtsp_url,
            decode_format,
            frame_callback=self._queue_frame,  # type: ignore
            fps=fps,
        )
        self.message_callback: Optional[Callable] = None
        self.prompt = prompt
        self.batch_size = batch_size
        self.batch_interval = batch_interval
        self.frame_queue: deque = deque(maxlen=batch_size)
        self.batch_task: Optional[asyncio.Task] = None

    def _queue_frame(self, frame_data: str):
        """
        Queue a video frame for batch processing.

        Parameters
        ----------
        frame_data : str
            A JSON string containing base64 encoded frame data.
        """
        try:
            frame = json.loads(frame_data)["frame"]
            self.frame_queue.append(frame)
            logging.debug(f"Queued frame, queue size: {len(self.frame_queue)}")
        except Exception as e:
            logging.error(f"Error queuing frame: {e}")

    async def _process_batch(self):
        """
        Process batches of frames at regular intervals.
        """
        while self.running:
            await asyncio.sleep(self.batch_interval)

            if len(self.frame_queue) > 0:
                frames_to_process = list(self.frame_queue)
                self.frame_queue.clear()

                await self._send_batch_to_openai(frames_to_process)

    async def _send_batch_to_openai(self, frames: List[str]):
        """
        Send a batch of frames to OpenAI API.

        Parameters
        ----------
        frames : List[str]
            List of base64 encoded frame data.
        """
        processing_start = time.perf_counter()
        try:
            content: List[Dict[str, Any]] = [
                {
                    "type": "text",
                    "text": f"{self.prompt} (Analyzing {len(frames)} frames)",
                }
            ]

            for i, frame in enumerate(frames):
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{frame}",
                            "detail": "low",
                        },
                    }
                )

            response = await self.api_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": content,  # type: ignore
                    }
                ],
                max_tokens=300,
            )

            processing_latency = time.perf_counter() - processing_start
            logging.debug(f"Batch processing latency: {processing_latency:.3f} seconds")
            logging.debug(f"Processed {len(frames)} frames")
            logging.debug(f"OpenAI LLM VLM Response: {response}")

            if self.message_callback:
                self.message_callback(response)

        except Exception as e:
            logging.error(f"Error processing batch: {e}")

    def register_message_callback(self, message_callback: Optional[Callable]):
        """
        Register a callback for processing VLM results.

        Parameters
        ----------
        message_callback: Optional[Callable]
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

        # Start the batch processing task
        self.batch_task = asyncio.create_task(self._process_batch())

        logging.info("OpenAI VLM RTSP provider started with batch processing")

    def stop(self):
        """
        Stop the VLM RTSP provider.

        Stops the websocket client, video stream, and processing thread.
        """
        self.running = False

        self.video_stream.stop()

        # Cancel the batch processing task
        if self.batch_task and not self.batch_task.done():
            self.batch_task.cancel()

        # Clear any remaining frames
        self.frame_queue.clear()

        logging.info("OpenAI VLM RTSP provider stopped")
