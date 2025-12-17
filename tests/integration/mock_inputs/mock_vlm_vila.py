import asyncio
import base64
import json
import logging
import threading
import time
from queue import Queue
from typing import List, Optional

import cv2
from om1_vlm import VideoStream

from inputs.base.loop import FuserInput
from inputs.plugins.vlm_vila import Message, VLMVila, VLMVilaConfig
from providers.io_provider import IOProvider
from providers.vlm_vila_provider import VLMVilaProvider
from tests.integration.mock_inputs.data_providers.mock_image_provider import (
    get_next_opencv_image,
)


class MockVideoStream(VideoStream):
    """
    Mock video stream that sends mock images to frame callbacks instead of camera frames.
    """

    def __init__(self, original_video_stream, vlm_provider=None):
        """
        Initialize mock video stream with the same callbacks as the original.

        Parameters
        ----------
        original_video_stream : VideoStream
            The original video stream to copy callbacks from
        """
        self.frame_callbacks = original_video_stream.frame_callbacks
        self.running = False
        self._video_thread: Optional[threading.Thread] = None
        self.fps = original_video_stream.fps
        self.frame_delay = 1.0 / self.fps
        self.encode_quality = original_video_stream.encode_quality
        self.vlm_provider = vlm_provider  # Store the provider reference

        # Create event loop for async callbacks (same as original)
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self._start_loop, daemon=True)
        self.loop_thread.start()

        logging.info(
            "MockVideoStream initialized with %d callbacks", len(self.frame_callbacks)
        )

    def register_frame_callback(self, frame_callback):
        """Register a new frame callback - exactly like the original."""
        if frame_callback and frame_callback not in self.frame_callbacks:
            self.frame_callbacks.append(frame_callback)
            logging.debug("MockVideoStream: Registered new frame callback")

    def _start_loop(self):
        """Set and run the event loop forever in a dedicated thread."""
        asyncio.set_event_loop(self.loop)
        logging.debug("MockVideoStream: Started background event loop")
        self.loop.run_forever()

    def _wait_for_connections(self, timeout=30):
        """
        Wait for WebSocket connections to be established before sending images.

        Parameters
        ----------
        timeout : int
            Maximum time to wait for connections in seconds
        """
        if not self.vlm_provider:
            logging.warning(
                "MockVideoStream: No VLM provider reference, skipping connection wait"
            )
            return True

        start_time = time.time()
        while time.time() - start_time < timeout:
            # Check if both WebSocket clients are connected
            main_connected = (
                hasattr(self.vlm_provider, "ws_client")
                and self.vlm_provider.ws_client.is_connected()
            )

            stream_connected = True  # Default to True if no stream client
            if (
                hasattr(self.vlm_provider, "stream_ws_client")
                and self.vlm_provider.stream_ws_client
            ):
                stream_connected = self.vlm_provider.stream_ws_client.is_connected()

            if main_connected and stream_connected:
                logging.info("MockVideoStream: WebSocket connections established")
                return True

            logging.debug(
                f"MockVideoStream: Waiting for connections... main: {main_connected}, stream: {stream_connected}"
            )
            time.sleep(0.5)

        logging.warning(
            f"MockVideoStream: Connection timeout after {timeout}s, proceeding anyway"
        )
        return False

    def on_video(self):
        """
        Mock video capture loop that sends mock images to callbacks.
        """
        logging.info("MockVideoStream: Starting video processing")

        if not self._wait_for_connections():
            logging.warning(
                "MockVideoStream: Proceeding without all connections established"
            )

        frame_time = 1.0 / self.fps
        last_frame_time = time.perf_counter()
        images_sent = 0

        try:
            while self.running:
                current_time = time.perf_counter()
                elapsed = current_time - last_frame_time

                # Get mock image - LOOP INSTEAD OF STOPPING
                mock_image = get_next_opencv_image()
                if mock_image is None:
                    # Reset the image provider and try again instead of stopping
                    logging.debug(
                        "MockVideoStream: Resetting image provider to loop images"
                    )
                    from tests.integration.mock_inputs.data_providers.mock_image_provider import (
                        get_image_provider,
                    )

                    get_image_provider().reset()
                    mock_image = get_next_opencv_image()

                    if mock_image is None:
                        logging.error(
                            "MockVideoStream: No images available even after reset"
                        )
                        self.running = False
                        break

                # Process frame similar to original VideoStream
                if elapsed <= 1.5 * frame_time and self.frame_callbacks:
                    _, buffer = cv2.imencode(".jpg", mock_image, self.encode_quality)
                    frame_data = base64.b64encode(buffer.tobytes()).decode("utf-8")

                    # Send to all registered callbacks
                    for frame_callback in self.frame_callbacks:
                        try:
                            if asyncio.iscoroutinefunction(frame_callback):
                                asyncio.run_coroutine_threadsafe(
                                    frame_callback(frame_data), self.loop
                                )
                            else:
                                frame_callback(frame_data)
                        except Exception as e:
                            logging.error(
                                f"MockVideoStream: Error calling frame callback: {e}"
                            )

                    images_sent += 1

                # Maintain frame rate
                elapsed_time = time.perf_counter() - last_frame_time
                if elapsed_time < frame_time:
                    time.sleep(frame_time - elapsed_time)
                last_frame_time = time.perf_counter()

                # Log progress every 10 images to reduce spam
                if images_sent > 0 and images_sent % 10 == 0:
                    logging.debug(f"MockVideoStream: Sent {images_sent} images to VLM")

        except Exception as e:
            logging.error(f"MockVideoStream: Error in video processing: {e}")
        finally:
            logging.info(f"MockVideoStream: Finished processing {images_sent} images")

    def start(self):
        """Start the mock video processing thread."""
        if self._video_thread is None or not self._video_thread.is_alive():
            self.running = True
            self._video_thread = threading.Thread(target=self.on_video, daemon=True)
            self._video_thread.start()
            logging.info("MockVideoStream: Started video thread")

    def stop(self):
        """Stop the mock video processing."""
        self.running = False
        if self._video_thread and self._video_thread.is_alive():
            self._video_thread.join(timeout=1.0)
        logging.info("MockVideoStream: Stopped video thread")


class MockVLM_Vila(VLMVila):
    """
    Mock implementation of VLM_Vila that uses the central image provider.

    This class overrides only the camera input functionality to get images from
    the mock image provider, while maintaining all the real object detection logic.
    """

    def __init__(self, config: VLMVilaConfig = VLMVilaConfig()):
        """
        Initialize with the real VLM implementation but without opening camera.

        Parameters
        ----------
        config : VLMVilaConfig, optional
            Configuration for the sensor
        """

        # Initialize base FuserInput class
        super(FuserInput, self).__init__(config)  # Skip VLMVila.__init__

        # Override the descriptor to indicate this is a mock
        self.descriptor_for_LLM = "MOCK Vision INPUT (Vila Test)"

        # Track IO
        self.io_provider = IOProvider()

        # Buffer for storing the final output
        self.messages: List[Message] = []

        # Buffer for storing messages
        self.message_buffer: Queue[str] = Queue()

        # Initialize VLM provider
        api_key = getattr(self.config, "api_key", None)
        base_url = getattr(self.config, "base_url", "wss://api-vila.openmind.org")
        stream_base_url = getattr(
            self.config,
            "stream_base_url",
            f"wss://api.openmind.org/api/core/teleops/stream/video?api_key={api_key}",
        )
        self.vlm: VLMVilaProvider = VLMVilaProvider(
            ws_url=base_url, stream_url=stream_base_url
        )
        self.vlm.video_stream.stop()
        logging.debug("MockVLM_Vila: Stopped original video stream")
        self.vlm.video_stream = MockVideoStream(self.vlm.video_stream, self.vlm)

        # Start the mock video stream (it will wait for connections)
        self.vlm.start()
        self.vlm.register_message_callback(self._handle_vlm_message)

        logging.info("MockVLM_Vila initialized successfully")

    def _handle_vlm_message(self, raw_message: str):
        """
        Process incoming VLM messages - EXACTLY like the real VLMVila.
        """
        logging.debug(
            "MockVLM_Vila: Received message: %s",
            raw_message[:100] + "..." if len(raw_message) > 100 else raw_message,
        )

        # Add the missing JSON parsing logic from the real VLMVila
        try:
            json_message = json.loads(raw_message)
            if "vlm_reply" in json_message:
                vlm_reply = json_message["vlm_reply"]
                self.message_buffer.put(vlm_reply)
                logging.info(
                    "MockVLM_Vila: Detected VLM message: %s",
                    vlm_reply[:50] + "..." if len(vlm_reply) > 50 else vlm_reply,
                )
        except json.JSONDecodeError:
            pass  # Handle non-JSON messages gracefully

    async def _poll(self) -> Optional[str]:
        """
        Enhanced poll method with debugging.
        """
        await asyncio.sleep(0.5)

        # Only log when there are messages to avoid spam
        buffer_size = self.message_buffer.qsize()
        if buffer_size > 0:
            logging.debug(f"MockVLM_Vila: Buffer contains {buffer_size} messages")

        try:
            message = self.message_buffer.get_nowait()
            logging.debug("MockVLM_Vila: Retrieved message from buffer")
            return message
        except Exception:
            # Don't log this as it's expected when buffer is empty
            return None

    def cleanup(self):
        """
        Synchronous cleanup method for proper resource cleanup.
        """
        try:
            if hasattr(self, "vlm") and self.vlm:
                if hasattr(self.vlm, "video_stream"):
                    self.vlm.video_stream.stop()
                self.vlm.stop()
            logging.info("MockVLM_Vila: Cleanup completed")
        except Exception as e:
            logging.error(f"MockVLM_Vila: Error during cleanup: {e}")

    def __del__(self):
        """
        Properly clean up resources when the object is destroyed.
        """
        self.cleanup()
