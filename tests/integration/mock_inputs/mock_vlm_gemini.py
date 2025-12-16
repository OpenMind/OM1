import asyncio
import base64
import logging
import time

import cv2

from inputs.plugins.vlm_gemini import VLMGemini, VLMGeminiConfig
from tests.integration.mock_inputs.data_providers.mock_image_provider import (
    get_next_opencv_image,
)


class MockVLM_Gemini(VLMGemini):
    """
    Mock implementation of VLM_Gemini that uses the central image provider.

    This class overrides only the camera input functionality to get images from
    the mock image provider, while maintaining all the real object detection logic.
    """

    def __init__(self, config: VLMGeminiConfig = VLMGeminiConfig()):
        """
        Initialize with the real VLM implementation but without opening camera.

        Parameters
        ----------
        config : VLMGeminiConfig, optional
            Configuration for the sensor
        """
        # Initialize using the real VLM_Gemini implementation
        super().__init__(config)

        # Override the descriptor to indicate this is a mock
        self.descriptor_for_LLM = "MOCK Vision INPUT (Gemini Test)"

        # Stop the original video stream
        if hasattr(self.vlm, "video_stream"):
            self.vlm.video_stream.stop()
            logging.info("MockVLM_Gemini: Stopped original video stream")

        # Store the last processed time to rate-limit our mock frames
        self.last_processed_time = 0

        # Track if we've processed all images
        self.images_processed = False

        # Start the mock image processing loop
        self.running = True
        self._mock_thread = asyncio.create_task(self._process_mock_images())

        logging.info("MockVLM_Gemini initialized - using mock image provider")

    async def _process_mock_images(self):
        """
        Continuously process mock images and send them to the VLM provider.
        """
        while self.running and not self.images_processed:
            # Rate limit to avoid overwhelming the API
            current_time = time.time()
            if (
                current_time - self.last_processed_time < 1.0
            ):  # One second minimum between frames
                await asyncio.sleep(0.1)
                continue

            self.last_processed_time = current_time

            # Get a mock image
            image = get_next_opencv_image()
            if image is None:
                logging.info("MockVLM_Gemini: No more images to process")
                self.images_processed = True
                continue

            logging.info(
                f"MockVLM_Gemini: Processing mock image with shape {image.shape}"
            )

            # Convert image to base64 string
            _, buffer = cv2.imencode(".jpg", image)
            base64_image = base64.b64encode(buffer.tobytes()).decode("utf-8")

            # Process the image using the VLM provider's frame callback
            if hasattr(self.vlm, "_process_frame"):
                await self.vlm._process_frame(base64_image)

            await asyncio.sleep(0.1)

    def cleanup(self):
        """
        Synchronous cleanup method for proper resource cleanup.
        """
        try:
            self.running = False
            if hasattr(self, "_mock_thread") and self._mock_thread:
                self._mock_thread.cancel()

            if hasattr(self, "vlm") and self.vlm and hasattr(self.vlm, "video_stream"):
                self.vlm.video_stream.stop()

            logging.info("MockVLM_Gemini: Cleanup completed")
        except Exception as e:
            logging.error(f"MockVLM_Gemini: Error during cleanup: {e}")

    def __del__(self):
        """Clean up resources when the object is destroyed."""
        self.cleanup()
