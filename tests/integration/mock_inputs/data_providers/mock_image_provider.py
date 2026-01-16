import logging
from typing import List, Optional

import numpy as np
from PIL import Image


# This is a singleton class that will store and serve images for any VLM that needs them
class MockImageProvider:
    """
    Singleton class to provide mock images to any VLM implementation.

    This class serves as a central repository for test images that can be
    used by different VLM implementations during testing.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MockImageProvider, cls).__new__(cls)
            cls._instance.test_images = []
            cls._instance.current_index = 0
            logging.info("Initialized MockImageProvider singleton")
        return cls._instance

    def load_images(self, images: List[Image.Image]):
        """
        Load a sequence of test images.

        Parameters
        ----------
        images : List[Image.Image]
            List of PIL images to use for testing
        """
        self.test_images = images
        self.current_index = 0
        logging.info(f"MockImageProvider loaded {len(self.test_images)} test images")

    def get_next_image(self) -> Optional[Image.Image]:
        """
        Get the next image in the sequence as a PIL Image.

        Returns
        -------
        Optional[Image.Image]
            Next test image or None if no more images
        """
        if not self.test_images or self.current_index >= len(self.test_images):
            return None

        image = self.test_images[self.current_index]
        self.current_index += 1
        return image

    def get_next_opencv_image(self) -> Optional[np.ndarray]:
        """
        Get the next image in the sequence as an OpenCV-compatible numpy array (BGR).

        Returns
        -------
        Optional[np.ndarray]
            Next test image as a numpy array, or None if no more images
        """
        pil_image = self.get_next_image()
        if pil_image is None:
            return None

        # Convert PIL Image to numpy array in BGR format (what OpenCV would return)
        if pil_image.mode == "RGB":
            # Convert RGB PIL Image to BGR numpy array (OpenCV format)
            np_image = np.array(pil_image)
            np_image = np_image[:, :, ::-1].copy()  # RGB to BGR
        else:
            # If not RGB, convert to RGB first then to BGR
            np_image = np.array(pil_image.convert("RGB"))
            np_image = np_image[:, :, ::-1].copy()  # RGB to BGR

        return np_image

    def reset(self):
        """Reset the image provider to start from the first image again."""
        self.current_index = 0


# Helper functions to access the singleton
def get_image_provider() -> MockImageProvider:
    """Get the singleton image provider instance."""
    return MockImageProvider()


def load_test_images(images: List[Image.Image]):
    """Load test images into the provider."""
    provider = get_image_provider()
    provider.load_images(images)


def get_next_opencv_image() -> Optional[np.ndarray]:
    """Get the next test image as an OpenCV-compatible numpy array."""
    provider = get_image_provider()
    return provider.get_next_opencv_image()
