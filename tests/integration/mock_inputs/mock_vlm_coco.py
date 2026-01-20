import asyncio
import logging
from typing import Optional

import numpy as np
from torchvision.models import detection as detection_model

from inputs.base.loop import FuserInput
from inputs.plugins.vlm_coco_local import VLM_COCO_Local, VLM_COCO_LocalConfig
from providers.io_provider import IOProvider
from tests.integration.mock_inputs.data_providers.mock_image_provider import (
    get_next_opencv_image,
)


class MockVLM_COCO(VLM_COCO_Local):
    """
    Mock implementation of VLM_COCO_Local that uses the central image provider.

    This class overrides only the camera input functionality to get images from
    the mock image provider, while maintaining all the real object detection logic.
    """

    def __init__(self, config: VLM_COCO_LocalConfig = VLM_COCO_LocalConfig()):
        """
        Initialize with the real VLM implementation but without opening camera.

        Parameters
        ----------
        config : VLM_COCO_LocalConfig, optional
            Configuration for the sensor
        """
        # Initialize base FuserInput class
        super(FuserInput, self).__init__(config)  # Skip VLM_COCO_Local.__init__

        # Set up the model and class labels like the parent class
        self.device = "cpu"
        self.detection_threshold = 0.7
        self.messages = []
        self.io_provider = IOProvider()
        self.descriptor_for_LLM = "MOCK Vision INPUT (COCO Test)"

        # Initialize the model like the parent class
        self.model = detection_model.fasterrcnn_mobilenet_v3_large_320_fpn(
            weights="FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1",
            progress=True,
            weights_backbone="MobileNet_V3_Large_Weights.IMAGENET1K_V1",
        ).to(self.device)
        self.class_labels = (
            detection_model.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT.meta[
                "categories"
            ]
        )
        self.model.eval()

        # Set mock camera properties without opening real camera
        self.have_cam = True  # Pretend we have a camera
        self.cap = None  # Don't create actual capture object
        self.width = 1280  # Standard webcam width
        self.height = 720  # Standard webcam height
        self.cam_third = int(self.width / 3)

        logging.info("MockVLM_COCO initialized - using mock image provider")

    async def _poll(self) -> Optional[np.ndarray]:
        """
        Override the camera polling to get images from the mock image provider.

        Returns
        -------
        Optional[np.ndarray]
            Next test image as a numpy array (OpenCV format), or None if no more images
        """
        await asyncio.sleep(0.1)  # Small delay to simulate real polling
        logging.info("MockVLM_COCO: Polling for image")

        # Get the next image from the central provider
        image = get_next_opencv_image()
        if image is not None:
            logging.info(f"MockVLM_COCO: Retrieved test image with shape {image.shape}")
        return image
