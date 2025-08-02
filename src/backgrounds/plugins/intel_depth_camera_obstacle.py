import logging

from backgrounds.base import Background, BackgroundConfig
from providers.intel_depth_camera_obstacle_provider import (
    IntelDepthCameraObstacleProvider,
)


class IntelDepthCameraObstacle(Background):
    """
    Background task for detecting obstacles using Intel Depth Camera.
    """

    def __init__(self, config: BackgroundConfig = BackgroundConfig()):
        super().__init__(config)

        self.depth_camera_provider = IntelDepthCameraObstacleProvider(
            depth_threshold=0.2571
        )
        self.depth_camera_provider.start()
        logging.info("Intel Depth Camera Obstacle Provider started in background")
