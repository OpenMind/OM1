import logging
from typing import Callable, Optional

import zenoh

from zenoh_msgs import Pose, nav_msgs

from .singleton import singleton
from .zenoh_listener_provider import ZenohListenerProvider


@singleton
class UnitreeGo2LidarLocalizationProvider(ZenohListenerProvider):
    """
    Lidar Localization provider for Unitree Go2 robot.
    """

    def __init__(
        self,
        topic: str = "om/localization_pose",
        quality_tolerance: float = 0.9,
    ):
        """
        Initialize the Lidar Localization Provider with a specific topic.

        Parameters
        ----------
        topic : str, optional
            The topic on which to subscribe for lidar localization messages (default is "om/localization_pose").
        quality_tolerance : float, optional
            The tolerance for localization quality percent (default is 0.9).
        """
        super().__init__(topic)
        logging.info("Lidar Localization Provider initialized with topic: %s", topic)

        self.localization_pose: Optional[Pose] = None
        self.localization_status = False
        self.quality_tolerance = quality_tolerance

    def lidar_localization_message_callback(self, data: zenoh.Sample):
        """
        Process an incoming lidar localization message.

        Parameters
        ----------
        data : zenoh.Sample
            The Zenoh sample received, which should have a 'payload' attribute.
        """
        if data.payload:
            message: nav_msgs.LidarLocalization = (
                nav_msgs.LidarLocalization.deserialize(data.payload.to_bytes())
            )
            logging.debug("Received Lidar Localization message: %s", message)

            quality_percent = message.quality_percent
            self.localization_status = quality_percent >= self.quality_tolerance
            self.localization_pose = message.pose

            logging.debug(
                "Localization Status: %s, Pose: %s",
                self.localization_status,
                self.localization_pose,
            )

        else:
            logging.warning("Received empty lidar localization message")

    def start(self, message_callback: Optional[Callable] = None):
        """
        Start the Lidar Localization Provider by registering the message callback.
        """
        if not self.running:
            self.register_message_callback(self.lidar_localization_message_callback)
            self.running = True
            logging.info(
                "Lidar Localization Provider started and listening for messages"
            )
        else:
            logging.warning("Lidar Localization Provider is already running")

    @property
    def is_localized(self) -> bool:
        """
        Check if the robot is localized based on the Lidar Localization data.

        Returns
        -------
        bool
            True if the robot is localized, False otherwise.
        """
        return self.localization_status

    @property
    def pose(self) -> Optional[Pose]:
        """
        Get the current localization pose.

        Returns
        -------
        Optional[Pose]
            The current pose if available, None otherwise.
        """
        return self.localization_pose
