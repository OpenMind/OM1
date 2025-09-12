import logging
from typing import Optional

import numpy as np
import zenoh

from zenoh_msgs import (
    AIControlStatus,
    Pose,
    nav_msgs,
    open_zenoh_session,
    prepare_header,
)

from .singleton import singleton
from .zenoh_listener_provider import ZenohListenerProvider


@singleton
class UnitreeGo2AMCLProvider(ZenohListenerProvider):
    """
    AMCL Provider for Unitree Go2 robot.
    """

    def __init__(
        self,
        topic: str = "amcl_pose",
        pose_tolerance: float = 0.2,
        yaw_tolerance: float = 0.1,
    ):
        """
        Initialize the AMCL Provider with a specific topic.
        Parameters
        ----------
        topic : str, optional
            The topic on which to subscribe for AMCL messages (default is "amcl").
        pose_tolerance : float, optional
            The tolerance for pose covariance (default is 0.2).
        yaw_tolerance : float, optional
            The tolerance for yaw covariance (default is 0.1).
        """
        super().__init__(topic)
        logging.info("AMCL Provider initialized with topic: %s", topic)

        self.localization_pose: Optional[Pose] = None
        self.localization_status = False
        self.pose_tolerance = pose_tolerance
        self.yaw_tolerance = yaw_tolerance

        self.topic = "robot/status/ai_control"
        self.session: Optional[zenoh.Session] = None
        self.pub = None

        try:
            self.session = open_zenoh_session()
            self.pub = self.session.declare_publisher(self.topic)
            logging.info("Zenoh client opened for AMCL Provider")
        except Exception as e:
            logging.error(f"Error opening Zenoh client: {e}")
            self.session = None
            self.pub = None

    def amcl_message_callback(self, data: zenoh.Sample):
        """
        Process an incoming AMCL message.
        Parameters
        ----------
        data : zenoh.Sample
            The Zenoh sample received, which should have a 'payload' attribute.
        """
        if data.payload:
            message: nav_msgs.AMCLPose = nav_msgs.AMCLPose.deserialize(
                data.payload.to_bytes()
            )
            logging.debug("Received AMCL message: %s", message)
            covariance = message.covariance

            pos_uncertainty = np.sqrt(covariance[0] + covariance[7])
            yaw_uncertainty = np.sqrt(covariance[35])

            self.localization_status = (
                pos_uncertainty < self.pose_tolerance
                and yaw_uncertainty < self.yaw_tolerance
            )
            self.localization_pose = message.pose
            logging.info(
                "Localization Status: %s, Pose: %s",
                self.localization_status,
                self.localization_pose,
            )

            if self.pub is not None:
                status_msg = AIControlStatus()
                status_msg.header = prepare_header(message.header.frame_id)
                status_msg.status = 0 if self.localization_status else 1
                self.pub.put(status_msg.serialize())
        else:
            logging.warning("Received empty AMCL message")

    def start(self):
        """
        Start the AMCL Provider by registering the message callback.
        """
        if not self.running:
            self.register_message_callback(self.amcl_message_callback)
            self.running = True
            logging.info("AMCL Provider started and listening for messages")
        else:
            logging.warning("AMCL Provider is already running")

    @property
    def is_localized(self) -> bool:
        """
        Check if the robot is localized based on the AMCL data.
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
