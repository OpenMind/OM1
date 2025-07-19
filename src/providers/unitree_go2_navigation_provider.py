import logging
from typing import Optional

import zenoh
from zenoh import ZBytes

from zenoh_idl import geometry_msgs, nav_msgs

from .singleton import singleton

status_map = {
    0: "UNKNOWN",
    1: "ACCEPTED",
    2: "EXECUTING",
    3: "CANCELING",
    4: "SUCCEEDED",
    5: "CANCELED",
    6: "ABORTED",
}


@singleton
class UnitreeGo2NavigationProvider:
    """
    Navigation Provider for Unitree Go2 robot.
    """

    def __init__(
        self,
        navigation_status_topic: str = "navigate_to_pose/_action/status",
        goal_pose_topic: str = "goal_pose",
    ):
        """
        Initialize the Unitree Go2 Navigation Provider with a specific topic.

        Parameters
        ----------
        navigation_status_topic : str, optional
            The topic on which to subscribe for navigation messages (default is "navigate_to_pose/_action/status").
        goal_pose_topic : str, optional
            The topic on which to publish goal poses (default is "goal_pose").
        """
        self.session: Optional[zenoh.Session] = None

        try:
            self.session = zenoh.open(zenoh.Config())
            logging.info("Zenoh client opened")
        except Exception as e:
            logging.error(f"Error opening Zenoh client: {e}")

        self.navigation_status_topic = navigation_status_topic
        self.navigation_status = "UNKNOWN"

        self.goal_pose_topic = goal_pose_topic

        self.running: bool = False

    def navigation_status_message_callback(self, data: zenoh.Sample):
        """
        Process an incoming navigation status message.

        Parameters
        ----------
        data : zenoh.Sample
            The Zenoh sample received, which should have a 'payload' attribute.
        """
        if data.payload:
            message: nav_msgs.Nav2Status = nav_msgs.Nav2Status.deserialize(
                data.payload.to_bytes()
            )
            logging.debug("Received Navigation Status message: %s", message)
            status_list = message.status_list
            if status_list:
                latest_status = status_list[-1]
                logging.info("Latest Navigation Status: %s", latest_status)
                self.navigation_status = status_map.get(latest_status.status, "UNKNOWN")
                logging.info("Navigation Status: %s", self.navigation_status)
        else:
            logging.warning("Received empty navigation status message")

    def start(self):
        """
        Start the navigation provider by registering the message callback and starting the listener.
        """
        if self.session is None:
            logging.error(
                "Cannot start navigation provider; Zenoh session is not available."
            )
            return

        if not self.running:
            self.session.declare_subscriber(
                self.navigation_status_topic, self.navigation_status_message_callback
            )
            logging.info(
                "Subscribed to navigation status topic: %s",
                self.navigation_status_topic,
            )

            self.running = True
            logging.info("Navigation Provider started and listening for messages")
            return

        logging.warning("Navigation Provider is already running")

    def publish_goal_pose(self, pose: geometry_msgs.PoseStamped):
        """
        Publish a goal pose to the navigation topic.

        Parameters
        ----------
        pose : geometry_msgs.PoseStamped
            The goal pose to be published.
        """
        if self.session is None:
            logging.error("Cannot publish goal pose; Zenoh session is not available.")
            return

        payload = ZBytes(pose.serialize())
        self.session.put(self.goal_pose_topic, payload)
        logging.info("Published goal pose to topic: %s", self.goal_pose_topic)

    @property
    def navigation_state(self) -> str:
        """
        Get the current navigation state.

        Returns
        -------
        str
            The current navigation state as a string.
        """
        return self.navigation_status
