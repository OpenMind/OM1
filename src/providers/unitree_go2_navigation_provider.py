import logging
from typing import Optional
from uuid import uuid4

import zenoh
from zenoh import ZBytes

from providers.elevenlabs_tts_provider import ElevenLabsTTSProvider
from zenoh_msgs import (
    AIStatusRequest,
    String,
    geometry_msgs,
    nav_msgs,
    open_zenoh_session,
    prepare_header,
)

from .singleton import singleton

# Nav2 Action Status Codes
status_map = {
    0: "UNKNOWN",
    1: "ACCEPTED",
    2: "EXECUTING",
    3: "CANCELING",
    4: "SUCCEEDED",  # Only this status re-enables AI mode
    5: "CANCELED",
    6: "ABORTED",
}


@singleton
class UnitreeGo2NavigationProvider:
    """
    Navigation Provider for Unitree Go2 robot.

    This class implements a singleton pattern to manage:
        * Navigation goal publishing to ROS2 Nav2 stack
        * Navigation status monitoring from ROS2 action server
        * Automatic AI mode control based on navigation state

    The provider automatically manages AI mode control during navigation:
    - Disables AI mode when navigation starts (ACCEPTED/EXECUTING status)
    - Re-enables AI mode only on successful navigation completion (SUCCEEDED status)
    - Keeps AI mode disabled on navigation failure/cancellation (CANCELED/ABORTED status)
    """

    def __init__(
        self,
        navigation_status_topic: str = "navigate_to_pose/_action/status",
        goal_pose_topic: str = "goal_pose",
        cancel_goal_topic: str = "navigate_to_pose/_action/cancel_goal",
    ):
        """
        Initialize the Unitree Go2 Navigation Provider with a specific topic.

        Parameters
        ----------
        navigation_status_topic : str, optional
            The ROS2 topic to subscribe for navigation status messages.
            Default: "navigate_to_pose/_action/status"
            Alternative: "navigate_to_pose/_action/feedback" for more detailed updates
        goal_pose_topic : str, optional
            The topic on which to publish goal poses (default is "goal_pose").
        cancel_goal_topic : str, optional
            The topic on which to publish goal cancellations (default is "navigate_to_pose/_action/cancel_goal").
        """
        self.session: Optional[zenoh.Session] = None

        try:
            self.session = open_zenoh_session()
            logging.info("Zenoh client opened")
        except Exception as e:
            logging.error(f"Error opening Zenoh client: {e}")

        self.navigation_status_topic = navigation_status_topic
        self.navigation_status = "UNKNOWN"

        self.goal_pose_topic = goal_pose_topic
        self.cancel_goal_topic = cancel_goal_topic

        self.running: bool = False
        self._nav_in_progress: bool = False
        self._current_destination: Optional[str] = None  # Track destination name

        # TTS provider for speech feedback
        self.tts_provider = ElevenLabsTTSProvider()

        # AI status control
        self.ai_status_topic = "om/ai/request"
        self.ai_status_pub = None
        if self.session:
            try:
                self.ai_status_pub = self.session.declare_publisher(
                    self.ai_status_topic
                )
                logging.info(
                    "AI status publisher initialized on topic: %s", self.ai_status_topic
                )
            except Exception as e:
                logging.error(f"Error creating AI status publisher: {e}")

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
                latest_status = status_list[-1]  # type: ignore
                status_code = latest_status.status
                self.navigation_status = status_map.get(status_code, "UNKNOWN")
                logging.info(
                    "Received navigation status from ROS2 topic '/navigate_to_pose/_action/status': %s (code=%d)",
                    self.navigation_status,
                    status_code,
                )

                # Track navigation state and AI mode control
                # AI mode is ONLY re-enabled on STATUS_SUCCEEDED (4)
                if status_code in (1, 2):  # ACCEPTED or EXECUTING
                    if not self._nav_in_progress:
                        self._nav_in_progress = True
                        self._publish_ai_status(
                            enabled=False
                        )  # Disable AI during navigation
                        logging.info("Navigation started - AI mode disabled")
                elif (
                    status_code == 4
                ):  # STATUS_SUCCEEDED - Navigation completed successfully
                    if self._nav_in_progress:
                        self._nav_in_progress = False
                        self._publish_ai_status(
                            enabled=True
                        )  # Re-enable AI ONLY on success
                        logging.info("Navigation succeeded - AI mode re-enabled")

                        # Add speech feedback for successful navigation
                        if self._current_destination:
                            self.tts_provider.add_pending_message(
                                f"Yaaay! I have reached the {self._current_destination}. Woof! Woof!"
                            )
                        else:
                            self.tts_provider.add_pending_message(
                                "Yaaay! I have reached my destination. Woof! Woof!"
                            )
                elif status_code in (5, 6):  # CANCELED or ABORTED
                    if self._nav_in_progress:
                        self._nav_in_progress = False
                        # Do NOT re-enable AI mode on failure/cancellation
                        logging.warning(
                            "Navigation %s (code=%d) - AI mode remains disabled",
                            self.navigation_status,
                            status_code,
                        )
        else:
            logging.warning("Received empty navigation status message")

    def _publish_ai_status(self, enabled: bool):
        """
        Publish AI status to enable or disable AI mode during navigation.

        Parameters
        ----------
        enabled : bool
            True to enable AI mode, False to disable.
        """
        if self.ai_status_pub is None:
            logging.warning("AI status publisher not available")
            return

        try:
            header = prepare_header("map")
            status_msg = AIStatusRequest(
                header=header,
                request_id=String(str(uuid4())),
                code=1 if enabled else 0,
            )
            self.ai_status_pub.put(status_msg.serialize())
            logging.info(
                "AI mode %s during navigation", "enabled" if enabled else "disabled"
            )
        except Exception as e:
            logging.error(f"Error publishing AI status: {e}")

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

    def publish_goal_pose(
        self, pose: geometry_msgs.PoseStamped, destination_name: Optional[str] = None
    ):
        """
        Publish a goal pose to the navigation topic.

        Parameters
        ----------
        pose : geometry_msgs.PoseStamped
            The goal pose to be published.
        destination_name : Optional[str]
            Name of the destination for speech feedback
        """
        if self.session is None:
            logging.error("Cannot publish goal pose; Zenoh session is not available.")
            return

        # Store destination name for speech feedback
        self._current_destination = destination_name

        # Disable AI mode immediately when navigation goal is published
        if not self._nav_in_progress:
            self._publish_ai_status(enabled=False)
            logging.info("Navigation goal published - AI mode disabled immediately")

        self._nav_in_progress = True
        payload = ZBytes(pose.serialize())
        self.session.put(self.goal_pose_topic, payload)
        logging.info("Published goal pose to topic: %s", self.goal_pose_topic)

    def clear_goal_pose(self):
        """
        Clear/cancel all active navigation goals.
        Publishes to the cancel_goal topic to stop navigation.
        """
        if self.session is None:
            logging.error("Cannot cancel goal; Zenoh session is not available.")
            return

        try:
            # Send cancel request to Nav2
            # Empty payload should cancel all active goals
            cancel_payload = ZBytes(b"")
            self.session.put(self.cancel_goal_topic, cancel_payload)
            logging.info("Sent cancel all goals request to: %s", self.cancel_goal_topic)
            self._nav_in_progress = False
        except Exception:
            logging.exception("Failed to cancel navigation goals")

    def stop(self):
        """
        Stop the navigation provider by unsubscribing from topics and cleaning up resources.
        """
        self.running = False

        if self.session:
            self.session.close()
            logging.info("Zenoh session closed")

        if self.ai_status_pub:
            self.ai_status_pub.undeclare()
            logging.info("AI status publisher closed")

        logging.warning("Navigation Provider is not running")

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

    @property
    def is_navigating(self) -> bool:
        """
        Check if navigation is currently in progress.

        Returns
        -------
        bool
            True if navigation is in progress, False otherwise.
        """
        return self._nav_in_progress
