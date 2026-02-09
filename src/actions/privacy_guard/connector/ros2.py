"""Privacy Guard ROS2 connector.

Translates privacy action commands into physical robot responses
via Zenoh message publishing:
- look_away  → publishes a head tilt Vector3 (pitch down by configured angle)
- resume     → publishes a head tilt Vector3 (neutral position)
- stop_stream → publishes CameraStatus DISABLED + head tilt down
- idle       → no-op, robot continues current behavior
"""

import logging
import time
from uuid import uuid4

from actions.base import ActionConfig, ActionConnector
from pydantic import Field

from zenoh_msgs import (
    CameraStatus,
    Vector3,
    prepare_header,
    open_zenoh_session,
)

logger = logging.getLogger(__name__)


class PrivacyGuardConfig(ActionConfig):
    """Configuration for the Privacy Guard connector.

    Parameters
    ----------
    look_away_angle : int
        Degrees to tilt the camera/head downward on breach detection.
    resume_delay_seconds : float
        Minimum seconds to wait before allowing a resume after look_away.
    head_tilt_topic : str
        Zenoh topic for head tilt commands.
    stream_ctrl_topic : str
        Zenoh topic for camera stream control.
    """

    look_away_angle: int = Field(
        default=90,
        description="Degrees to tilt camera/head down on privacy breach",
    )
    resume_delay_seconds: float = Field(
        default=3.0,
        description="Minimum delay before resuming after a look_away",
    )
    head_tilt_topic: str = Field(
        default="om/privacy_guard/head_tilt",
        description="Zenoh topic for head tilt commands",
    )
    stream_ctrl_topic: str = Field(
        default="om/privacy_guard/stream_ctrl",
        description="Zenoh topic for camera stream control",
    )


class PrivacyGuardROS2Connector(ActionConnector["PrivacyGuardConfig", "PrivacyGuardInput"]):
    """Executes privacy guard actions via Zenoh topics.

    Publishes to:
    - om/privacy_guard/head_tilt   (Vector3: pitch/yaw/roll in degrees)
    - om/privacy_guard/stream_ctrl (CameraStatus: ENABLED/DISABLED)
    """

    def __init__(self, config: PrivacyGuardConfig):
        super().__init__(config)
        self._is_looking_away = False
        self._last_look_away_time: float = 0.0

        # Zenoh session and publishers
        self.session = None
        self.head_tilt_pub = None
        self.stream_ctrl_pub = None

        try:
            self.session = open_zenoh_session()
            self.head_tilt_pub = self.session.declare_publisher(
                self.config.head_tilt_topic
            )
            self.stream_ctrl_pub = self.session.declare_publisher(
                self.config.stream_ctrl_topic
            )
            logger.info(
                "Privacy Guard Zenoh session opened — publishing on "
                f"{self.config.head_tilt_topic}, {self.config.stream_ctrl_topic}"
            )
        except Exception as e:
            logger.error(f"Error opening Privacy Guard Zenoh session: {e}")

    def _publish_head_tilt(self, pitch: float, yaw: float = 0.0, roll: float = 0.0):
        """Publish a head tilt command via Zenoh."""
        tilt = Vector3(x=pitch, y=yaw, z=roll)
        if self.head_tilt_pub:
            self.head_tilt_pub.put(tilt.serialize())

    def _publish_stream_ctrl(self, enabled: bool):
        """Publish a camera stream control command via Zenoh."""
        status = CameraStatus(
            header=prepare_header(str(uuid4())),
            status=CameraStatus.STATUS.ENABLED.value if enabled else CameraStatus.STATUS.DISABLED.value,
        )
        if self.stream_ctrl_pub:
            self.stream_ctrl_pub.put(status.serialize())

    async def connect(self, output_interface) -> None:
        """Execute the privacy action on the robot.

        Parameters
        ----------
        output_interface : PrivacyGuardInput
            The privacy action to execute.
        """
        action = output_interface.action

        if action == "look_away":
            self._is_looking_away = True
            self._last_look_away_time = time.time()
            angle = self.config.look_away_angle

            self._publish_head_tilt(pitch=-float(angle))
            self._publish_stream_ctrl(enabled=True)

            logger.warning(
                f"PRIVACY BREACH — Tilting head down {angle}° to avert gaze"
            )

        elif action == "resume":
            if self._is_looking_away:
                elapsed = time.time() - self._last_look_away_time
                if elapsed < self.config.resume_delay_seconds:
                    logger.info(
                        f"Resume blocked — {self.config.resume_delay_seconds - elapsed:.1f}s "
                        "remaining in cooldown"
                    )
                    return

                self._is_looking_away = False
                self._publish_head_tilt(pitch=0.0)
                self._publish_stream_ctrl(enabled=True)
                logger.info("Privacy threat cleared — Resuming neutral position")
            else:
                logger.debug("Already in neutral position, no action needed")

        elif action == "stop_stream":
            self._is_looking_away = True
            self._last_look_away_time = time.time()

            self._publish_head_tilt(pitch=-90.0)
            self._publish_stream_ctrl(enabled=False)

            logger.critical(
                "CRITICAL PRIVACY BREACH — Killing video stream"
            )

        elif action == "idle":
            logger.debug("Environment secure — No privacy action needed")

        else:
            logger.warning(f"Unknown privacy action: {action}")

    def stop(self) -> None:
        """Clean up Zenoh session and publishers."""
        if self.session:
            self.session.close()
            logger.info("Privacy Guard Zenoh session closed")
