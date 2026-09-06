import logging
import math
import time
from typing import Optional, Tuple

import zenoh
from pydantic import Field
from zenoh import ZBytes

from actions.base import ActionConfig, ActionConnector
from actions.explore.interface import ExploreInput
from providers.elevenlabs_tts_provider import ElevenLabsTTSProvider
from providers.unitree_go2_frontier_exploration import (
    UnitreeGo2FrontierExplorationProvider,
)
from providers.unitree_go2_navigation_provider import UnitreeGo2NavigationProvider
from providers.unitree_go2_odom_provider import UnitreeGo2OdomProvider
from zenoh_msgs import (
    Header,
    Point,
    Pose,
    PoseStamped,
    Quaternion,
    Time,
    open_zenoh_session,
)


class UnitreeGo2ExploreConfig(ActionConfig):
    """Configuration for Unitree Go2 Explore connector."""

    explore_start_topic: str = Field(
        default="explore/start",
        description="Topic untuk memulai eksplorasi",
    )
    explore_stop_topic: str = Field(
        default="explore/stop",
        description="Topic untuk menghentikan eksplorasi",
    )
    unitree_ethernet: str = Field(
        default="eth0",
        description="Ethernet channel untuk odometry",
    )
    return_speed: float = Field(
        default=0.5,
        description="Kecepatan saat kembali ke awal (m/s)",
    )


class UnitreeGo2ExploreConnector(
    ActionConnector[UnitreeGo2ExploreConfig, ExploreInput]
):
    """Explore connector for Unitree Go2 robots.

    Manages frontier-based autonomous exploration by:
    - Publishing start/stop commands via Zenoh
    - Monitoring exploration completion via UnitreeGo2FrontierExplorationProvider
    - Optionally returning to the starting position when done
    - Providing voice feedback via ElevenLabs TTS
    """

    def __init__(self, config: UnitreeGo2ExploreConfig):
        """Initialize the UnitreeGo2ExploreConnector.

        Parameters
        ----------
        config : UnitreeGo2ExploreConfig
            Configuration for the connector.
        """
        super().__init__(config)

        self._exploring: bool = False
        self._start_position: Optional[Tuple[float, float, float]] = None
        self._start_time: Optional[float] = None
        self._duration: Optional[int] = None
        self._return_to_start: bool = True

        self.odom_provider = UnitreeGo2OdomProvider(
            channel=self.config.unitree_ethernet
        )
        self.exploration_provider = UnitreeGo2FrontierExplorationProvider()
        self.exploration_provider.start()

        self.navigation_provider = UnitreeGo2NavigationProvider()
        self.navigation_provider.start()

        self.tts_provider = ElevenLabsTTSProvider()

        self.session: Optional[zenoh.Session] = None
        self._start_pub = None
        self._stop_pub = None

        try:
            self.session = open_zenoh_session()
            self._start_pub = self.session.declare_publisher(
                self.config.explore_start_topic
            )
            self._stop_pub = self.session.declare_publisher(
                self.config.explore_stop_topic
            )
            logging.info(
                "ExploreConnector: Zenoh publishers declared on '%s' and '%s'",
                self.config.explore_start_topic,
                self.config.explore_stop_topic,
            )
        except Exception as e:
            logging.error("ExploreConnector: Failed to open Zenoh session: %s", e)

    async def connect(self, output_interface: ExploreInput) -> None:
        """Handle an explore command from the LLM or user.

        Parameters
        ----------
        output_interface : ExploreInput
            Contains action ("explore" or "stop explore"), optional duration,
            and return_to_start flag.
        """
        action = output_interface.action.lower().strip()

        if action == "explore":
            if self._exploring:
                logging.warning("ExploreConnector: Already exploring, ignoring.")
                return

            try:
                pos = self.odom_provider.position
                if pos:
                    self._start_position = (
                        float(pos.get("odom_x", 0.0)),
                        float(pos.get("odom_y", 0.0)),
                        float(pos.get("odom_yaw_m180_p180", 0.0)),
                    )
                    logging.info(
                        "ExploreConnector: Start position captured: %s",
                        self._start_position,
                    )
                else:
                    logging.warning("ExploreConnector: Odom not available.")
                    self._start_position = None
            except Exception as e:
                logging.error("ExploreConnector: Error reading odom: %s", e)
                self._start_position = None

            self._start_time = time.time()
            self._duration = output_interface.duration
            self._return_to_start = output_interface.return_to_start
            self._exploring = True
            self.exploration_provider.exploration_complete = False

            self._publish(self._start_pub, b"start", self.config.explore_start_topic)
            self.tts_provider.add_pending_message(
                "Starting exploration. I will map the area. Woof!"
            )
            logging.info(
                "ExploreConnector: Exploration started. Duration=%s, ReturnToStart=%s",
                self._duration,
                self._return_to_start,
            )

        elif action == "stop explore":
            if not self._exploring:
                logging.warning("ExploreConnector: Not exploring, ignoring stop.")
                return
            self._stop_exploration()
            self.tts_provider.add_pending_message("Exploration stopped. Woof!")

        else:
            logging.warning("ExploreConnector: Unknown action '%s'.", action)

    def tick(self) -> None:
        """Periodic tick called by the runtime loop.

        Checks duration and exploration completion status.
        Triggers return-to-start if needed.
        """
        if not self._exploring:
            self.sleep(1.0)
            return

        duration_exceeded = (
            self._duration is not None
            and self._start_time is not None
            and (time.time() - self._start_time) > self._duration
        )
        exploration_complete = self.exploration_provider.status

        if duration_exceeded or exploration_complete:
            reason = "duration exceeded" if duration_exceeded else "no more frontiers"
            logging.info("ExploreConnector: Stopping exploration (%s).", reason)
            self._stop_exploration()
            self.tts_provider.add_pending_message(
                "Exploration complete. I have finished mapping the area. Woof!"
            )
            if self._return_to_start and self._start_position is not None:
                self._navigate_to_start()
        else:
            self.sleep(1.0)

    def _stop_exploration(self) -> None:
        """Publish stop command dan reset state eksplorasi."""
        self._publish(self._stop_pub, b"stop", self.config.explore_stop_topic)
        self._exploring = False
        logging.info("ExploreConnector: Exploration stopped.")

    def _navigate_to_start(self) -> None:
        """Navigasi robot kembali ke posisi awal saat eksplorasi selesai."""
        if self._start_position is None:
            logging.warning("ExploreConnector: No start position, cannot return.")
            return

        x, y, yaw = self._start_position

        try:
            now = Time(sec=int(time.time()), nanosec=0)
            header = Header(stamp=now, frame_id="map")
            position_msg = Point(x=x, y=y, z=0.0)
            qz = math.sin(yaw / 2.0)
            qw = math.cos(yaw / 2.0)
            orientation_msg = Quaternion(x=0.0, y=0.0, z=qz, w=qw)
            pose_msg = Pose(position=position_msg, orientation=orientation_msg)
            goal_pose = PoseStamped(header=header, pose=pose_msg)
            self.navigation_provider.publish_goal_pose(goal_pose, "starting point")
            self.tts_provider.add_pending_message(
                "Returning to the starting point. Woof!"
            )
            logging.info("ExploreConnector: Navigation to start initiated.")
        except Exception as e:
            logging.error("ExploreConnector: Failed to navigate to start: %s", e)

    def _publish(self, publisher, payload: bytes, topic: str) -> None:
        """Publish payload ke Zenoh topic, dengan penanganan error."""
        if publisher is None:
            logging.warning(
                "ExploreConnector: Publisher for '%s' not available.", topic
            )
            return
        try:
            publisher.put(ZBytes(payload))
            logging.info("ExploreConnector: Published to '%s'.", topic)
        except Exception as e:
            logging.error("ExploreConnector: Failed to publish to '%s': %s", topic, e)

    def stop(self) -> None:
        """Stop the connector and clean up resources."""
        if self._exploring:
            logging.info("ExploreConnector: Stopping on shutdown.")
            self._stop_exploration()

        if self._start_pub:
            try:
                self._start_pub.undeclare()
            except Exception:
                pass

        if self._stop_pub:
            try:
                self._stop_pub.undeclare()
            except Exception:
                pass

        if self.session:
            try:
                self.session.close()
                logging.info("ExploreConnector: Zenoh session closed.")
            except Exception as e:
                logging.error("ExploreConnector: Error closing session: %s", e)

        super().stop()
