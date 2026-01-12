import asyncio
import logging
import math
import re
import time
from queue import Queue
from threading import Lock
from typing import Dict, Optional, Tuple

import zenoh
from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.follow_person.interface import FollowPersonInput
from providers.io_provider import IOProvider
from providers.odom_provider import OdomProvider
from zenoh_msgs import geometry_msgs, open_zenoh_session


class FollowPersonZenohConfig(ActionConfig):
    """
    Configuration for FollowPerson Zenoh connector.

    Parameters
    ----------
    URID : Optional[str]
        URID for Zenoh topics (required for Zenoh communication).
    person_detection_topic : str
        Zenoh topic for person detection messages.
    movement_command_topic : str
        Zenoh topic for sending movement commands.
    update_rate_hz : float
        Update rate for following control loop.
    max_following_distance : float
        Maximum following distance in meters.
    min_following_distance : float
        Minimum safe following distance in meters.
    linear_speed_max : float
        Maximum linear velocity in m/s.
    angular_speed_max : float
        Maximum angular velocity in rad/s.
    position_tolerance : float
        Distance tolerance in meters for considering target reached.
    angle_tolerance : float
        Angle tolerance in radians for alignment.
    """

    URID: Optional[str] = Field(
        default=None,
        description="URID for Zenoh topics.",
    )
    person_detection_topic: str = Field(
        default="person_detection",
        description="Zenoh topic for person detection messages.",
    )
    movement_command_topic: str = Field(
        default="cmd_vel",
        description="Zenoh topic for sending movement commands.",
    )
    update_rate_hz: float = Field(
        default=10.0,
        description="Update rate for following control loop.",
    )
    max_following_distance: float = Field(
        default=5.0,
        description="Maximum following distance in meters.",
    )
    min_following_distance: float = Field(
        default=0.8,
        description="Minimum safe following distance in meters.",
    )
    linear_speed_max: float = Field(
        default=0.5,
        description="Maximum linear velocity in m/s.",
    )
    angular_speed_max: float = Field(
        default=0.5,
        description="Maximum angular velocity in rad/s.",
    )
    position_tolerance: float = Field(
        default=0.2,
        description="Distance tolerance in meters for considering target reached.",
    )
    angle_tolerance: float = Field(
        default=0.1,
        description="Angle tolerance in radians for alignment.",
    )


class FollowPersonZenohConnector(ActionConnector[FollowPersonZenohConfig, FollowPersonInput]):
    """
    Connector for following a person using Zenoh.

    This connector uses Zenoh for communication, subscribing to person detection
    topics and publishing movement commands to maintain a safe following distance.
    """

    def __init__(self, config: FollowPersonZenohConfig):
        """
        Initialize the FollowPerson Zenoh connector.

        Parameters
        ----------
        config : FollowPersonZenohConfig
            Configuration for the connector.
        """
        super().__init__(config)
        self.io_provider = IOProvider()

        # State management
        self._is_following = False
        self._target_person_id: Optional[str] = None
        self._follow_mode: Optional[str] = None
        self._target_distance: float = 1.5
        self._follow_speed: float = 0.5
        self._stop_on_arrival: bool = True
        self._timeout_sec: float = 30.0

        # Person tracking
        self._last_person_position: Optional[Tuple[float, float, float]] = None
        self._last_update_time = time.time()
        self._person_lost_time: Optional[float] = None

        # Control state
        self._control_lock = Lock()
        self._movement_queue: Queue[Dict[str, float]] = Queue()

        # Zenoh session
        self.session: Optional[zenoh.Session] = None
        self.odom: Optional[OdomProvider] = None

        # Setup Zenoh
        URID = self.config.URID
        if URID is None:
            logging.warning("FollowPerson Zenoh: No URID provided, Zenoh features disabled")
        else:
            try:
                self.session = open_zenoh_session()
                logging.info(f"FollowPerson Zenoh: Session opened with URID: {URID}")

                # Setup topics
                self.person_detection_topic = f"{URID}/{self.config.person_detection_topic}"
                self.cmd_vel_topic = f"{URID}/{self.config.movement_command_topic}"

                # Setup odometry provider
                self.odom = OdomProvider(URID=URID, use_zenoh=True)

                # Subscribe to person detection
                self.session.declare_subscriber(
                    self.person_detection_topic,
                    self._on_person_detection
                )
                logging.info(f"FollowPerson Zenoh: Subscribed to {self.person_detection_topic}")

            except Exception as e:
                logging.error(f"FollowPerson Zenoh: Error opening session: {e}")
                self.session = None

    def _on_person_detection(self, sample: zenoh.Sample) -> None:
        """
        Callback for person detection messages from Zenoh.

        Parameters
        ----------
        sample : zenoh.Sample
            The Zenoh sample containing person detection data.
        """
        try:
            # In a real implementation, this would deserialize the message
            # For now, we'll parse from the payload or use a standard format
            data = sample.payload.to_bytes()
            # Parse person detection data here
            # This is a placeholder - actual implementation depends on message format
            logging.debug(f"FollowPerson Zenoh: Received person detection: {len(data)} bytes")
        except Exception as e:
            logging.warning(f"FollowPerson Zenoh: Error processing person detection: {e}")

    def _parse_person_from_vlm(self) -> Optional[Dict[str, any]]:
        """
        Parse person detection information from VLM inputs.

        Returns
        -------
        Optional[Dict]
            Dictionary with person information, or None if not found.
        """
        inputs = self.io_provider.inputs

        for key, input_obj in inputs.items():
            if "VLM" in key or "vision" in key.lower() or "coco" in key.lower():
                text = input_obj.input.lower()

                # Try to extract person information
                person_patterns = [
                    r"person\s+named\s+(\w+)",
                    r"(\w+)\s+is\s+(\d+\.?\d*)\s+meters?\s+away",
                    r"(\w+)\s+(\d+\.?\d*)\s+meters?",
                ]

                for pattern in person_patterns:
                    matches = re.findall(pattern, text)
                    if matches:
                        logging.debug(f"Found person info in {key}: {matches}")

        return None

    def _get_person_position(
        self, person_id: Optional[str] = None
    ) -> Optional[Tuple[float, float]]:
        """
        Get the current position of the target person.

        Returns
        -------
        Optional[Tuple[float, float]]
            Tuple of (distance, angle) in meters and radians, or None if not found.
        """
        target_id = person_id or self._target_person_id

        if target_id is None:
            return None

        # Try to parse from VLM inputs
        person_info = self._parse_person_from_vlm()
        if person_info:
            return (person_info.get("distance", 0.0), person_info.get("angle", 0.0))

        # In a real implementation, would also check Zenoh person detection topic
        return None

    def _calculate_movement_command(
        self, distance: float, angle: float, target_distance: float
    ) -> Dict[str, float]:
        """
        Calculate movement command to follow the person.

        Parameters
        ----------
        distance : float
            Current distance to person in meters.
        angle : float
            Current angle to person in radians.
        target_distance : float
            Desired following distance in meters.

        Returns
        -------
        Dict[str, float]
            Movement command with 'linear' and 'angular' velocities.
        """
        distance_error = distance - target_distance
        angle_error = angle

        # PID-like control
        kp_distance = 0.3
        kp_angle = 0.5

        linear_vel = kp_distance * distance_error * self._follow_speed
        linear_vel = max(
            -self.config.linear_speed_max,
            min(self.config.linear_speed_max, linear_vel),
        )

        angular_vel = kp_angle * angle_error * self._follow_speed
        angular_vel = max(
            -self.config.angular_speed_max,
            min(self.config.angular_speed_max, angular_vel),
        )

        if abs(distance_error) < self.config.position_tolerance:
            linear_vel *= 0.3
        if abs(angle_error) < self.config.angle_tolerance:
            angular_vel *= 0.3

        return {"linear": linear_vel, "angular": angular_vel}

    def _publish_movement(self, linear: float, angular: float) -> None:
        """
        Publish movement command via Zenoh.

        Parameters
        ----------
        linear : float
            Linear velocity in m/s.
        angular : float
            Angular velocity in rad/s.
        """
        if self.session is None:
            logging.warning("FollowPerson Zenoh: No session, cannot publish movement")
            return

        try:
            twist = geometry_msgs.Twist(
                linear=geometry_msgs.Vector3(x=float(linear), y=0.0, z=0.0),
                angular=geometry_msgs.Vector3(x=0.0, y=0.0, z=float(angular)),
            )
            self.session.put(self.cmd_vel_topic, twist.serialize())
            logging.debug(
                f"FollowPerson Zenoh: Published cmd_vel: linear={linear:.3f}, angular={angular:.3f}"
            )
            self._write_status(
                f"moving linear={linear:.2f} angular={angular:.2f}"
            )
        except Exception as e:
            logging.error(f"FollowPerson Zenoh: Error publishing movement: {e}")

    def _write_status(self, message: str) -> None:
        """Write status message to fuser."""
        try:
            self.io_provider.add_input("FollowPersonStatus", message, time.time())
        except Exception as e:
            logging.warning(f"Failed to write FollowPersonStatus: {e}")

    async def connect(self, output_interface: FollowPersonInput) -> None:
        """
        Start following the specified person.

        Parameters
        ----------
        output_interface : FollowPersonInput
            The follow person command with target person and parameters.
        """
        action = output_interface.action.lower().strip()

        if action == "stop" or action == "stop following":
            self.stop_following()
            self._write_status("stopped")
            return

        # Determine follow mode
        if action == "nearest":
            follow_mode = "nearest"
            person_id = None
        elif action == "last_seen" or action == "me":
            follow_mode = "last_seen"
            person_id = None
        else:
            follow_mode = "by_name"
            person_id = action

        # Update configuration
        with self._control_lock:
            self._target_person_id = person_id
            self._follow_mode = follow_mode
            self._target_distance = max(
                self.config.min_following_distance,
                min(self.config.max_following_distance, output_interface.distance),
            )
            self._follow_speed = max(0.0, min(1.0, output_interface.speed))
            self._stop_on_arrival = output_interface.stop_on_arrival
            self._timeout_sec = output_interface.timeout_sec
            self._is_following = True
            self._person_lost_time = None

        logging.info(
            f"FollowPerson Zenoh: Starting to follow: mode={follow_mode}, person={person_id}, "
            f"distance={self._target_distance}m, speed={self._follow_speed}"
        )

        self._write_status(
            f"following mode={follow_mode} person={person_id or 'auto'} "
            f"distance={self._target_distance:.1f}m"
        )

        try:
            await self._follow_control_loop()
        except Exception as e:
            logging.error(f"FollowPerson Zenoh: Error in control loop: {e}", exc_info=True)
            self._is_following = False
            self._write_status(f"error reason={str(e)}")

    async def _follow_control_loop(self) -> None:
        """Main control loop for following behavior."""
        update_interval = 1.0 / self.config.update_rate_hz
        start_time = time.time()

        while self._is_following:
            current_time = time.time()
            elapsed = current_time - start_time

            if elapsed > self._timeout_sec:
                logging.warning(f"FollowPerson Zenoh: Timeout after {elapsed:.1f}s")
                self._write_status("timeout stopping")
                self.stop_following()
                break

            person_position = self._get_person_position()

            if person_position is None:
                if self._person_lost_time is None:
                    self._person_lost_time = current_time
                    self._write_status("person not detected")
                elif current_time - self._person_lost_time > 5.0:
                    logging.warning("FollowPerson Zenoh: Person lost for >5s")
                    self._write_status("person lost stopping")
                    self.stop_following()
                    break
                else:
                    self._write_status(
                        f"person lost {current_time - self._person_lost_time:.1f}s"
                    )
                    self._publish_movement(0.0, 0.0)
                await asyncio.sleep(update_interval)
                continue

            self._person_lost_time = None
            distance, angle = person_position
            self._last_person_position = (distance, angle, current_time)
            self._last_update_time = current_time

            if distance > self.config.max_following_distance:
                self._write_status(
                    f"person too far distance={distance:.2f}m"
                )
                self._publish_movement(0.0, 0.0)
                await asyncio.sleep(update_interval)
                continue

            if distance < self.config.min_following_distance:
                self._write_status(
                    f"person too close distance={distance:.2f}m stopping"
                )
                cmd = self._calculate_movement_command(
                    distance, angle, self.config.min_following_distance + 0.2
                )
                self._publish_movement(cmd["linear"], cmd["angular"])
                await asyncio.sleep(update_interval)
                continue

            cmd = self._calculate_movement_command(
                distance, angle, self._target_distance
            )

            distance_error = abs(distance - self._target_distance)
            angle_error = abs(angle)

            if (
                distance_error < self.config.position_tolerance
                and angle_error < self.config.angle_tolerance
                and self._stop_on_arrival
            ):
                self._write_status(
                    f"at target distance={distance:.2f}m"
                )
                self._publish_movement(0.0, 0.0)
            else:
                self._write_status(
                    f"following distance={distance:.2f}m target={self._target_distance:.2f}m "
                    f"error={distance_error:.2f}m"
                )
                self._publish_movement(cmd["linear"], cmd["angular"])

            await asyncio.sleep(update_interval)

    def stop_following(self) -> None:
        """Stop following the person."""
        with self._control_lock:
            self._is_following = False
            self._target_person_id = None
            self._follow_mode = None
            self._person_lost_time = None

        self._publish_movement(0.0, 0.0)
        logging.info("FollowPerson Zenoh: Stopped following")

    def tick(self) -> None:
        """Periodic tick method for connector maintenance."""
        if self._is_following:
            time_since_update = time.time() - self._last_update_time
            if time_since_update > self._timeout_sec:
                logging.warning(
                    f"FollowPerson Zenoh: Person lost for {time_since_update:.1f}s"
                )
                self.stop_following()
                self._write_status("timeout stopping")

