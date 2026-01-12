import asyncio
import logging
import re
import time
from queue import Queue
from threading import Lock
from typing import Dict, Optional, Tuple

from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.follow_person.interface import FollowPersonInput
from providers.io_provider import IOProvider


class FollowPersonConfig(ActionConfig):
    """
    Configuration for FollowPerson ROS2 connector.

    Parameters
    ----------
    person_detection_topic : str
        ROS2 topic for person detection messages (default: "/person_detection").
    movement_command_topic : str
        ROS2 topic for sending movement commands (default: "/cmd_vel").
    update_rate_hz : float
        Update rate for following control loop (default: 10.0).
    max_following_distance : float
        Maximum following distance in meters (default: 5.0).
    min_following_distance : float
        Minimum safe following distance in meters (default: 0.8).
    linear_speed_max : float
        Maximum linear velocity in m/s (default: 0.5).
    angular_speed_max : float
        Maximum angular velocity in rad/s (default: 0.5).
    position_tolerance : float
        Distance tolerance in meters for considering target reached (default: 0.2).
    angle_tolerance : float
        Angle tolerance in radians for alignment (default: 0.1).
    """

    person_detection_topic: str = Field(
        default="/person_detection",
        description="ROS2 topic for person detection messages.",
    )
    movement_command_topic: str = Field(
        default="/cmd_vel",
        description="ROS2 topic for sending movement commands.",
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


class FollowPersonConnector(ActionConnector[FollowPersonConfig, FollowPersonInput]):
    """
    Connector for following a person using ROS2.

    This connector subscribes to person detection topics (or reads from VLM inputs)
    and publishes movement commands to maintain a safe following distance.
    """

    def __init__(self, config: FollowPersonConfig):
        """
        Initialize the FollowPerson connector.

        Parameters
        ----------
        config : FollowPersonConfig
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
        self._last_person_position: Optional[Tuple[float, float, float]] = None  # (distance, angle, timestamp)
        self._last_update_time = time.time()
        self._person_lost_time: Optional[float] = None

        # Control state
        self._control_lock = Lock()
        self._movement_queue: Queue[Dict[str, float]] = Queue()

        # ROS2 setup would go here in a real implementation
        # For now, we'll use logging to indicate what would be sent
        logging.info(
            f"FollowPerson ROS2 connector initialized: "
            f"detection_topic={config.person_detection_topic}, "
            f"cmd_topic={config.movement_command_topic}"
        )

    def _parse_person_from_vlm(self) -> Optional[Dict[str, any]]:
        """
        Parse person detection information from VLM inputs.

        This method reads from IOProvider inputs (typically from VLM_COCO_Local
        or similar vision inputs) and extracts person position information.

        Returns
        -------
        Optional[Dict]
            Dictionary with person information, or None if not found.
            Format: {
                "name": str,
                "distance": float,  # in meters
                "angle": float,      # in radians
                "confidence": float
            }
        """
        inputs = self.io_provider.inputs

        # Look for VLM inputs that might contain person information
        for key, input_obj in inputs.items():
            if "VLM" in key or "vision" in key.lower() or "coco" in key.lower():
                text = input_obj.input.lower()

                # Try to extract person information from VLM descriptions
                # Example: "You see a person named alice, 2.5 meters away, to your left"
                person_patterns = [
                    r"person\s+named\s+(\w+)",
                    r"(\w+)\s+is\s+(\d+\.?\d*)\s+meters?\s+away",
                    r"(\w+)\s+(\d+\.?\d*)\s+meters?",
                    r"(\d+\.?\d*)\s+meters?\s+.*?(\w+)",
                ]

                for pattern in person_patterns:
                    matches = re.findall(pattern, text)
                    if matches:
                        # Found potential person information
                        # This is a simplified parser - real implementation would be more robust
                        logging.debug(f"Found person info in {key}: {matches}")

        return None

    def _get_person_position(
        self, person_id: Optional[str] = None
    ) -> Optional[Tuple[float, float]]:
        """
        Get the current position of the target person.

        This method attempts to get person position from:
        1. ROS2 person detection topic (if available)
        2. VLM inputs parsed from IOProvider
        3. Face presence provider (if person is enrolled)

        Parameters
        ----------
        person_id : Optional[str]
            The person identifier to track. If None, uses self._target_person_id.

        Returns
        -------
        Optional[Tuple[float, float]]
            Tuple of (distance, angle) in meters and radians, or None if not found.
        """
        target_id = person_id or self._target_person_id

        if target_id is None:
            return None

        # Method 1: Try to parse from VLM inputs
        person_info = self._parse_person_from_vlm()
        if person_info:
            return (person_info.get("distance", 0.0), person_info.get("angle", 0.0))

        # Method 2: Check face presence provider data
        # In a real implementation, this would integrate with face detection service
        # to get position information

        # Method 3: ROS2 topic subscription (would be implemented here)
        # if self.ros2_node:
        #     person_msg = self.ros2_node.get_latest_person_detection()
        #     if person_msg and person_msg.person_id == target_id:
        #         return (person_msg.distance, person_msg.angle)

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

        # PID-like control (simplified)
        kp_distance = 0.3
        kp_angle = 0.5

        # Linear velocity: move forward/backward based on distance error
        linear_vel = kp_distance * distance_error * self._follow_speed
        linear_vel = max(
            -self.config.linear_speed_max,
            min(self.config.linear_speed_max, linear_vel),
        )

        # Angular velocity: turn to face the person
        angular_vel = kp_angle * angle_error * self._follow_speed
        angular_vel = max(
            -self.config.angular_speed_max,
            min(self.config.angular_speed_max, angular_vel),
        )

        # If very close to target, reduce speeds
        if abs(distance_error) < self.config.position_tolerance:
            linear_vel *= 0.3
        if abs(angle_error) < self.config.angle_tolerance:
            angular_vel *= 0.3

        return {"linear": linear_vel, "angular": angular_vel}

    def _publish_movement(self, linear: float, angular: float) -> None:
        """
        Publish movement command to ROS2.

        Parameters
        ----------
        linear : float
            Linear velocity in m/s.
        angular : float
            Angular velocity in rad/s.
        """
        # In a real implementation, this would publish to ROS2:
        # msg = Twist()
        # msg.linear.x = linear
        # msg.angular.z = angular
        # self.cmd_vel_publisher.publish(msg)

        # For now, log the command
        logging.debug(
            f"FollowPerson: Publishing movement cmd_vel: linear={linear:.3f}, angular={angular:.3f}"
        )

        # Also write to status for debugging
        self._write_status(
            f"moving linear={linear:.2f} angular={angular:.2f}"
        )

    def _write_status(self, message: str) -> None:
        """
        Write status message to fuser.

        Parameters
        ----------
        message : str
            Status message.
        """
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

        # Handle stop command
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
            f"Starting to follow: mode={follow_mode}, person={person_id}, "
            f"distance={self._target_distance}m, speed={self._follow_speed}"
        )

        self._write_status(
            f"following mode={follow_mode} person={person_id or 'auto'} "
            f"distance={self._target_distance:.1f}m"
        )

        # Start the control loop
        try:
            await self._follow_control_loop()
        except Exception as e:
            logging.error(f"Error in follow control loop: {e}", exc_info=True)
            self._is_following = False
            self._write_status(f"error reason={str(e)}")

    async def _follow_control_loop(self) -> None:
        """
        Main control loop for following behavior.

        This loop runs at the configured update rate and continuously:
        1. Gets person position
        2. Calculates movement commands
        3. Publishes commands
        4. Handles edge cases (person lost, too close, etc.)
        """
        update_interval = 1.0 / self.config.update_rate_hz
        start_time = time.time()

        while self._is_following:
            current_time = time.time()
            elapsed = current_time - start_time

            # Check timeout
            if elapsed > self._timeout_sec:
                logging.warning(f"Follow timeout after {elapsed:.1f}s")
                self._write_status("timeout stopping")
                self.stop_following()
                break

            # Get person position
            person_position = self._get_person_position()

            if person_position is None:
                # Person not detected
                if self._person_lost_time is None:
                    self._person_lost_time = current_time
                    self._write_status("person not detected")
                elif current_time - self._person_lost_time > 5.0:
                    # Person lost for more than 5 seconds
                    logging.warning("Person lost for >5s, stopping follow")
                    self._write_status("person lost stopping")
                    self.stop_following()
                    break
                else:
                    # Person lost but within timeout, keep trying
                    self._write_status(
                        f"person lost {current_time - self._person_lost_time:.1f}s"
                    )
                    # Stop movement while person is lost
                    self._publish_movement(0.0, 0.0)
                await asyncio.sleep(update_interval)
                continue

            # Person found - reset lost timer
            self._person_lost_time = None
            distance, angle = person_position

            # Update last known position
            self._last_person_position = (distance, angle, current_time)
            self._last_update_time = current_time

            # Check if person is too far
            if distance > self.config.max_following_distance:
                self._write_status(
                    f"person too far distance={distance:.2f}m max={self.config.max_following_distance:.2f}m"
                )
                # Stop movement
                self._publish_movement(0.0, 0.0)
                await asyncio.sleep(update_interval)
                continue

            # Check if person is too close
            if distance < self.config.min_following_distance:
                self._write_status(
                    f"person too close distance={distance:.2f}m min={self.config.min_following_distance:.2f}m stopping"
                )
                # Move backward slightly
                cmd = self._calculate_movement_command(
                    distance, angle, self.config.min_following_distance + 0.2
                )
                self._publish_movement(cmd["linear"], cmd["angular"])
                await asyncio.sleep(update_interval)
                continue

            # Calculate movement command
            cmd = self._calculate_movement_command(
                distance, angle, self._target_distance
            )

            # Check if we're at target distance
            distance_error = abs(distance - self._target_distance)
            angle_error = abs(angle)

            if (
                distance_error < self.config.position_tolerance
                and angle_error < self.config.angle_tolerance
                and self._stop_on_arrival
            ):
                # At target position, stop
                self._write_status(
                    f"at target distance={distance:.2f}m error={distance_error:.2f}m"
                )
                self._publish_movement(0.0, 0.0)
            else:
                # Move towards target
                self._write_status(
                    f"following distance={distance:.2f}m target={self._target_distance:.2f}m "
                    f"error={distance_error:.2f}m angle={angle:.2f}rad"
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

        # Stop movement
        self._publish_movement(0.0, 0.0)
        logging.info("FollowPerson: Stopped following")

    def tick(self) -> None:
        """
        Periodic tick method for connector maintenance.

        This is called periodically by the action orchestrator to allow
        the connector to perform maintenance tasks.
        """
        # Check if following is still active and handle timeouts
        if self._is_following:
            time_since_update = time.time() - self._last_update_time
            if time_since_update > self._timeout_sec:
                logging.warning(
                    f"Person lost for {time_since_update:.1f}s, stopping follow"
                )
                self.stop_following()
                self._write_status("timeout stopping")
