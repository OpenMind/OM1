import logging
import subprocess
import time
from dataclasses import dataclass

from actions.base import ActionConfig, ActionConnector
from actions.move.interface import MoveInput


@dataclass
class Velocity:
    linear_x: float
    linear_y: float
    linear_z: float
    angular_x: float
    angular_y: float
    angular_z: float


# Predefined velocity commands
VELOCITY_PRESETS = {
    "stand still": Velocity(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "run": Velocity(0.3, 0.0, 0.0, 0.0, 0.0, 0.0),
    "walk forward": Velocity(0.15, 0.0, 0.0, 0.0, 0.0, 0.0),
    "walk backward": Velocity(-0.15, 0.0, 0.0, 0.0, 0.0, 0.0),
    "turn left": Velocity(0.0, 0.0, 0.0, 0.0, 0.0, 0.3),
    "turn right": Velocity(0.0, 0.0, 0.0, 0.0, 0.0, -0.3),
    "look left": Velocity(0.0, 0.0, 0.0, 0.0, 0.0, 0.15),
    "look right": Velocity(0.0, 0.0, 0.0, 0.0, 0.0, -0.15),
    "move left": Velocity(0.0, 0.15, 0.0, 0.0, 0.0, 0.0),
    "move right": Velocity(0.0, -0.15, 0.0, 0.0, 0.0, 0.0),
}


class GazeboConnector(ActionConnector[MoveInput]):
    """
    A connector that publishes Move messages using Gazebo Topics.

    When a Move input is received, the connector publishes the message via the
    gz topic command

    """

    def __init__(self, config: ActionConfig):
        super().__init__(config)
        robot_name = getattr(self.config, "robot_name", None)

        if robot_name is None:
            robot_name = "go2"
            logging.info(f"Robot name not provided. Using default name: {robot_name}")

        self.robot_name = robot_name

    async def connect(self, output_interface: MoveInput) -> None:
        new_msg = {"move": ""}

        # stub to show how to do this
        if output_interface.action == "stand still":
            new_msg["move"] = "stand still"
        elif output_interface.action == "run":
            new_msg["move"] = "run"
        elif output_interface.action == "walk forward":
            new_msg["move"] = "walk forward"
        elif output_interface.action == "walk backward":
            new_msg["move"] = "walk backward"
        elif output_interface.action == "turn left":
            new_msg["move"] = "turn left"
        elif output_interface.action == "turn right":
            new_msg["move"] = "turn right"
        elif output_interface.action == "look left":
            new_msg["move"] = "look left"
        elif output_interface.action == "look right":
            new_msg["move"] = "look right"
        elif output_interface.action == "move left":
            new_msg["move"] = "move left"
        elif output_interface.action == "move right":
            new_msg["move"] = "move right"
        else:
            logging.info(f"Other move type: {output_interface.action}")
            # raise ValueError(f"Unknown move type: {output_interface.action}")

        logging.info(f"SendThisToGazeboConnector: {new_msg}")

        # Publish the Move message using ROS2PublisherProvider.
        self._send_velocity_command(output_interface.action)

    def _send_velocity_command(self, velocity_key: str):
        """
        Sends a velocity command to the Gazebo simulation.

        Args:
            velocity_key (str): Key from VELOCITY_PRESETS dictionary.
        """
        if velocity_key not in VELOCITY_PRESETS:
            logging.info(
                f"WARNING: Preset '{velocity_key}' not found. Defaulting to stand still"
            )
            velocity_key = "stand still"

        velocity = VELOCITY_PRESETS[velocity_key]  # Get the Velocity object

        command = [
            "gz",
            "topic",
            "-t",
            "/model/go2/cmd_vel",
            "-m",
            "gz.msgs.Twist",
            "-p",
            f"linear: {{x: {velocity.linear_x}, y: {velocity.linear_y}, z: {velocity.linear_z}}}, "
            f"angular: {{x: {velocity.angular_x}, y: {velocity.angular_y}, z: {velocity.angular_z}}}",
        ]

        try:
            subprocess.run(command, check=True, timeout=5)
            logging.info(f"Velocity command sent: {velocity}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error sending velocity command: {e}")

    def tick(self) -> None:
        time.sleep(0.1)
