"""
This script is designed to control a robot remotely using keyboard inputs.
It uses the OpenMind API to send movement commands based on key presses.
The script listens for key events and updates the robot's movement accordingly.
w - Move forward
s - Move backward
a - Turn left
d - Turn right
"""

import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass

from om1_utils import ws
from pynput import keyboard

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

OM_API_KEY = os.environ.get("OM_API_KEY", None)
if not OM_API_KEY:
    logging.error("OM_API_KEY environment variable not set.")
    sys.exit(1)


@dataclass
class CommandStatus:
    """
    Data class to represent the command status of a teleops system.
    """

    vx: float
    vy: float
    vyaw: float
    timestamp: str

    def to_dict(self) -> dict:
        """
        Convert the CommandStatus object to a dictionary.

        Returns
        -------
        dict
            Dictionary representation of the CommandStatus object.
        """
        return {
            "vx": self.vx,
            "vy": self.vy,
            "vyaw": self.vyaw,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CommandStatus":
        """
        Populate the CommandStatus object from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing command status information.
        """
        return cls(
            vx=data.get("vx", 0.0),
            vy=data.get("vy", 0.0),
            vyaw=data.get("vyaw", 0.0),
            timestamp=data.get("timestamp", time.time()),
        )


class RemoteMoveController:
    def __init__(self):
        self.ws_client = ws.Client(
            url=f"wss://api.openmind.org/api/core/teleops/command?api_key={OM_API_KEY}"
        )
        self.ws_client.start()

        self.linear = 0.0
        self.angular = 0.0
        self.key_pressed = set()

    def on_press(self, key):
        try:
            k = key.char
        except Exception:
            k = key.name

        self.key_pressed.add(k)
        self.update_movement()

    def on_release(self, key):

        try:
            k = key.char
        except Exception:
            k = key.name

        if k in self.key_pressed:
            self.key_pressed.remove(k)

        self.update_movement()

    def publish_command(self):
        command = CommandStatus(
            vx=self.linear,
            vy=0.0,
            vyaw=self.angular,
            timestamp=time.time(),
        )
        self.ws_client.send_message(json.dumps(command.to_dict()))

        logging.info(f"Published command: {command.to_dict()}")

    def update_movement(self):
        linear = 0.0
        angular = 0.0

        if "w" in self.key_pressed:
            linear = 0.5
        if "s" in self.key_pressed:
            linear = -0.5
        if "a" in self.key_pressed:
            angular = 0.5
        if "d" in self.key_pressed:
            angular = -0.5

        self.linear = linear
        self.angular = angular

        self.publish_command()

        if self.linear > 0:
            logging.info(f"Moving forward {self.linear}")
        elif self.linear < 0:
            logging.info(f"Moving backward {self.linear}")

        if self.angular > 0:
            logging.info(f"Turning left {self.angular}")
        elif self.angular < 0:
            logging.info(f"Turning right {self.angular}")

    def continuous_publish_thread(self):
        publish_rate = 10  # Hz (10 times per second)
        sleep_time = 1.0 / publish_rate

        while True:
            if self.key_pressed:
                self.publish_command()
            time.sleep(sleep_time)

    def control_loop(self):
        listener = keyboard.Listener(
            on_press=lambda key: self.on_press(key),
            on_release=lambda key: self.on_release(key),
        )
        listener.start()

        logging.info("Keyboard Control Active:")
        logging.info(" W - Move Forward")
        logging.info(" S - Move Backward")
        logging.info(" A - Turn Left")
        logging.info(" D - Turn Right")

        publisher_thread = threading.Thread(
            target=self.continuous_publish_thread, daemon=True
        )
        publisher_thread.start()

        while True:
            time.sleep(0.1)


if __name__ == "__main__":
    try:
        move_controller = RemoteMoveController()

        control_thread = threading.Thread(
            target=move_controller.control_loop, daemon=True
        )
        control_thread.start()

        while True:
            time.sleep(0.5)

    except KeyboardInterrupt:
        logging.info("Program interrupted")
    finally:
        logging.info("Exiting program")
        time.sleep(0.5)
        sys.exit(0)
