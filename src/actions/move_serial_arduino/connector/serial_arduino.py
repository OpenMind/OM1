"""
This only works if you actually have a serial port connected to your computer, such as, via a USB serial dongle. On Mac, you can determine the correct name to use via `ls /dev/cu.usb*`.
"""

import logging
import time

import serial
from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.move.interface import MoveInput


class MoveSerialConfig(ActionConfig):
    """
    Configuration for Serial to Arduino connector.

    Parameters
    ----------
    port : str
        The serial port to connect to the Arduino (e.g., COM3 or /dev/cu.usbmodem14101). Leave empty to simulate.
    """

    port: str = Field(
        default="",
        description="The serial port to connect to the Arduino (e.g., COM3 or /dev/cu.usbmodem14101). Leave empty to simulate.",
    )


class MoveSerialConnector(ActionConnector[MoveSerialConfig, MoveInput]):
    """
    Connector that sends move commands via serial to an Arduino.
    """

    def __init__(self, config: MoveSerialConfig):
        """
        Initialize the MoveSerialConnector.

        Parameters
        ----------
        config : MoveSerialConfig
            Configuration for the action connector.
        """
        super().__init__(config)

        self.port = self.config.port
        self.ser = None
        if self.port:
            self.ser = serial.Serial(self.port, 9600)

    async def connect(self, output_interface: MoveInput) -> None:
        """
        Connect the input protocol to the move action via serial to Arduino.

        Parameters
        ----------
        output_interface : MoveInput
            The input protocol containing the action details.
        """
        new_msg = {"move": ""}

        if output_interface.action == "be still":
            new_msg["move"] = "0"
        elif output_interface.action == "small jump":
            new_msg["move"] = "1"
        elif output_interface.action == "medium jump":
            new_msg["move"] = "2"
        elif output_interface.action == "big jump":
            new_msg["move"] = "3"
        else:
            logging.info(f"Other move type: {output_interface.action}")
            # raise ValueError(f"Unknown move type: {output_interface.action}")

        message = f"actuator:{new_msg['move']}\r\n"
        # Convert the string to bytes using UTF-8 encoding
        byte_data = message.encode("utf-8")

        if self.ser is not None and self.ser.is_open:
            logging.info(f"SendToArduinoSerial: {message}")
            self.ser.write(byte_data)
        else:
            logging.info(f"SerialNotOpen - Simulating transmit: {message}")

    def tick(self) -> None:
        """
        Periodic tick function to maintain connection.
        """
        time.sleep(0.1)
