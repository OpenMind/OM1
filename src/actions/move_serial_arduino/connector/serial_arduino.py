"""
This only works if you actually have a serial port connected to your computer, such as, via a USB serial dongle. On Mac, you can determine the correct name to use via `ls /dev/cu.usb*`.
"""

import logging
import time

import serial

from actions.base import ActionConfig, ActionConnector
from actions.move.interface import MoveInput


class MoveSerialConnector(ActionConnector[MoveInput]):

    def __init__(self, config: ActionConfig):
        super().__init__(config)

        # Open the serial port
        self.port = (
            ""  # specify your serial port here, such as COM1 or /dev/cu.usbmodem14101
        )
        self.ser = None
        if self.port:
            self.ser = serial.Serial(self.port, 9600)

    async def connect(self, output_interface: MoveInput) -> None:

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
        time.sleep(0.1)
        # logging.info("Connector Tick")
