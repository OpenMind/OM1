import asyncio
import logging
import time
from typing import Optional

import serial

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider


class SerialReader(FuserInput[SensorConfig, Optional[str]]):
    """
    Reads data from serial port, typically from an Arduino.

    Maintains a buffer of messages.
    """

    # simple code example to ingest serial data written by an Arduino, such as:

    #       if (grip_force > grip_force_threshold) {
    #         Serial.println("Grip: Elevated");
    #       } else {
    #         Serial.println("Grip: Normal");
    #       }

    #       if (pulse_rate > pulse_threshold) {
    #         Serial.println("Pulse: Elevated");
    #       } else {
    #         Serial.println("Pulse: Normal");
    #       }

    #

    def __init__(self, config: SensorConfig):
        """
        Initialize with empty message buffer.
        """
        super().__init__(config)

        # Configure the serial port
        port = "/dev/cu.usbmodem1101"  # Replace with your serial port
        baudrate = 9600
        timeout = 1  # Optional: set a timeout for reading

        self.ser = None

        try:
            # Open the serial port
            self.ser = serial.Serial(port, baudrate, timeout=timeout)
            logging.info(f"Connected to {port} at {baudrate} baud")
        except serial.SerialException as e:
            logging.error(f"Error: {e}")

        # Track IO
        self.io_provider = IOProvider()

        # Messages buffer
        self.messages: list[Message] = []

        self.descriptor_for_LLM = "Heart Rate and Grip Strength"

    async def _poll(self) -> Optional[str]:
        """
        Poll for serial data.

        Returns
        -------
        Optional[str]
            The latest line read from the serial port, or None if no data
        """
        await asyncio.sleep(0.5)

        if self.ser is None:
            return None

        data = self.ser.readline().decode("utf-8").strip()
        # Read a line, decode, and remove whitespace

        if data:
            logging.info(f"Serial: {data}")
            return data
        else:
            return None

    async def _raw_to_text(self, raw_input: Optional[str]) -> Optional[Message]:
        """
        Process raw string to higher level text description.

        Parameters
        ----------
        raw_input : Optional[str]
            Raw input string to be processed

        Returns
        -------
        Optional[Message]
            A timestamped message containing the processed input
        """
        if raw_input is None:
            return None

        if "Pulse:" in raw_input:
            value = raw_input.split(" ")
            message = f"The child's pulse rate is {value[1]}."
        elif "Grip:" in raw_input:
            value = raw_input.split(" ")
            message = f"The child's grip strength is {value[1]}."
        else:
            message = "No serial data."

        return Message(timestamp=time.time(), message=message)

    async def raw_to_text(self, raw_input: Optional[str]):
        """
        Update message buffer.

        Parameters
        ----------
        raw_input : Optional[str]
            Raw input to be processed, or None if no input is available
        """
        pending_message = await self._raw_to_text(raw_input)

        if pending_message is not None:
            self.messages.append(pending_message)

    def formatted_latest_buffer(self) -> Optional[str]:
        """
        Format and clear the latest buffer contents.

        Formats the most recent message with timestamp and class name,
        adds it to the IO provider, then clears the buffer.

        Returns
        -------
        Optional[str]
            Formatted string of buffer contents or None if buffer is empty
        """
        if len(self.messages) == 0:
            return None

        latest_message = self.messages[-1]

        result = f"""
INPUT: {self.descriptor_for_LLM}
// START
{latest_message.message}
// END
"""

        self.io_provider.add_input(
            self.__class__.__name__, latest_message.message, latest_message.timestamp
        )
        self.messages = []

        return result
