import datetime as datetime
import logging
import re
import threading
import time
from typing import Optional

import serial
from pynmeagps import NMEAReader

from .singleton import singleton


@singleton
class RtkProvider:
    """
    RTK Provider.

    This class implements a singleton pattern to manage:
        * RTK data from serial

    Parameters
    ----------
    serial_port: str = ""
    """

    def __init__(self, serial_port: str = ""):
        """
        Robot and sensor configuration.
        """
        logging.info("Booting RTK Provider")

        baudrate = 115200
        timeout = 0.2  # seconds

        self.serial_connection = None
        try:
            self.serial_connection = serial.Serial(
                serial_port, baudrate, timeout=timeout
            )
            self.serial_connection.reset_input_buffer()
            logging.info(f"Connected to {serial_port} at {baudrate} baud")
        except serial.SerialException as e:
            logging.error(f"Error: {e}")

        self._rtk: Optional[dict] = None

        self.lat = 0.0
        self.lon = 0.0
        self.alt = 0.0
        self.sat = 0
        self.qua = 0
        self.unix_ts = 0.0

        self.running = False
        self._thread: Optional[threading.Thread] = None
        self.start()

    def utc_time_obj_to_unix(self, utc_time_obj):
        """
        Convert a UTC datetime.time object to a Unix timestamp by combining
        it with the local computer's current date.
        """
        if not isinstance(utc_time_obj, datetime.time):
            raise TypeError("Expected a datetime.time object")

        # Get the local date
        local_date = datetime.date.today()

        # Combine local date with provided UTC time
        dt = datetime.datetime.combine(local_date, utc_time_obj).replace(
            tzinfo=datetime.timezone.utc
        )

        # Convert to Unix timestamp
        return dt.timestamp()

    def get_latest_gngga_message(self, nmea_data):
        """
        Extract the latest GNGGA message from a block of NMEA data.

        Parameters
        ----------
        nmea_data : str
            The block of NMEA data as a string.
        """
        pattern = re.compile(
            r"(\$GNGGA,(?P<time>\d{6}(?:\.\d+)?),[^*]*\*[0-9A-Fa-f]{2})", re.MULTILINE
        )

        gngga_entries = []

        matches = pattern.finditer(nmea_data)

        for match in matches:
            # logging.info(f"matches: {match}")
            full_msg = match.group(1)
            time_str = match.group("time")
            try:
                time_val = float(time_str)
                gngga_entries.append((time_val, full_msg))
            except ValueError:
                continue  # Skip if time field is malformed

        # Sort by time and return the latest message
        if gngga_entries:
            most_recent = max(gngga_entries, key=lambda x: x[0])
            # "most_recent" is a time and the message,
            # the [1] just returns the message
            return most_recent[1]

    def magRTKProcessor(self, msg):
        """
        Process incoming RTK NMEA messages.

        Parameters
        ----------
        msg : NMEA message object
            The NMEA message to process.
        """
        try:
            logging.debug(f"RTK:{msg}")

            # NMEA-GN-GGA
            # Description:
            # Standard NMEA: Global positioning system fix data. This message contains time, date,
            # position (in LLH coordinates), fix quality, number of satellites, and horizontal dilution of
            # precision (HDOP) data provided by the selected source.

            if msg and msg.msgID == "GGA":
                try:
                    # round to 1 cm localisation in x,y, and 1 cm in z
                    logging.debug(f"RTK GGA:{msg}")

                    if msg.lat:
                        self.lat = round(float(msg.lat), 7)
                        self.lon = round(float(msg.lon), 7)
                        self.alt = round(float(msg.alt), 2)

                        self.sat = int(msg.numSV)
                        self.qua = int(msg.quality)

                        # the data look something like this: 23:12:25.300000
                        self.unix_ts = self.utc_time_obj_to_unix(msg.time)
                    logging.debug(
                        (
                            f"RTK:{self.lat},{self.lon},ALT:{self.alt},"
                            f"QUA:{self.qua},SAT:{self.sat},TIME:{self.unix_ts}"
                        )
                    )
                except Exception as e:
                    logging.warning(f"Failed to parse GGA message: {msg} ({e})")
        except Exception as e:
            logging.warning(f"Error processing serial RTK input: {msg} ({e})")

        self._rtk = {
            "rtk_lat": self.lat,
            "rtk_lon": self.lon,
            "rtk_alt": self.alt,
            "rtk_sat": self.sat,
            "rtk_qua": self.qua,
            "rtk_unix_ts": self.unix_ts,
        }

    def start(self):
        """
        Starts the RTK Provider and processing thread
        if not already running.
        """
        if self._thread and self._thread.is_alive():
            return

        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        """
        Main loop for the RTK provider.
        """
        while self.running:

            if self.serial_connection:
                bytes_waiting = self.serial_connection.in_waiting
                while bytes_waiting > 0:
                    data = self.serial_connection.read(size=bytes_waiting)
                    if data:
                        data = data.decode("utf-8", errors="ignore")
                        latest_GNGGA = self.get_latest_gngga_message(data)
                        if latest_GNGGA:
                            parsed_nmea = NMEAReader.parse(latest_GNGGA)
                            self.magRTKProcessor(parsed_nmea)
                    bytes_waiting = self.serial_connection.in_waiting

            time.sleep(0.1)

    def stop(self):
        """
        Stop the RTK provider.
        """
        self.running = False
        if self._thread:
            logging.info("Stopping RTK provider")
            self._thread.join(timeout=5)

    @property
    def data(self) -> Optional[dict]:
        """
        Get the current robot RTK data.

        Returns
        -------
        Optional[dict]
            Dictionary containing RTK position data or None if not available
        """
        return self._rtk
