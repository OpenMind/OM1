import logging
import re
import threading
import time
from datetime import datetime, timezone
from typing import List, Optional

import serial

from providers.fabric_map_provider import RFDataRaw

from .singleton import singleton


@singleton
class GpsProvider:
    """
    GPS Provider.

    This class implements a singleton pattern to manage:
        * GPS data from serial

    Parameters
    ----------
    serial_port: str = ""
        The Serial port the Arduino is connected to
    """

    def __init__(self, serial_port: str = ""):
        """
        Robot and sensor configuration.
        """
        logging.info(f"GPS_Provider booting GPS Provider at serial: {serial_port}")

        baudrate = 115200
        timeout = 1

        self.serial_connection = None
        try:
            self.serial_connection = serial.Serial(
                serial_port, baudrate, timeout=timeout
            )
            self.serial_connection.reset_input_buffer()
            logging.info(f"Connected to {serial_port} at {baudrate} baud")
        except serial.SerialException as e:
            logging.error(f"Error: {e}")

        self._gps: Optional[dict] = None

        self.lat = 0.0
        self.lon = 0.0
        self.alt = 0.0
        self.sat = 0
        self.qua = 0

        self.gps_unix_ts = 0.0

        self.yaw_mag_0_360 = 0.0
        self.yaw_mag_cardinal = ""

        self.ble_scan: List[RFDataRaw] = []

        self.running = False
        self._thread: Optional[threading.Thread] = None
        self.start()

    def string_to_unix_timestamp(self, time_str: str) -> float:
        """
        Convert a time string in the format 'YYYY:MM:DD:HH:MM:SS:ms'
        to a Unix timestamp (UTC).

        Parameters
        ----------
        time_str : str
            The input time string.

        Returns
        -------
        float
            The corresponding Unix timestamp.
        """
        dt = datetime.strptime(time_str, "%Y:%m:%d:%H:%M:%S:%f")
        dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()

    def magGPSProcessor(self, data: str):
        """
        Process incoming MAG/GPS/BLE data from serial input.

        Parameters
        ----------
        data : str
            The input data string from serial.
        """
        try:
            if data.startswith("HDG:"):
                parts = data.split(":")
                if len(parts) >= 2:
                    # that's a HDG packet
                    self.yaw_mag_0_360 = float(parts[1])
                    self.yaw_mag_cardinal = self.compass_heading_to_direction(
                        self.yaw_mag_0_360
                    )
                    logging.debug(f"MAG: {self.yaw_mag_0_360}")
                else:
                    logging.warning(f"Unable to parse heading: {data}")
            elif data.startswith("YPR:"):
                yaw, pitch, roll = map(str.strip, data[4:].split(","))
                logging.debug(
                    f"Orientation is Yaw: {yaw}°, Pitch: {pitch}°, Roll: {roll}°."
                )
            elif data.startswith("SAT:"):
                logging.info(f"{data}")
            elif data.startswith("GPS:"):
                try:
                    logging.info(f"{data}")
                    parts = data[4:].split(",")
                    lat = parts[0]
                    lon = parts[1]
                    heading = parts[3].split(":")[1]
                    alt = parts[4].split(":")[1]
                    sat = parts[5].split(":")[1]
                    time = parts[6][5:]
                    # turn 25 into full year -> 2025, for example
                    self.gps_unix_ts = self.string_to_unix_timestamp("20" + time)

                    qua = 0
                    if len(parts) > 7:
                        qua = parts[7].split(":")[1]

                    if "N" in lat:
                        self.lat = float(lat.replace("N", ""))
                    else:
                        self.lat = -1.0 * float(lat.replace("S", ""))

                    if "W" in lon:
                        self.lon = -1.0 * float(lon.replace("W", ""))
                    else:
                        self.lon = float(lon.replace("E", ""))

                    # round to 10 cm localisation in x,y, and 1 cm in z
                    self.lon = round(self.lon, 6)
                    self.lat = round(self.lat, 6)
                    self.alt = round(float(alt), 2)

                    self.sat = int(sat)
                    self.qua = int(qua)

                    logging.debug(
                        (
                            f"Current location is {self.lat}, {self.lon} at {alt}m altitude. "
                            f"GPS Heading {heading}° with {sat} satellites locked. "
                            f"The unix timestamp is {self.gps_unix_ts}. "
                            f"The fix quality is {self.qua}."
                        )
                    )
                except Exception as e:
                    logging.warning(f"Failed to parse GPS: {data} ({e})")
            elif data.startswith("BLE:"):
                try:
                    self.ble_scan = self.parse_ble_triang_string(data)
                    logging.debug(f"nRF BLE data {self.ble_scan}")
                except Exception as e:
                    logging.warning(f"Failed to parse BLE: {data} ({e})")
        except Exception as e:
            logging.warning(f"Error processing serial MAG/GPS/BLE input: {data} ({e})")

        self._gps = {
            "yaw_mag_0_360": self.yaw_mag_0_360,
            "yaw_mag_cardinal": self.yaw_mag_cardinal,
            "gps_lat": self.lat,
            "gps_lon": self.lon,
            "gps_alt": self.alt,
            "gps_sat": self.sat,
            "gps_qua": self.qua,
            "gps_unix_ts": self.gps_unix_ts,
            "ble_scan": self.ble_scan,
        }

    def compass_heading_to_direction(self, degrees: float) -> str:
        """
        Convert compass heading in degrees to cardinal direction.

        Parameters
        ----------
        degrees : float
            Compass heading in degrees.

        Returns
        -------
        str
            Cardinal direction as a string.
        """
        directions = [
            "North",
            "North East",
            "East",
            "South East",
            "South",
            "South West",
            "West",
            "North West",
        ]
        index = int((degrees + 22.5) % 360 / 45)
        return directions[index]

    def parse_ble_triang_string(self, input_string: str) -> List[RFDataRaw]:
        """
        Parse BLE triangulation data from a string.

        Parameters
        ----------
        input_string : str
            The input string containing BLE triangulation data.

        Returns
        -------
        List[RFDataRaw]
            A list of RFDataRaw objects parsed from the input string.
        """
        if not input_string.startswith("BLE:"):
            return []

        data = input_string[4:].strip()
        pattern = r"([0-9A-Fa-f]{12}):([+-]?\d+):([0-9A-Fa-f]{2,})"
        matches = re.findall(pattern, data)
        unix_ts = time.time()

        devices: List[RFDataRaw] = []

        for match in matches:
            address = match[0].upper()
            rssi = int(match[1])
            packet = match[2].lower()
            devices.append(
                RFDataRaw(unix_ts=unix_ts, address=address, rssi=rssi, packet=packet)
            )

        return devices

    def start(self):
        """
        Starts the GPS Provider and processing thread
        if not already running.
        """
        if self._thread and self._thread.is_alive():
            return

        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        """
        Main loop for the GPS provider.
        WARNING: this assumes that the data from the Arduino are arriving
        relatively slowly.
        """
        while self.running:
            if self.serial_connection:
                # Read a line, decode, and remove whitespace
                data = self.serial_connection.readline().decode("utf-8").strip()
                logging.debug(f"Serial GPS/MAG: {data}")
                self.magGPSProcessor(data)
            time.sleep(0.1)

    def stop(self):
        """
        Stop the GPS provider.
        """
        self.running = False
        if self._thread:
            logging.info("Stopping GPS provider")
            self._thread.join(timeout=5)

    @property
    def data(self) -> Optional[dict]:
        """
        Get the latest GPS data.

        Returns
        -------
        Optional[dict]
            The current GPS data dictionary or None if no data is available.
        """
        return self._gps
