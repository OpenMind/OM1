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
    GPS provider handling GPS, MAG, and BLE data from a serial connection.
    """

    def __init__(self, serial_port: str = ""):
        """
        Initialize the GPS provider and start the reader thread.
        """
        logging.info(f"GPS_Provider booting GPS Provider at serial: {serial_port}")

        baudrate = 115200
        timeout = 1

        self.serial_connection = None
        try:
            self.serial_connection = serial.Serial(
                serial_port,
                baudrate,
                timeout=timeout,
            )
            self.serial_connection.reset_input_buffer()
            logging.info(f"Connected to {serial_port} at {baudrate} baud")
        except serial.SerialException as e:
            logging.error(f"GPS serial connection error: {e}")

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

    #
    # ---------- UTIL ----------
    #

    def string_to_unix_timestamp(self, time_str: str) -> float:
        """
        Convert a GPS time string to a Unix timestamp.
        """
        try:
            dt = datetime.strptime(time_str, "%Y:%m:%d:%H:%M:%S:%f")
            return dt.replace(tzinfo=timezone.utc).timestamp()
        except Exception as e:
            logging.warning(f"Invalid GPS time format '{time_str}' ({e})")
            return 0.0

    def compass_heading_to_direction(self, degrees: float) -> str:
        """
        Convert a compass heading in degrees to a cardinal direction.
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

    #
    # ---------- BLE PARSER ----------
    #

    def parse_ble_triang_string(self, input_string: str) -> List[RFDataRaw]:
        """
        Parse BLE triangulation data from a serial string.
        """
        if not input_string.startswith("BLE:"):
            return []

        data = input_string[4:].strip()
        pattern = r"([0-9A-Fa-f]{12}):([+-]?\d+):([0-9A-Fa-f]{2,})"
        matches = re.findall(pattern, data)
        unix_ts = time.time()

        devices: List[RFDataRaw] = []

        for addr, rssi, packet in matches:
            try:
                devices.append(
                    RFDataRaw(
                        unix_ts=unix_ts,
                        address=addr.upper(),
                        rssi=int(rssi),
                        packet=packet.lower(),
                    )
                )
            except Exception as e:
                logging.warning(f"Failed to parse BLE device '{addr}' ({e})")

        return devices

    #
    # ---------- CORE SERIAL PROCESSOR ----------
    #

    def magGPSProcessor(self, data: str):
        """
        Process MAG, GPS, and BLE packets from the serial stream.
        """
        try:
            if data.startswith("HDG:"):
                parts = data.split(":")
                if len(parts) >= 2:
                    try:
                        self.yaw_mag_0_360 = float(parts[1])
                        self.yaw_mag_cardinal = self.compass_heading_to_direction(
                            self.yaw_mag_0_360
                        )
                    except ValueError:
                        logging.warning(f"Invalid HDG value: {parts[1]}")

            elif data.startswith("YPR:"):
                try:
                    yaw, pitch, roll = map(str.strip, data[4:].split(","))
                    logging.debug(f"Orientation → Yaw:{yaw} Pitch:{pitch} Roll:{roll}")
                except Exception:
                    logging.warning(f"Invalid YPR packet: {data}")

            elif data.startswith("GPS:"):
                parts = data[4:].split(",")

                if len(parts) < 7:
                    logging.warning(f"Incomplete GPS packet: {data}")
                    return

                lat_str = parts[0]
                lon_str = parts[1]

                alt = parts[4].split(":")[-1]
                sat = parts[5].split(":")[-1]
                time_raw = parts[6][5:]

                self.gps_unix_ts = self.string_to_unix_timestamp("20" + time_raw)

                if lat_str.endswith("N"):
                    self.lat = float(lat_str.replace("N", ""))
                elif lat_str.endswith("S"):
                    self.lat = -float(lat_str.replace("S", ""))

                if lon_str.endswith("E"):
                    self.lon = float(lon_str.replace("E", ""))
                elif lon_str.endswith("W"):
                    self.lon = -float(lon_str.replace("W", ""))

                self.lat = round(self.lat, 6)
                self.lon = round(self.lon, 6)
                self.alt = round(float(alt), 2)

                self.sat = int(sat)

            elif data.startswith("BLE:"):
                self.ble_scan = self.parse_ble_triang_string(data)

        except Exception as e:
            logging.warning(f"Error processing GPS data '{data}' ({e})")

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

    #
    # ---------- THREAD LOOP ----------
    #

    def start(self):
        """Start the GPS reader thread."""
        if self._thread and self._thread.is_alive():
            return

        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        """Continuously read and process serial data."""
        while self.running:
            if self.serial_connection:
                try:
                    data = (
                        self.serial_connection.readline()
                        .decode("utf-8", errors="ignore")
                        .strip()
                    )
                    if data:
                        self.magGPSProcessor(data)
                except Exception as e:
                    logging.warning(f"Serial read error ({e})")

            time.sleep(0.1)

    def stop(self):
        """Stop the GPS provider thread."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)

    @property
    def data(self) -> Optional[dict]:
        """Return the latest GPS data snapshot."""
        return self._gps
