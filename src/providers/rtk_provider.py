import datetime
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
    RTK (Real-Time Kinematic) Provider for high-precision GPS positioning.

    This class implements a singleton pattern to manage RTK GPS data acquisition
    from serial communication. RTK technology provides centimeter-level
    positioning accuracy by using carrier-phase measurements and correction data
    from a base station or network.

    The provider continuously reads NMEA messages from a serial-connected RTK GPS
    receiver, extracts GNGGA (Global Navigation Satellite System Fix Data) messages,
    and processes them to extract position, altitude, satellite count, and fix quality
    information. The processed data is made available through the `data` property.

    Attributes
    ----------
    lat : float
        Current latitude in decimal degrees (rounded to 7 decimal places).
    lon : float
        Current longitude in decimal degrees (rounded to 7 decimal places).
    alt : float
        Current altitude in meters (rounded to 2 decimal places).
    sat : int
        Number of satellites used in the position fix.
    qua : int
        Fix quality indicator (0=invalid, 1=GPS fix, 2=DGPS fix, etc.).
    unix_ts : float
        Unix timestamp of the most recent position update.
    _rtk : Optional[dict]
        Dictionary containing the latest RTK position data.
    """

    def __init__(self, serial_port: str = ""):
        """
        Initialize the RTK Provider with serial connection configuration.

        Parameters
        ----------
        serial_port : str, optional
            The serial port path for the RTK GPS receiver connection.
            Examples: Linux: "/dev/ttyUSB0", Windows: "COM3".
            If empty string, serial connection will not be established.

        Notes
        -----
        The provider automatically starts a background processing thread upon
        initialization. The thread continuously reads NMEA data from the serial
        connection and processes GNGGA messages to update position data.

        Serial connection parameters:
        - Baud rate: 115200
        - Timeout: 0.2 seconds
        - Input buffer is reset upon connection

        If serial connection fails, the provider will log an error but continue
        to run, allowing for graceful degradation when hardware is unavailable.
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
        Convert a UTC datetime.time object to a Unix timestamp.

        Combines the provided UTC time with the local computer's current date
        to create a complete datetime object, then converts it to a Unix timestamp.

        Parameters
        ----------
        utc_time_obj : datetime.time
            The UTC time object to convert.

        Returns
        -------
        float
            Unix timestamp (seconds since epoch).

        Raises
        ------
        TypeError
            If the input is not a datetime.time object.
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

        Searches for all GNGGA (Global Navigation Satellite System Fix Data)
        messages in the input string, identifies the most recent one based on
        the embedded time field, and returns the complete message string.

        Parameters
        ----------
        nmea_data : str
            The block of NMEA data as a string, potentially containing multiple
            GNGGA messages.

        Returns
        -------
        str or None
            The most recent GNGGA message string if found, None otherwise.
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
        Process incoming RTK NMEA messages and update position data.

        Parses GNGGA (Global Navigation Satellite System Fix Data) messages
        to extract position, altitude, satellite count, and fix quality information.
        The extracted data is rounded to appropriate precision and stored in instance
        attributes. Position coordinates are rounded to 7 decimal places (approximately
        1 cm precision), altitude to 2 decimal places (1 cm precision).

        Parameters
        ----------
        msg : NMEA message object
            The parsed NMEA message object from pynmeagps.NMEAReader.
            Expected message type: GNGGA (Global positioning system fix data).

        Notes
        -----
        Only GNGGA messages are processed. Other message types are ignored.
        If message parsing fails, a warning is logged but processing continues.
        The processed data is stored in the `_rtk` dictionary and can be
        accessed via the `data` property.
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
        Start the RTK Provider background processing thread.

        Initializes and starts a daemon thread that continuously reads NMEA data
        from the serial connection and processes GNGGA messages. If a thread is
        already running, this method returns without creating a new thread.

        Notes
        -----
        The background thread runs the `_run` method, which polls the serial
        connection every 0.1 seconds for incoming data. The thread is marked
        as a daemon thread, so it will automatically terminate when the main
        program exits.
        """
        if self._thread and self._thread.is_alive():
            return

        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        """
        Main processing loop for the RTK provider background thread.

        Continuously reads NMEA data from the serial connection, extracts
        GNGGA messages, parses them, and updates position data. This method
        runs in a separate daemon thread and should not be called directly.

        Notes
        -----
        The loop checks for available data in the serial input buffer every
        0.1 seconds. When data is available, it is decoded as UTF-8 (with
        error handling for invalid characters), the latest GNGGA message is
        extracted, parsed using NMEAReader, and processed by magRTKProcessor.
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
        Stop the RTK provider and terminate the background processing thread.

        Sets the running flag to False and waits for the background thread to
        terminate. The thread join operation has a 5-second timeout to prevent
        indefinite blocking.

        Notes
        -----
        After calling this method, the provider will no longer process incoming
        NMEA data. The serial connection remains open but data processing stops.
        If the thread does not terminate within 5 seconds, the method returns
        without further waiting.
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
