# type: ignore
"""Simple and lightweight module for working with RPLidar rangefinder scanners.

Usage example:

>>> from rplidar import RPLidar
>>> lidar = RPLidar('/dev/ttyUSB0')
>>>
>>> info = lidar.get_info()
>>> print(info)
>>>
>>> health = lidar.get_health()
>>> print(health)
>>>
>>> for i, scan in enumerate(lidar.iter_scans()):
...  print('%d: Got %d measures' % (i, len(scan)))
...  if i > 10:
...   break
...
>>> lidar.stop()
>>> lidar.stop_motor()
>>> lidar.disconnect()

For additional information please refer to the RPLidar class documentation.
"""

import codecs
import logging
import struct
import sys
import time
from collections import namedtuple

import serial

# Protocol constants
SYNC_BYTE = b"\xa5"
SYNC_BYTE2 = b"\x5a"

# Command bytes
GET_INFO_BYTE = b"\x50"
GET_HEALTH_BYTE = b"\x52"
STOP_BYTE = b"\x25"
RESET_BYTE = b"\x40"
SET_PWM_BYTE = b"\xf0"

# Scan configuration
_SCAN_TYPE = {
    "normal": {"byte": b"\x20", "response": 129, "size": 5},
    "force": {"byte": b"\x21", "response": 129, "size": 5},
    "express": {"byte": b"\x82", "response": 130, "size": 84},
}

# Response lengths
DESCRIPTOR_LEN = 7
INFO_LEN = 20
HEALTH_LEN = 3

# Response types
INFO_TYPE = 4
HEALTH_TYPE = 6

# Motor control
MAX_MOTOR_PWM = 1023
DEFAULT_MOTOR_PWM = 400

# Health status mapping
_HEALTH_STATUSES = {
    0: "Good",
    1: "Warning",
    2: "Error",
}


class RPLidarException(Exception):
    """Basic exception class for RPLidar."""


def _b2i(byte):
    """Converts byte to integer (for Python 2 compatibility)."""
    return byte if int(sys.version[0]) == 3 else ord(byte)


def _showhex(signal):
    """Converts string bytes to hex representation (useful for debugging)."""
    return [format(_b2i(b), "#02x") for b in signal]


def _process_scan(raw):
    """Processes input raw data and returns measurement data."""
    new_scan = bool(_b2i(raw[0]) & 0b1)
    inversed_new_scan = bool((_b2i(raw[0]) >> 1) & 0b1)
    quality = _b2i(raw[0]) >> 2
    if new_scan == inversed_new_scan:
        raise RPLidarException("New scan flags mismatch")
    check_bit = _b2i(raw[1]) & 0b1
    if check_bit != 1:
        raise RPLidarException("Check bit not equal to 1")
    angle = ((_b2i(raw[1]) >> 1) + (_b2i(raw[2]) << 7)) / 64.0
    distance = (_b2i(raw[3]) + (_b2i(raw[4]) << 8)) / 4.0
    return new_scan, quality, angle, distance


def _process_express_scan(data, new_angle, trame):
    new_scan = (new_angle < data.start_angle) & (trame == 1)
    angle = (
        data.start_angle
        + ((new_angle - data.start_angle) % 360) / 32 * trame
        - data.angle[trame - 1]
    ) % 360
    distance = data.distance[trame - 1]
    # print(
    #     f"Express scan: new_scan={new_scan}, angle={angle}, "
    #     f"distance={distance}, trame={trame}"
    # )
    return new_scan, None, angle, distance


class RPDriver(object):
    """Class for communicating with RPLidar rangefinder scanners."""

    def __init__(self, port, baudrate=115200, timeout=1, logger=None):
        """Initialize RPLidar object for communicating with the sensor.

        Parameters
        ----------
        port : str
            Serial port name to which sensor is connected
        baudrate : int, optional
            Baudrate for serial connection (the default is 115200)
        timeout : float, optional
            Serial port connection timeout in seconds (the default is 1)
        logger : logging.Logger instance, optional
            Logger instance, if none is provided new instance is created
        """
        self._serial = None
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._motor_speed = DEFAULT_MOTOR_PWM
        self.scanning = [False, 0, "normal"]
        self.express_trame = 32
        self.express_data = False
        self.motor_running = None
        if logger is None:
            logger = logging.getLogger("rplidar")
        self.logger = logger
        self.connect()

    def connect(self):
        """Connects to the serial port with the name `self.port`. If it was
        connected to another serial port disconnects from it first.
        """
        if self._serial is not None:
            self.disconnect()
        try:
            self._serial = serial.Serial(
                self.port,
                self.baudrate,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=self.timeout,
                write_timeout=self.timeout,
            )
        except serial.SerialException as err:
            raise RPLidarException(
                "Failed to connect to the sensor " "due to: %s" % err
            )

    def disconnect(self):
        """Disconnects from the serial port."""
        if self._serial is None:
            return
        self._serial.close()

    def _set_pwm(self, pwm):
        payload = struct.pack("<H", pwm)
        self._send_payload_cmd(SET_PWM_BYTE, payload)

    @property
    def motor_speed(self):
        """Get current motor speed (PWM value)."""
        return self._motor_speed

    @motor_speed.setter
    def motor_speed(self, pwm: int):
        """
        Set motor speed (PWM value).

        Parameters
        ----------
        pwm : int
            PWM value to set the motor speed to.
        """
        if not (0 <= pwm <= MAX_MOTOR_PWM):
            raise ValueError(
                f"PWM value must be between 0 and {MAX_MOTOR_PWM}, got {pwm}"
            )
        self._motor_speed = pwm
        if self.motor_running:
            self._set_pwm(self._motor_speed)

    def start_motor(self):
        """Starts sensor motor."""
        self.logger.info("Starting motor")
        # For A1
        self._serial.setDTR(False)

        # For A2
        self._set_pwm(self._motor_speed)
        self.motor_running = True

    def stop_motor(self):
        """Stops sensor motor."""
        self.logger.info("Stopping motor")
        # For A2
        self._set_pwm(0)
        time.sleep(0.001)
        # For A1
        self._serial.setDTR(True)
        self.motor_running = False

    def _send_payload_cmd(self, cmd, payload):
        """Sends `cmd` command with `payload` to the sensor."""
        size = struct.pack("B", len(payload))
        req = SYNC_BYTE + cmd + size + payload
        checksum = 0
        for v in struct.unpack("B" * len(req), req):
            checksum ^= v
        req += struct.pack("B", checksum)
        self._serial.write(req)
        self.logger.debug("Command sent: %s" % _showhex(req))

    def _send_cmd(self, cmd):
        """Sends `cmd` command to the sensor."""
        req = SYNC_BYTE + cmd
        self._serial.write(req)
        self.logger.debug("Command sent: %s" % _showhex(req))

    def _read_descriptor(self):
        """Reads descriptor packet."""
        descriptor = self._serial.read(DESCRIPTOR_LEN)
        self.logger.debug("Received descriptor: %s", _showhex(descriptor))
        if len(descriptor) != DESCRIPTOR_LEN:
            raise RPLidarException("Descriptor length mismatch")
        elif not descriptor.startswith(SYNC_BYTE + SYNC_BYTE2):
            raise RPLidarException("Incorrect descriptor starting bytes")
        is_single = _b2i(descriptor[-2]) == 0
        return _b2i(descriptor[2]), is_single, _b2i(descriptor[-1])

    def _read_response(self, dsize):
        """Reads response packet with length of `dsize` bytes."""
        self.logger.debug("Trying to read response: %d bytes", dsize)
        while self._serial.inWaiting() < dsize:
            time.sleep(0.001)
        data = self._serial.read(dsize)
        self.logger.debug("Received data: %s", _showhex(data))
        return data

    def get_info(self):
        """Get device information.

        Returns
        -------
        dict
            Dictionary with the sensor information
        """
        if self._serial.inWaiting() > 0:
            self.clean_input()
            time.sleep(0.5)
            # return "Buffer is full! Run clean_input() to empty the buffer."
        self._send_cmd(GET_INFO_BYTE)
        dsize, is_single, dtype = self._read_descriptor()
        if dsize != INFO_LEN:
            raise RPLidarException("Wrong get_info reply length")
        if not is_single:
            raise RPLidarException("Not a single response mode")
        if dtype != INFO_TYPE:
            raise RPLidarException("Wrong response data type")
        raw = self._read_response(dsize)
        serialnumber = codecs.encode(raw[4:], "hex").upper()
        serialnumber = codecs.decode(serialnumber, "ascii")
        data = {
            "model": _b2i(raw[0]),
            "firmware": (_b2i(raw[2]), _b2i(raw[1])),
            "hardware": _b2i(raw[3]),
            "serial number": serialnumber,
        }
        return data

    def get_health(self):
        """Get device health state. When the core system detects some
        potential risk that may cause hardware failure in the future,
        the returned status value will be 'Warning'. But sensor can still work
        as normal. When sensor is in the Protection Stop state, the returned
        status value will be 'Error'. In case of warning or error statuses
        non-zero error code will be returned.

        Returns
        -------
        status : str
            'Good', 'Warning' or 'Error' statuses
        error_code : int
            The related error code that caused a warning/error.
        """
        if self._serial.inWaiting() > 0:
            self.clean_input()
            time.sleep(0.5)
            # return "Data in buffer. " "Run clean_input() to empty the buffer."
        self.logger.info("Asking for health")
        self._send_cmd(GET_HEALTH_BYTE)
        dsize, is_single, dtype = self._read_descriptor()
        if dsize != HEALTH_LEN:
            raise RPLidarException("Wrong get_info reply length")
        if not is_single:
            raise RPLidarException("Not a single response mode")
        if dtype != HEALTH_TYPE:
            raise RPLidarException("Wrong response data type")
        raw = self._read_response(dsize)
        status = _HEALTH_STATUSES[_b2i(raw[0])]
        error_code = (_b2i(raw[1]) << 8) + _b2i(raw[2])
        return status, error_code

    def clean_input(self):
        """Clean input buffer by reading all available data."""
        if self.scanning[0]:
            return "Cleaning not allowed during active scanning!"
        self._serial.flushInput()
        self.express_trame = 32
        self.express_data = False

    def stop(self):
        """Stops scanning process, disables laser diode and the measurement
        system, moves sensor to the idle state.
        """
        self.logger.info("Stop scanning")
        self._send_cmd(STOP_BYTE)
        time.sleep(0.1)
        self.scanning[0] = False
        self.clean_input()

    def start(self, scan_type="normal"):
        """Start the scanning process.

        Parameters
        ----------
        scan_type : str
            Scan mode: normal, force, or express.
        """
        if self.scanning[0]:
            return "Scan already running!"

        # Check sensor health before starting
        status, error_code = self.get_health()
        self.logger.debug("Health status: %s [%d]", status, error_code)

        if status == _HEALTH_STATUSES[2]:
            self.logger.warning(
                "Trying to reset sensor due to error. Error code: %d", error_code
            )
            self.reset()
            status, error_code = self.get_health()
            if status == _HEALTH_STATUSES[2]:
                raise RPLidarException(
                    "RPLidar hardware failure. Error code: %d" % error_code
                )
        elif status == _HEALTH_STATUSES[1]:
            self.logger.warning(
                "Warning sensor status detected! Error code: %d", error_code
            )

        cmd = _SCAN_TYPE[scan_type]["byte"]
        self.logger.info("Starting scan process in %s mode", scan_type)

        if scan_type == "express":
            self._send_payload_cmd(cmd, b"\x00\x00\x00\x00\x00")
        else:
            self._send_cmd(cmd)

        dsize, is_single, dtype = self._read_descriptor()
        if dsize != _SCAN_TYPE[scan_type]["size"]:
            raise RPLidarException("Wrong get_info reply length")
        if is_single:
            raise RPLidarException("Not a multiple response mode")
        if dtype != _SCAN_TYPE[scan_type]["response"]:
            raise RPLidarException("Wrong response data type")
        self.scanning = [True, dsize, scan_type]

    def reset(self):
        """Resets sensor core, reverting it to a similar state as it has
        just been powered up.
        """
        self.logger.info("Resetting the RPLidar")
        self._send_cmd(RESET_BYTE)
        time.sleep(2)
        self.clean_input()

    def iter_measures(self, scan_type="normal", max_buf_meas=3000):
        """Iterate over measurements. Note that consumer must be fast enough,
        otherwise data will accumulate inside buffer and consumer will get
        data with increasing lag.

        Parameters
        ----------
        max_buf_meas : int or False if you want unlimited buffer
            Maximum number of bytes to be stored inside the buffer. Once
            number exceeds this limit buffer will be emptied out.

        Yields
        ------
        new_scan : bool
            True if measurement belongs to a new scan
        quality : int
            Reflected laser pulse strength
        angle : float
            The measurement heading angle in degree units [0, 360)
        distance : float
            Measured object distance related to the sensor's rotation center.
            In millimeters. Set to 0 when measure is invalid.
        """
        self.start_motor()
        if not self.scanning[0]:
            self.start(scan_type)

        while True:
            dsize = self.scanning[1]

            if max_buf_meas:
                data_in_buf = self._serial.inWaiting()
                if data_in_buf > max_buf_meas:
                    self.logger.warning(
                        "Too many bytes in the input buffer: %d/%d. "
                        "Cleaning buffer...",
                        data_in_buf,
                        max_buf_meas,
                    )
                    self.stop()
                    self.start(self.scanning[2])

            if self.scanning[2] == "normal":
                self.logger.debug("Normal scanning mode")
                raw = self._read_response(dsize)
                self.logger.debug(f"Raw data: {raw}")
                yield _process_scan(raw)
            if self.scanning[2] == "express":
                try:
                    # print("M: Express scanning")
                    if self.express_trame == 32:
                        self.express_trame = 0
                        # print("self.express_data: ",  self.express_data)
                        if not self.express_data:
                            self.logger.debug("reading first time bytes")
                            raw_data = self._read_response(dsize)
                            self.express_data = ExpressPacket.from_string(raw_data)

                        self.express_old_data = self.express_data
                        self.logger.debug(
                            "set old_data with start_angle %f",
                            self.express_old_data.start_angle,
                        )
                        raw_data = self._read_response(dsize)
                        self.express_data = ExpressPacket.from_string(raw_data)
                        self.logger.debug(
                            "set new_data with start_angle %f",
                            self.express_data.start_angle,
                        )

                    self.express_trame += 1
                    self.logger.debug(
                        "process scan of frame %d with angle : %f and angle new : %f",
                        self.express_trame,
                        self.express_old_data.start_angle,
                        self.express_data.start_angle,
                    )
                    # print(
                    #     "process scan of frame %d with angle : %f and angle new : %f",
                    #     self.express_trame,
                    #     self.express_old_data.start_angle,
                    #     self.express_data.start_angle,
                    # )
                    yield _process_express_scan(
                        self.express_old_data,
                        self.express_data.start_angle,
                        self.express_trame,
                    )

                except ValueError as e:
                    self.logger.warning("Error while processing express scan: %s", e)
                    self.express_trame = 32
                    self.express_data = False

                    self.stop()
                    time.sleep(0.1)
                    self.clean_input()
                    time.sleep(0.1)
                    self.start("express")
                    continue

    def iter_scans(self, scan_type="normal", max_buf_meas=3000, min_len=5):
        """Iterate over scans. Note that consumer must be fast enough,
        otherwise data will be accumulated inside buffer and consumer will get
        data with increasing lag.

        Parameters
        ----------
        max_buf_meas : int
            Maximum number of measures to be stored inside the buffer. Once
            number exceeds this limit buffer will be cleared.
        min_len : int
            Minimum number of measures in the scan for it to be returned.

        Yields
        ------
        scan : list
            List of the measurements. Each measurement is a tuple with following
            format: (quality, angle, distance). For values description please
            refer to `iter_measures` method's documentation.
        """
        scan_list = []
        iterator = self.iter_measures(scan_type, max_buf_meas)
        for new_scan, quality, angle, distance in iterator:
            if new_scan:
                if len(scan_list) > min_len:
                    yield scan_list
                scan_list = []
            if distance > 0:
                scan_list.append((quality, angle, distance))

    def iter_scans_local(
        self, scan_type="normal", max_buf_meas=3000, min_len=5, max_distance_mm=2000
    ):
        """Iterate over scans. Note that consumer must be fast enough,
        otherwise data will accumulate inside buffer and consumer will get
        data with increasing lag.

        Parameters
        ----------
        max_buf_meas : int
            Maximum number of measures to be stored inside the buffer. Once
            number exceeds this limit buffer will be cleared.
        min_len : int
            Minimum number of measures in the scan for it to be returned.

        Yields
        ------
        scan : list
            List of the measurements. Each measurement is a tuple with following
            format: (angle, distance). For values description please
            refer to `iter_measures` method's documentation.
        """
        scan_list = []
        iterator = self.iter_measures(scan_type, max_buf_meas)
        for new_scan, quality, angle, distance in iterator:
            if new_scan:
                if len(scan_list) > min_len:
                    yield scan_list
                scan_list = []
            if distance > 0 and distance < max_distance_mm:
                scan_list.append((angle, distance))


class ExpressPacket(
    namedtuple("express_packet", "distance angle new_scan start_angle")
):
    """
    Class representing an express scan packet from RPLidar.
    """

    sync1 = 0xA
    sync2 = 0x5
    sign = {0: 1, 1: -1}

    @classmethod
    def from_string(cls, data):
        """
        Create ExpressPacket from raw data string.

        Parameters
        ----------
        data : bytes
            Raw data bytes to parse.
        """
        packet = bytearray(data)

        if (packet[0] >> 4) != cls.sync1 or (packet[1] >> 4) != cls.sync2:
            raise ValueError("trying to parse corrupted data ({})".format(packet))

        checksum = 0
        for b in packet[2:]:
            checksum ^= b
        if checksum != (packet[0] & 0b00001111) + ((packet[1] & 0b00001111) << 4):
            raise ValueError("Invalid checksum ({})".format(packet))

        new_scan = packet[3] >> 7
        start_angle = (packet[2] + ((packet[3] & 0b01111111) << 8)) / 64

        d = a = ()
        for i in range(0, 80, 5):
            d += ((packet[i + 4] >> 2) + (packet[i + 5] << 6),)
            a += (
                ((packet[i + 8] & 0b00001111) + ((packet[i + 4] & 0b00000001) << 4))
                / 8
                * cls.sign[(packet[i + 4] & 0b00000010) >> 1],
            )
            d += ((packet[i + 6] >> 2) + (packet[i + 7] << 6),)
            a += (
                ((packet[i + 8] >> 4) + ((packet[i + 6] & 0b00000001) << 4))
                / 8
                * cls.sign[(packet[i + 6] & 0b00000010) >> 1],
            )
        return cls(d, a, new_scan, start_angle)
