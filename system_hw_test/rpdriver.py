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

SYNC_BYTE = b"\xA5"
SYNC_BYTE2 = b"\x5A"

GET_INFO_BYTE = b"\x50"
GET_HEALTH_BYTE = b"\x52"

STOP_BYTE = b"\x25"
RESET_BYTE = b"\x40"

_SCAN_TYPE = {
    "normal": {"byte": b"\x20", "response": 129, "size": 5},
    "force": {"byte": b"\x21", "response": 129, "size": 5},
    "express": {"byte": b"\x82", "response": 130, "size": 84},
}

DESCRIPTOR_LEN = 7
INFO_LEN = 20
HEALTH_LEN = 3

INFO_TYPE = 4
HEALTH_TYPE = 6

# Constants & Command to start A2 motor
MAX_MOTOR_PWM = 1023
DEFAULT_MOTOR_PWM = 660
SET_PWM_BYTE = b"\xF0"
AUTO_START_MOTOR    = True   # <--- set False if your host/thread manages motor

_HEALTH_STATUSES = {0: "Good", 1: "Warning", 2: "Error"}

class RPLidarException(Exception):
    """Basic exception class for RPLidar"""
    pass

def _b2i(byte):
    """Converts byte to integer (for Python 2 compatibility)"""
    return byte if int(sys.version[0]) == 3 else ord(byte)

def _showhex(signal):
    """Converts string bytes to hex representation (useful for debugging)"""
    return [format(_b2i(b), "#02x") for b in signal]

def _process_scan(raw):
    """Processes input raw data and returns measurement data"""
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

class RPDriver(object):
    """
    Robust serial driver for RPLIDAR (normal mode).

    Key behavior:
    - Uses descriptor re-sync (hunts for A5 5A).
    - Applies hard timeouts and short-read checks.
    - Drains/flushes residual bytes before starting a stream.
    - Optionally starts motor automatically when starting a scan.
    """

    def __init__(self, port, baudrate=115200, timeout=0.2, logger=None):
        self._serial = None
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._motor_speed = DEFAULT_MOTOR_PWM
        self.scanning = [False, 0, "normal"]
        self.motor_running = None
        if logger is None:
            logger = logging.getLogger("rplidar")
        self.logger = logger
        self.connect()
    
    def _drain_for(self, duration_s=0.06):
        """Eat any residual bytes for a short window to ensure clean state."""
        end = time.time() + duration_s
        while time.time() < end:
            try:
                n = self._serial.in_waiting
            except AttributeError:
                n = self._serial.inWaiting()
            if n:
                self._serial.read(n)
            time.sleep(0.003)

    # ---- Serial lifecycle ----------------------------------------------------
    def connect(self):
        if self._serial is not None:
            self.disconnect()
        try:
            self._serial = serial.Serial(
                self.port, self.baudrate,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=self.timeout,
            )
            print(f"[diag][connect] open={self._serial.is_open} port={self.port}")
            # Keep lines deasserted; some USB-serials care
            self._serial.setDTR(False)
            try: self._serial.setRTS(False)
            except Exception: pass
        except serial.SerialException as err:
            raise RPLidarException(f"Failed to connect to the sensor: {err}")

    def disconnect(self):
        if self._serial is None:
            return
        self._serial.close()

    # ---- Motor control -------------------------------------------------------
    def _set_pwm(self, pwm):
        payload = struct.pack("<H", pwm)
        self._send_payload_cmd(SET_PWM_BYTE, payload)

    @property
    def motor_speed(self):
        return self._motor_speed

    @motor_speed.setter
    def motor_speed(self, pwm):
        assert 0 <= pwm <= MAX_MOTOR_PWM
        self._motor_speed = pwm
        if self.motor_running:
            self._set_pwm(self._motor_speed)

    def start_motor(self):
        """Spin the motor:
           - A1: DTR low
           - A2: PWM set to DEFAULT_MOTOR_PWM"""
        self.logger.info("Starting motor")
        # A1: DTR low; A2: PWM
        self._serial.setDTR(False)
        self._set_pwm(self._motor_speed)
        self.motor_running = True
        print(f"[diag][start_motor] pwm={self._motor_speed}")

    def stop_motor(self):
        """Stop the motor and place lines in idle state."""
        self.logger.info("Stopping motor")
        self._set_pwm(0)
        time.sleep(0.001)
        self._serial.setDTR(True)
        self.motor_running = False

    def _send_payload_cmd(self, cmd, payload):
        size = struct.pack("B", len(payload))
        req = SYNC_BYTE + cmd + size + payload
        checksum = 0
        for v in struct.unpack("B" * len(req), req):
            checksum ^= v
        req += struct.pack("B", checksum)
        self._serial.write(req)
        self.logger.debug("Command sent: %s" % _showhex(req))

    def _send_cmd(self, cmd):
        req = SYNC_BYTE + cmd
        self._serial.write(req)
        self.logger.debug("Command sent: %s" % _showhex(req))

    def _read_descriptor(self, deadline_s=1.0):
        """
        Robustly read the 7-byte descriptor:
        - Hunt for the A5 5A sync header
        - If a short-read happens, keep hunting until deadline
        """
        t_end = time.time() + deadline_s
        header = bytearray()
        # hunt for SYNC bytes
        while time.time() < t_end:
            b = self._serial.read(1)
            if not b:
                continue
            header += b
            # keep only last 2 bytes to match against A5 5A
            if len(header) > 2:
                header = header[-2:]
            if header == SYNC_BYTE + SYNC_BYTE2:
                rest = self._serial.read(DESCRIPTOR_LEN - 2)
                if len(rest) != DESCRIPTOR_LEN - 2:
                    # try again if short
                    header.clear()
                    continue
                descriptor = (SYNC_BYTE + SYNC_BYTE2 + rest)
                self.logger.debug("Received descriptor: %s", _showhex(descriptor))
                is_single = _b2i(descriptor[-2]) == 0
                return _b2i(descriptor[2]), is_single, _b2i(descriptor[-1])
        raise RPLidarException("Descriptor sync timeout")

    def _read_response(self, dsize, timeout_s=2.0):
        """
        Wait for at least dsize bytes with a hard timeout, then read exactly dsize.
        Raises if a timeout or short-read occurs.
        """
        self.logger.debug("Trying to read response: %d bytes", dsize)
        dsize = int(dsize)

        def _bytes_waiting():
            try:
                return self._serial.in_waiting
            except AttributeError:
                return self._serial.inWaiting()

        start = time.time()
        while True:
            if _bytes_waiting() >= dsize:
                break
            if time.time() - start > timeout_s:
                raise RPLidarException(f"Timeout waiting for {dsize} bytes; have {_bytes_waiting()}.")
            time.sleep(0.001)

        data = self._serial.read(dsize)
        if len(data) != dsize:
            raise RPLidarException(f"Short read: expected {dsize} bytes, got {len(data)}.")
        self.logger.debug("Received data: %s", _showhex(data))
        return data

    def get_info(self):
        """Read device info block (model/firmware/hardware/serial)."""
        if self._serial.inWaiting() > 0:
            return "Buffer is full! Run clean_input() to empty the buffer."
        self._send_cmd(GET_INFO_BYTE)
        dsize, is_single, dtype = self._read_descriptor()
        if dsize != INFO_LEN or not is_single or dtype != INFO_TYPE:
            raise RPLidarException("Bad get_info response")
        raw = self._read_response(dsize)
        serialnumber = codecs.encode(raw[4:], "hex").upper()
        serialnumber = codecs.decode(serialnumber, "ascii")
        return {
            "model": _b2i(raw[0]),
            "firmware": (_b2i(raw[2]), _b2i(raw[1])),
            "hardware": _b2i(raw[3]),
            "serial number": serialnumber,
        }

    def get_health(self):
        """Read device health; returns (status_str, error_code)."""
        if self._serial.inWaiting() > 0:
            return "Data in buffer. Run clean_input() to empty the buffer."
        self.logger.info("Asking for health")
        self._send_cmd(GET_HEALTH_BYTE)
        dsize, is_single, dtype = self._read_descriptor()
        if dsize != HEALTH_LEN or not is_single or dtype != HEALTH_TYPE:
            raise RPLidarException("Bad get_health response")
        raw = self._read_response(dsize)
        status = _HEALTH_STATUSES[_b2i(raw[0])]
        error_code = (_b2i(raw[1]) << 8) + _b2i(raw[2])
        return status, error_code

    def clean_input(self):
        """Flush input/output buffers (safe even if stream is idle)."""
        try:
            self._serial.reset_input_buffer()
            self._serial.reset_output_buffer()
        except AttributeError:
            self._serial.flushInput()
            self._serial.flushOutput()

    def stop(self):
        """Stops scanning process, disables laser diode and the measurement
        system, moves sensor to the idle state."""
        self.logger.info("Stop scanning")
        self._send_cmd(STOP_BYTE)
        time.sleep(0.1)
        self.scanning[0] = False
        self.clean_input()

    def start(self, scan_type="normal"):
        """
        Start a normal-mode scan:
        - optional auto motor start (AUTO_START_MOTOR)
        - health check and auto-reset on 'Error'
        - STOP+flush+drain to guarantee a clean start
        - descriptor re-sync with one automatic retry
        """
        if self.scanning[0]:
            return "Scan already running!"
        
        # (1) Ensure motor is spinning if requested (prevents 'no bytes' state)
        if AUTO_START_MOTOR and not self.motor_running:
            self.start_motor()
            time.sleep(0.4)  # let RPM stabilize
        

        # (2) Check health and auto-reset on 'Error'
        try:
            status, error_code = self.get_health()
            self.logger.debug("Health status: %s [%d]", status, error_code)
            if status == _HEALTH_STATUSES[2]:  # "Error"
                self.logger.warning("Resetting due to health error: %d", error_code)
                self.reset()
                status, error_code = self.get_health()
                if status == _HEALTH_STATUSES[2]:
                    raise RPLidarException(f"RPLidar hardware failure. Error: {error_code}")
            elif status == _HEALTH_STATUSES[1]:  # "Warning"
                self.logger.warning("Sensor reported WARNING. Error: %d", error_code)
        except Exception as e:
            # If health query itself failed (e.g., empty buffer), we still try to start
            self.logger.warning("Health check failed (continuing anyway): %s", e)

        # Stop any old stream, flush, then drain a bit to guarantee silence
        try:
            self._send_cmd(STOP_BYTE)
            time.sleep(0.02)
        except Exception:
            pass
        self.clean_input()
        self._drain_for(0.06)

        # Start normal mode (we force normal for reliability)
        print("Starting scan in normal mode")
        self._send_cmd(_SCAN_TYPE["normal"]["byte"])

        # Read descriptor; if it fails once, flush & try once more
        try:
            dsize, is_single, dtype = self._read_descriptor()
        except Exception:
            self.clean_input()
            self._drain_for(0.06)
            dsize, is_single, dtype = self._read_descriptor()

        if dsize != _SCAN_TYPE["normal"]["size"] or is_single or dtype != _SCAN_TYPE["normal"]["response"]:
            raise RPLidarException("Bad scan start response (normal)")

        self.scanning = [True, dsize, "normal"]


    def reset(self):
        """Reset the device core; wait, then flush inputs."""
        self.logger.info("Resetting the sensor")
        self._send_cmd(RESET_BYTE)
        time.sleep(2)
        self.clean_input()

    def iter_measures(self, scan_type="normal", max_buf_meas=3000):
        # Caller manages DTR/motor; just ensure stream is started.
        if not self.scanning[0]:
            self.start(scan_type)
            time.sleep(0.2)

        dsize = self.scanning[1]
        misses = 0
        while True:
            if max_buf_meas:
                try:
                    data_in_buf = self._serial.in_waiting
                except AttributeError:
                    data_in_buf = self._serial.inWaiting()
                if data_in_buf > max_buf_meas:
                    self.logger.warning(
                        "Too many bytes in input buffer: %d/%d. Cleaning...",
                        data_in_buf, max_buf_meas
                    )
                    self.stop()
                    self.start("normal")
                    time.sleep(0.1)
                    dsize = self.scanning[1]
                    misses = 0
                    continue

            try:
                raw = self._read_response(dsize, timeout_s=2.5)
                misses = 0
                yield _process_scan(raw)
            except RPLidarException as e:
                # transient underflow / short read: skip a few before restart
                if "Timeout waiting" in str(e) or "Short read" in str(e):
                    misses += 1
                    if misses <= 5:     # tolerate ~100–200ms of hiccups
                        time.sleep(0.01)
                        continue
                # anything else or too many misses: bubble up
                raise

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
        for new_scan, quality, angle, distance in self.iter_measures("normal", max_buf_meas):
            if new_scan:
                if len(scan_list) > min_len:
                    yield scan_list
                scan_list = []
            if distance > 0:
                scan_list.append((quality, angle, distance))
