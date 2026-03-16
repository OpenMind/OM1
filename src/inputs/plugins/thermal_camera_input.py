import asyncio
import logging
import time
from typing import Optional

from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider

try:
    import adafruit_mlx90640 as _mlx90640_lib

    _MLX90640_AVAILABLE = True
except ImportError:
    _mlx90640_lib = None
    _MLX90640_AVAILABLE = False
    logging.warning(
        "adafruit-mlx90640 not found. MLX90640 connector unavailable. "
        "Install with: pip install adafruit-circuitpython-mlx90640"
    )

try:
    import adafruit_amg88xx as _amg88xx_lib

    _AMG8833_AVAILABLE = True
except ImportError:
    _amg88xx_lib = None
    _AMG8833_AVAILABLE = False
    logging.warning(
        "adafruit-amg88xx not found. AMG8833 connector unavailable. "
        "Install with: pip install adafruit-circuitpython-amg88xx"
    )

try:
    import cv2 as _cv2

    _CV2_AVAILABLE = True
except ImportError:
    _cv2 = None
    _CV2_AVAILABLE = False
    logging.warning("opencv not found. USB thermal connector unavailable.")

try:
    import serial as _serial

    _SERIAL_AVAILABLE = True
except ImportError:
    _serial = None
    _SERIAL_AVAILABLE = False
    logging.warning(
        "pyserial not found. Serial thermal connector unavailable. "
        "Install with: pip install pyserial"
    )

# Temperature thresholds
HUMAN_TEMP_MIN = 34.0
HUMAN_TEMP_MAX = 39.0
ALERT_TEMP_THRESHOLD = 60.0


class ThermalCameraConfig(SensorConfig):
    """
    Configuration for Thermal Camera input plugin.

    Supports five hardware backends via the ``connector`` field:

    - ``"mlx90640"`` : Melexis MLX90640 32x24 IR array via I2C.
    - ``"amg8833"``  : Panasonic AMG8833 8x8 IR array via I2C.
    - ``"usb"``      : USB thermal camera via OpenCV (V4L2).
    - ``"serial"``   : Arduino with IR sensor via pyserial.
    - ``"mock"``     : Simulated thermal data for development (default).

    Parameters
    ----------
    connector : str
        Hardware backend to use. Default is ``"mock"``.
    camera_index : int
        Camera index for USB connector. Default 0.
    port : str
        Serial port for serial connector. Default ``"/dev/ttyUSB0"``.
    baudrate : int
        Serial baudrate. Default 9600.
    serial_timeout : float
        Serial read timeout in seconds. Default 1.0.
    i2c_address : int
        I2C address for MLX90640. Default 0x33.
    refresh_rate : int
        MLX90640 refresh rate in Hz (1,2,4,8,16,32,64). Default 4.
    cooldown : float
        Minimum seconds between alerts forwarded to LLM. Default 3.0.
    human_temp_min : float
        Minimum temperature (°C) to classify as human. Default 34.0.
    human_temp_max : float
        Maximum temperature (°C) to classify as human. Default 39.0.
    alert_temp_threshold : float
        Temperature (°C) above which fire/equipment alert is triggered. Default 60.0.
    mock_scenario : str
        Mock scenario: ``"clear"``, ``"human"``, or ``"alert"``. Default ``"human"``.
    """

    connector: str = Field(
        default="mock",
        description=(
            "Hardware backend: 'mlx90640' (I2C 32x24), "
            "'amg8833' (I2C 8x8), "
            "'usb' (OpenCV), "
            "'serial' (Arduino), "
            "'mock' (testing)"
        ),
    )
    camera_index: int = Field(
        default=0,
        description="Camera index for USB thermal connector.",
    )
    port: str = Field(
        default="/dev/ttyUSB0",
        description="Serial port path. Used when connector='serial'.",
    )
    baudrate: int = Field(
        default=9600,
        description="Serial baudrate. Used when connector='serial'.",
    )
    serial_timeout: float = Field(
        default=1.0,
        description="Serial read timeout in seconds.",
    )
    i2c_address: int = Field(
        default=0x33,
        description="I2C address for MLX90640. Default 0x33.",
    )
    refresh_rate: int = Field(
        default=4,
        description="MLX90640 refresh rate in Hz.",
    )
    cooldown: float = Field(
        default=3.0,
        description="Minimum seconds between alerts forwarded to LLM.",
    )
    human_temp_min: float = Field(
        default=HUMAN_TEMP_MIN,
        description="Minimum temperature (°C) to classify as human presence.",
    )
    human_temp_max: float = Field(
        default=HUMAN_TEMP_MAX,
        description="Maximum temperature (°C) to classify as human presence.",
    )
    alert_temp_threshold: float = Field(
        default=ALERT_TEMP_THRESHOLD,
        description="Temperature (°C) above which fire/equipment alert triggers.",
    )
    mock_scenario: str = Field(
        default="human",
        description="Mock scenario: 'clear', 'human', or 'alert'.",
    )


class ThermalReading:
    """
    Container for a thermal sensor reading.

    Parameters
    ----------
    frame : list[float]
        Flat list of temperature values in Celsius.
    width : int
        Width of the thermal image in pixels.
    height : int
        Height of the thermal image in pixels.
    """

    def __init__(self, frame: list, width: int, height: int):
        """
        Initialize ThermalReading.

        Parameters
        ----------
        frame : list[float]
            Flat list of temperature values.
        width : int
            Image width in pixels.
        height : int
            Image height in pixels.
        """
        self.frame = frame
        self.width = width
        self.height = height

    @property
    def max_temp(self) -> float:
        """Get maximum temperature in the frame."""
        return max(self.frame) if self.frame else 0.0

    @property
    def min_temp(self) -> float:
        """Get minimum temperature in the frame."""
        return min(self.frame) if self.frame else 0.0

    def get_zone_max(self, zone: str) -> float:
        """
        Get maximum temperature in a horizontal zone.

        Parameters
        ----------
        zone : str
            Zone name: 'left', 'center', or 'right'.

        Returns
        -------
        float
            Maximum temperature in the specified zone.
        """
        third = self.width // 3
        pixels = []
        for row in range(self.height):
            for col in range(self.width):
                idx = row * self.width + col
                if idx >= len(self.frame):
                    continue
                if zone == "left" and col < third:
                    pixels.append(self.frame[idx])
                elif zone == "center" and third <= col < 2 * third:
                    pixels.append(self.frame[idx])
                elif zone == "right" and col >= 2 * third:
                    pixels.append(self.frame[idx])
        return max(pixels) if pixels else 0.0


class _MLX90640Connector:
    """
    Read thermal data from Melexis MLX90640 32x24 IR array via I2C.

    Wiring (MLX90640 to Raspberry Pi)
    -----------------------------------
    MLX90640 VIN  →  RPi Pin 1  (3.3V)
    MLX90640 GND  →  RPi Pin 6  (GND)
    MLX90640 SDA  →  RPi Pin 3  (GPIO2, I2C SDA)
    MLX90640 SCL  →  RPi Pin 5  (GPIO3, I2C SCL)
    """

    WIDTH = 32
    HEIGHT = 24

    def __init__(self, i2c_address: int, refresh_rate: int):
        """
        Initialize MLX90640 connector.

        Parameters
        ----------
        i2c_address : int
            I2C address of the sensor.
        refresh_rate : int
            Refresh rate in Hz.
        """
        self._sensor = None
        self._ready = False

        if not _MLX90640_AVAILABLE or _mlx90640_lib is None:
            logging.error("ThermalCamera MLX90640: library not available.")
            return
        try:
            import board
            import busio

            i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
            self._sensor = _mlx90640_lib.MLX90640(i2c, address=i2c_address)
            self._sensor.refresh_rate = getattr(
                _mlx90640_lib.RefreshRate,
                f"REFRESH_{refresh_rate}_HZ",
                _mlx90640_lib.RefreshRate.REFRESH_4_HZ,
            )
            self._ready = True
            logging.info(
                f"ThermalCamera MLX90640: initialized at address 0x{i2c_address:02x}"
            )
        except Exception as e:
            logging.error(f"ThermalCamera MLX90640: setup failed: {e}")

    async def read(self) -> Optional[ThermalReading]:
        """
        Read a thermal frame.

        Returns
        -------
        Optional[ThermalReading]
            Thermal reading or None if unavailable.
        """
        if not self._ready or self._sensor is None:
            return None
        try:
            frame = [0.0] * (self.WIDTH * self.HEIGHT)
            self._sensor.getFrame(frame)
            return ThermalReading(frame, self.WIDTH, self.HEIGHT)
        except Exception as e:
            logging.warning(f"ThermalCamera MLX90640: read error: {e}")
            return None

    def stop(self):
        """Release MLX90640 resources."""
        self._ready = False
        logging.info("ThermalCamera MLX90640: stopped")


class _AMG8833Connector:
    """
    Read thermal data from Panasonic AMG8833 8x8 IR array via I2C.

    Wiring (AMG8833 to Raspberry Pi)
    -----------------------------------
    AMG8833 VIN  →  RPi Pin 1  (3.3V)
    AMG8833 GND  →  RPi Pin 6  (GND)
    AMG8833 SDA  →  RPi Pin 3  (GPIO2, I2C SDA)
    AMG8833 SCL  →  RPi Pin 5  (GPIO3, I2C SCL)
    """

    WIDTH = 8
    HEIGHT = 8

    def __init__(self):
        """Initialize AMG8833 connector."""
        self._sensor = None
        self._ready = False

        if not _AMG8833_AVAILABLE or _amg88xx_lib is None:
            logging.error("ThermalCamera AMG8833: library not available.")
            return
        try:
            import board
            import busio

            i2c = busio.I2C(board.SCL, board.SDA)
            self._sensor = _amg88xx_lib.AMG88XX(i2c)
            self._ready = True
            logging.info("ThermalCamera AMG8833: initialized")
        except Exception as e:
            logging.error(f"ThermalCamera AMG8833: setup failed: {e}")

    async def read(self) -> Optional[ThermalReading]:
        """
        Read a thermal frame.

        Returns
        -------
        Optional[ThermalReading]
            Thermal reading or None if unavailable.
        """
        if not self._ready or self._sensor is None:
            return None
        try:
            pixels = self._sensor.pixels
            frame = [temp for row in pixels for temp in row]
            return ThermalReading(frame, self.WIDTH, self.HEIGHT)
        except Exception as e:
            logging.warning(f"ThermalCamera AMG8833: read error: {e}")
            return None

    def stop(self):
        """Release AMG8833 resources."""
        self._ready = False
        logging.info("ThermalCamera AMG8833: stopped")


class _USBThermalConnector:
    """
    Read thermal data from a USB thermal camera via OpenCV.

    Compatible with FLIR Lepton USB, Seek Thermal Compact,
    and any V4L2 thermal camera that outputs grayscale frames
    where pixel values map linearly to temperature.

    The temperature mapping assumes:
    - pixel value 0   → min_temp_c
    - pixel value 255 → max_temp_c

    These defaults cover typical room-to-fire temperature ranges.
    """

    WIDTH = 32
    HEIGHT = 24

    def __init__(
        self, camera_index: int, min_temp_c: float = 0.0, max_temp_c: float = 100.0
    ):
        """
        Initialize USB thermal connector.

        Parameters
        ----------
        camera_index : int
            OpenCV camera index.
        min_temp_c : float
            Temperature corresponding to pixel value 0.
        max_temp_c : float
            Temperature corresponding to pixel value 255.
        """
        self._cap = None
        self._ready = False
        self._min_temp = min_temp_c
        self._max_temp = max_temp_c
        self._temp_range = max_temp_c - min_temp_c

        if not _CV2_AVAILABLE or _cv2 is None:
            logging.error("ThermalCamera USB: opencv not available.")
            return
        try:
            self._cap = _cv2.VideoCapture(camera_index)
            if not self._cap.isOpened():
                logging.error(
                    f"ThermalCamera USB: cannot open camera index {camera_index}"
                )
                self._cap = None
                return
            self._ready = True
            logging.info(f"ThermalCamera USB: opened camera index {camera_index}")
        except Exception as e:
            logging.error(f"ThermalCamera USB: setup failed: {e}")

    async def read(self) -> Optional[ThermalReading]:
        """
        Read and convert a USB thermal camera frame.

        Returns
        -------
        Optional[ThermalReading]
            Thermal reading or None if unavailable.
        """
        if not self._ready or self._cap is None:
            return None
        try:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                return None
            import cv2 as cv2_local

            gray = (
                cv2_local.cvtColor(frame, cv2_local.COLOR_BGR2GRAY)
                if len(frame.shape) == 3
                else frame
            )
            h, w = gray.shape
            temps = (
                (gray.astype(float) / 255.0 * self._temp_range + self._min_temp)
                .flatten()
                .tolist()
            )
            return ThermalReading(temps, w, h)
        except Exception as e:
            logging.warning(f"ThermalCamera USB: read error: {e}")
            return None

    def stop(self):
        """Release USB camera resources."""
        if self._cap is not None:
            self._cap.release()
            logging.info("ThermalCamera USB: camera released")
        self._ready = False


class _SerialThermalConnector:
    """
    Read thermal data from an Arduino with IR sensor via serial.

    The paired Arduino sketch must send comma-separated temperature values
    followed by width and height::

        "THERMAL:24.1,25.3,36.8,...,8,8"

    Where the last two values are width and height of the sensor grid.

    Minimal Arduino sketch (AMG8833 via Adafruit library)
    -------------------------------------------------------
    .. code-block:: cpp

        #include <Wire.h>
        #include <Adafruit_AMG88xx.h>

        Adafruit_AMG88xx amg;
        float pixels[AMG88xx_PIXEL_ARRAY_SIZE];

        void setup() {
            Serial.begin(9600);
            amg.begin();
        }
        void loop() {
            amg.readPixels(pixels);
            Serial.print("THERMAL:");
            for (int i = 0; i < AMG88xx_PIXEL_ARRAY_SIZE; i++) {
                Serial.print(pixels[i]);
                if (i < AMG88xx_PIXEL_ARRAY_SIZE - 1) Serial.print(",");
            }
            Serial.println(",8,8");
            delay(500);
        }
    """

    def __init__(self, port: str, baudrate: int, timeout: float):
        """
        Initialize serial thermal connector.

        Parameters
        ----------
        port : str
            Serial port path.
        baudrate : int
            Serial baudrate.
        timeout : float
            Read timeout in seconds.
        """
        self._ser = None
        if not _SERIAL_AVAILABLE or _serial is None:
            logging.error("ThermalCamera Serial: pyserial not available.")
            return
        try:
            self._ser = _serial.Serial(port, baudrate, timeout=timeout)
            logging.info(f"ThermalCamera Serial: connected to {port} @ {baudrate} baud")
        except _serial.SerialException as e:
            logging.error(f"ThermalCamera Serial: failed to open {port}: {e}")

    async def read(self) -> Optional[ThermalReading]:
        """
        Read a thermal frame from serial.

        Returns
        -------
        Optional[ThermalReading]
            Thermal reading or None if unavailable.
        """
        if self._ser is None:
            return None
        try:
            raw = self._ser.readline().decode("utf-8").strip()
        except Exception as e:
            logging.warning(f"ThermalCamera Serial: read error: {e}")
            return None

        if not raw.startswith("THERMAL:"):
            if raw:
                logging.debug(f"ThermalCamera Serial: unrecognised line: '{raw}'")
            return None
        try:
            values = [float(v) for v in raw[8:].split(",") if v.strip()]
            if len(values) < 3:
                return None
            width = int(values[-1])
            height = int(values[-2])
            frame = values[:-2]
            if len(frame) != width * height:
                logging.warning(
                    f"ThermalCamera Serial: expected {width*height} pixels, got {len(frame)}"
                )
                return None
            return ThermalReading(frame, width, height)
        except (ValueError, IndexError) as e:
            logging.warning(f"ThermalCamera Serial: parse error: {e}")
            return None

    def stop(self):
        """Release serial resources."""
        if self._ser and self._ser.is_open:
            self._ser.close()
            logging.info("ThermalCamera Serial: port closed")


class _MockThermalConnector:
    """
    Simulated thermal connector for development and testing.

    Cycles through three scenarios: clear, human detection, and alert.
    """

    WIDTH = 8
    HEIGHT = 8

    def __init__(self, scenario: str = "human"):
        """
        Initialize mock thermal connector.

        Parameters
        ----------
        scenario : str
            Scenario to simulate: 'clear', 'human', or 'alert'.
        """
        import random

        self._random = random
        self._scenario = scenario
        self._call_count = 0
        logging.info(
            f"ThermalCamera mock: simulation active (scenario='{scenario}'). No real hardware used."
        )

    async def read(self) -> Optional[ThermalReading]:
        """
        Return simulated thermal frame.

        Returns
        -------
        Optional[ThermalReading]
            Simulated thermal reading.
        """
        await asyncio.sleep(0)
        self._call_count += 1

        ambient = 22.0 + self._random.uniform(-1.0, 1.0)
        frame = [
            ambient + self._random.uniform(-0.5, 0.5)
            for _ in range(self.WIDTH * self.HEIGHT)
        ]

        if self._scenario == "human":
            center_start = (self.HEIGHT // 2 - 1) * self.WIDTH + self.WIDTH // 4
            for i in range(4):
                if center_start + i < len(frame):
                    frame[center_start + i] = 36.5 + self._random.uniform(-0.5, 0.5)
        elif self._scenario == "alert":
            frame[0] = 75.0 + self._random.uniform(-2.0, 2.0)

        return ThermalReading(frame, self.WIDTH, self.HEIGHT)

    def stop(self):
        """Stop mock connector."""
        logging.info("ThermalCamera mock: stopped")


class ThermalCameraInput(FuserInput[ThermalCameraConfig, Optional[ThermalReading]]):
    """
    Universal Thermal Camera input plugin for OM1.

    Reads thermal data from IR sensors and converts temperature arrays
    into natural language context for the LLM. Detects human presence,
    abnormal heat sources, and fire hazards.

    Supports five hardware backends: MLX90640, AMG8833, USB thermal
    camera, Arduino serial, and mock simulation.

    A ``cooldown`` parameter prevents the LLM context from being flooded
    with repeated identical alerts.

    Example config entry::

        {
          "type": "ThermalCameraInput",
          "config": {
            "connector": "amg8833",
            "cooldown": 3.0,
            "human_temp_min": 34.0,
            "human_temp_max": 39.0,
            "alert_temp_threshold": 60.0
          }
        }
    """

    def __init__(self, config: ThermalCameraConfig):
        """
        Initialize ThermalCameraInput.

        Parameters
        ----------
        config : ThermalCameraConfig
            Plugin configuration.
        """
        super().__init__(config)

        self.descriptor_for_LLM = "Thermal Camera"
        self.io_provider = IOProvider()
        self.messages: list[Message] = []
        self._last_alert_time: float = 0.0

        connector = config.connector.lower()

        if connector == "mlx90640":
            self._connector = _MLX90640Connector(
                i2c_address=config.i2c_address,
                refresh_rate=config.refresh_rate,
            )
        elif connector == "amg8833":
            self._connector = _AMG8833Connector()
        elif connector == "usb":
            self._connector = _USBThermalConnector(
                camera_index=config.camera_index,
            )
        elif connector == "serial":
            self._connector = _SerialThermalConnector(
                port=config.port,
                baudrate=config.baudrate,
                timeout=config.serial_timeout,
            )
        elif connector == "mock":
            self._connector = _MockThermalConnector(
                scenario=config.mock_scenario,
            )
        else:
            logging.error(
                f"ThermalCameraInput: unknown connector '{connector}'. "
                "Valid options: mlx90640, amg8833, usb, serial, mock. "
                "Falling back to mock."
            )
            self._connector = _MockThermalConnector(
                scenario=config.mock_scenario,
            )

        logging.info(
            f"ThermalCameraInput initialized: connector='{connector}', "
            f"cooldown={config.cooldown}s"
        )

    async def _poll(self) -> Optional[ThermalReading]:
        """
        Poll the active connector for a thermal frame.

        Applies cooldown logic to suppress repeated alerts.

        Returns
        -------
        Optional[ThermalReading]
            Thermal reading if available, None otherwise.
        """
        await asyncio.sleep(0.5)
        return await self._connector.read()

    def _classify(self, reading: ThermalReading) -> tuple[str, float, str]:
        """
        Classify a thermal reading into a category.

        Parameters
        ----------
        reading : ThermalReading
            The thermal reading to classify.

        Returns
        -------
        tuple[str, float, str]
            Tuple of (category, peak_temp, zone) where category is
            'alert', 'human', or 'clear'.
        """
        max_temp = reading.max_temp

        zones = ["left", "center", "right"]
        zone_temps = {z: reading.get_zone_max(z) for z in zones}
        hottest_zone = max(zone_temps, key=lambda z: zone_temps[z])

        if max_temp >= self.config.alert_temp_threshold:
            return "alert", max_temp, hottest_zone

        if self.config.human_temp_min <= max_temp <= self.config.human_temp_max:
            return "human", max_temp, hottest_zone

        return "clear", max_temp, hottest_zone

    async def _raw_to_text(
        self, raw_input: Optional[ThermalReading]
    ) -> Optional[Message]:
        """
        Convert a thermal reading to a natural language message.

        Parameters
        ----------
        raw_input : Optional[ThermalReading]
            Thermal reading from the sensor.

        Returns
        -------
        Optional[Message]
            Timestamped message for LLM context, or None if suppressed.
        """
        if raw_input is None:
            return None

        category, peak_temp, zone = self._classify(raw_input)
        now = time.time()

        if category == "alert":
            if (now - self._last_alert_time) < self.config.cooldown:
                return None
            self._last_alert_time = now
            message = (
                f"THERMAL ALERT: Abnormal heat detected at {zone} ({peak_temp:.1f}°C). "
                f"Possible fire or overheating equipment. Immediate inspection recommended."
            )
        elif category == "human":
            if (now - self._last_alert_time) < self.config.cooldown:
                return None
            self._last_alert_time = now
            message = (
                f"Thermal camera: Human-like heat signature detected at {zone} "
                f"({peak_temp:.1f}°C). Possible person present."
            )
        else:
            message = (
                f"Thermal camera: No significant heat signatures detected. "
                f"Max temperature: {peak_temp:.1f}°C."
            )

        return Message(timestamp=now, message=message)

    async def raw_to_text(self, raw_input: Optional[ThermalReading]):
        """
        Convert raw thermal input to text and append to message buffer.

        Parameters
        ----------
        raw_input : Optional[ThermalReading]
            Raw thermal reading from the sensor connector.
        """
        pending = await self._raw_to_text(raw_input)
        if pending is not None:
            self.messages.append(pending)

    def formatted_latest_buffer(self) -> Optional[str]:
        """
        Format the latest buffered message for LLM context injection.

        Clears the buffer after formatting.

        Returns
        -------
        Optional[str]
            Formatted context string, or None if no events buffered.
        """
        if len(self.messages) == 0:
            return None

        latest = self.messages[-1]

        result = (
            f"\nINPUT: {self.descriptor_for_LLM}\n// START\n"
            f"{latest.message}\n// END\n"
        )

        self.io_provider.add_input(
            self.__class__.__name__, latest.message, latest.timestamp
        )
        self.messages = []
        return result

    def stop(self):
        """Gracefully shut down the thermal camera connector."""
        logging.info("ThermalCameraInput: stopping")
        if self._connector and hasattr(self._connector, "stop"):
            self._connector.stop()
        self.messages = []
