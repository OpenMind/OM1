import asyncio
import logging
import time
from typing import Optional

from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider

try:
    import adafruit_ens160 as _ens160_lib

    _ENS160_AVAILABLE = True
except ImportError:
    _ens160_lib = None
    _ENS160_AVAILABLE = False
    logging.warning(
        "adafruit-ens160 not found. ENS160 connector unavailable. "
        "Install with: pip install adafruit-circuitpython-ens160"
    )

try:
    import adafruit_sgp30 as _sgp30_lib

    _SGP30_AVAILABLE = True
except ImportError:
    _sgp30_lib = None
    _SGP30_AVAILABLE = False
    logging.warning(
        "adafruit-sgp30 not found. SGP30 connector unavailable. "
        "Install with: pip install adafruit-circuitpython-sgp30"
    )

try:
    import serial as _serial

    _SERIAL_AVAILABLE = True
except ImportError:
    _serial = None
    _SERIAL_AVAILABLE = False
    logging.warning(
        "pyserial not found. Serial connector unavailable. "
        "Install with: pip install pyserial"
    )

SMOKE_WARNING_THRESHOLD = 300
SMOKE_DANGER_THRESHOLD = 600
GAS_WARNING_THRESHOLD = 300
GAS_DANGER_THRESHOLD = 600


class SmokeGasDetectorConfig(SensorConfig):
    """
    Configuration for Smoke and Gas Detector input plugin.

    Supports five hardware backends via the ``connector`` field:

    - ``"serial"``     : Arduino with MQ-2/MQ-7/MQ-135 via pyserial.
    - ``"i2c_ens160"`` : ENS160 multi-gas sensor via I2C.
    - ``"i2c_sgp30"``  : SGP30 air quality sensor via I2C.
    - ``"mock"``       : Simulated data for development (default).

    Parameters
    ----------
    connector : str
        Hardware backend to use. Default is ``"mock"``.
    port : str
        Serial port path. Used when connector='serial'. Default ``"/dev/ttyUSB0"``.
    baudrate : int
        Serial baudrate. Default 9600.
    serial_timeout : float
        Serial read timeout in seconds. Default 1.0.
    cooldown : float
        Minimum seconds between alerts forwarded to LLM. Default 5.0.
    smoke_warning_threshold : int
        Smoke level (ppm) above which warning is triggered. Default 300.
    smoke_danger_threshold : int
        Smoke level (ppm) above which danger alert is triggered. Default 600.
    gas_warning_threshold : int
        Gas level (ppm) above which warning is triggered. Default 300.
    gas_danger_threshold : int
        Gas level (ppm) above which danger alert is triggered. Default 600.
    mock_scenario : str
        Mock scenario: ``"normal"``, ``"warning"``, or ``"danger"``. Default ``"normal"``.
    """

    connector: str = Field(
        default="mock",
        description=(
            "Hardware backend: 'serial' (Arduino MQ sensors), "
            "'i2c_ens160' (ENS160), "
            "'i2c_sgp30' (SGP30), "
            "'mock' (testing)"
        ),
    )
    port: str = Field(
        default="/dev/ttyUSB0",
        description="Serial port path. Used when connector='serial'.",
    )
    baudrate: int = Field(
        default=9600,
        description="Serial baudrate.",
    )
    serial_timeout: float = Field(
        default=1.0,
        description="Serial read timeout in seconds.",
    )
    cooldown: float = Field(
        default=5.0,
        description="Minimum seconds between alerts forwarded to LLM.",
    )
    smoke_warning_threshold: int = Field(
        default=SMOKE_WARNING_THRESHOLD,
        description="Smoke level (ppm) above which warning is triggered.",
    )
    smoke_danger_threshold: int = Field(
        default=SMOKE_DANGER_THRESHOLD,
        description="Smoke level (ppm) above which danger alert is triggered.",
    )
    gas_warning_threshold: int = Field(
        default=GAS_WARNING_THRESHOLD,
        description="Gas level (ppm) above which warning is triggered.",
    )
    gas_danger_threshold: int = Field(
        default=GAS_DANGER_THRESHOLD,
        description="Gas level (ppm) above which danger alert is triggered.",
    )
    mock_scenario: str = Field(
        default="normal",
        description="Mock scenario: 'normal', 'warning', or 'danger'.",
    )


class SmokeGasReading:
    """
    Container for a smoke/gas sensor reading.

    Parameters
    ----------
    smoke_ppm : float
        Smoke concentration in parts per million.
    gas_ppm : float
        Gas concentration in parts per million.
    sensor_type : str
        Name of the sensor that produced this reading.
    """

    def __init__(self, smoke_ppm: float, gas_ppm: float, sensor_type: str = "unknown"):
        """
        Initialize SmokeGasReading.

        Parameters
        ----------
        smoke_ppm : float
            Smoke concentration in ppm.
        gas_ppm : float
            Gas concentration in ppm.
        sensor_type : str
            Sensor type name.
        """
        self.smoke_ppm = smoke_ppm
        self.gas_ppm = gas_ppm
        self.sensor_type = sensor_type


class _SerialSmokeConnector:
    """
    Read smoke/gas data from Arduino with MQ sensors via serial.

    The paired Arduino sketch must send data in this format::

        "SMOKE:450,GAS:320"

    Minimal Arduino sketch (MQ-2 sensor)
    -------------------------------------------------------
    .. code-block:: cpp

        const int mq2Pin = A0;
        const int mq7Pin = A1;

        void setup() {
            Serial.begin(9600);
        }
        void loop() {
            int smoke = analogRead(mq2Pin);
            int gas = analogRead(mq7Pin);
            Serial.print("SMOKE:");
            Serial.print(smoke);
            Serial.print(",GAS:");
            Serial.println(gas);
            delay(500);
        }
    """

    def __init__(self, port: str, baudrate: int, timeout: float):
        """
        Initialize serial smoke connector.

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
            logging.error("SmokeGasDetector Serial: pyserial not available.")
            return
        try:
            self._ser = _serial.Serial(port, baudrate, timeout=timeout)
            logging.info(
                f"SmokeGasDetector Serial: connected to {port} @ {baudrate} baud"
            )
        except _serial.SerialException as e:
            logging.error(f"SmokeGasDetector Serial: failed to open {port}: {e}")

    async def read(self) -> Optional[SmokeGasReading]:
        """
        Read a smoke/gas frame from serial.

        Returns
        -------
        Optional[SmokeGasReading]
            Smoke/gas reading or None if unavailable.
        """
        if self._ser is None:
            return None
        try:
            raw = self._ser.readline().decode("utf-8").strip()
        except Exception as e:
            logging.warning(f"SmokeGasDetector Serial: read error: {e}")
            return None

        if not raw.startswith("SMOKE:"):
            if raw:
                logging.debug(f"SmokeGasDetector Serial: unrecognised line: '{raw}'")
            return None
        try:
            parts = dict(p.split(":") for p in raw.split(",") if ":" in p)
            smoke = float(parts.get("SMOKE", 0))
            gas = float(parts.get("GAS", 0))
            return SmokeGasReading(
                smoke_ppm=smoke, gas_ppm=gas, sensor_type="Arduino-MQ"
            )
        except (ValueError, KeyError) as e:
            logging.warning(f"SmokeGasDetector Serial: parse error: {e}")
            return None

    def stop(self):
        """Release serial resources."""
        if self._ser and self._ser.is_open:
            self._ser.close()
            logging.info("SmokeGasDetector Serial: port closed")


class _ENS160Connector:
    """
    Read air quality data from ENS160 multi-gas sensor via I2C.

    Wiring (ENS160 to Raspberry Pi)
    -----------------------------------
    ENS160 VIN  →  RPi Pin 1  (3.3V)
    ENS160 GND  →  RPi Pin 6  (GND)
    ENS160 SDA  →  RPi Pin 3  (GPIO2, I2C SDA)
    ENS160 SCL  →  RPi Pin 5  (GPIO3, I2C SCL)
    """

    def __init__(self):
        """Initialize ENS160 connector."""
        self._sensor = None
        self._ready = False

        if not _ENS160_AVAILABLE or _ens160_lib is None:
            logging.error("SmokeGasDetector ENS160: library not available.")
            return
        try:
            import board
            import busio

            i2c = busio.I2C(board.SCL, board.SDA)
            self._sensor = _ens160_lib.ENS160(i2c)
            self._ready = True
            logging.info("SmokeGasDetector ENS160: initialized")
        except Exception as e:
            logging.error(f"SmokeGasDetector ENS160: setup failed: {e}")

    async def read(self) -> Optional[SmokeGasReading]:
        """
        Read air quality from ENS160.

        Returns
        -------
        Optional[SmokeGasReading]
            Smoke/gas reading or None if unavailable.
        """
        if not self._ready or self._sensor is None:
            return None
        try:
            tvoc = float(self._sensor.TVOC)
            eco2 = float(self._sensor.eCO2)
            return SmokeGasReading(smoke_ppm=tvoc, gas_ppm=eco2, sensor_type="ENS160")
        except Exception as e:
            logging.warning(f"SmokeGasDetector ENS160: read error: {e}")
            return None

    def stop(self):
        """Release ENS160 resources."""
        self._ready = False
        logging.info("SmokeGasDetector ENS160: stopped")


class _SGP30Connector:
    """
    Read air quality data from SGP30 sensor via I2C.

    Wiring (SGP30 to Raspberry Pi)
    -----------------------------------
    SGP30 VIN  →  RPi Pin 1  (3.3V)
    SGP30 GND  →  RPi Pin 6  (GND)
    SGP30 SDA  →  RPi Pin 3  (GPIO2, I2C SDA)
    SGP30 SCL  →  RPi Pin 5  (GPIO3, I2C SCL)
    """

    def __init__(self):
        """Initialize SGP30 connector."""
        self._sensor = None
        self._ready = False

        if not _SGP30_AVAILABLE or _sgp30_lib is None:
            logging.error("SmokeGasDetector SGP30: library not available.")
            return
        try:
            import board
            import busio

            i2c = busio.I2C(board.SCL, board.SDA, frequency=100000)
            self._sensor = _sgp30_lib.Adafruit_SGP30(i2c)
            self._sensor.iaq_init()
            self._ready = True
            logging.info("SmokeGasDetector SGP30: initialized")
        except Exception as e:
            logging.error(f"SmokeGasDetector SGP30: setup failed: {e}")

    async def read(self) -> Optional[SmokeGasReading]:
        """
        Read air quality from SGP30.

        Returns
        -------
        Optional[SmokeGasReading]
            Smoke/gas reading or None if unavailable.
        """
        if not self._ready or self._sensor is None:
            return None
        try:
            tvoc = float(self._sensor.TVOC)
            eco2 = float(self._sensor.eCO2)
            return SmokeGasReading(smoke_ppm=tvoc, gas_ppm=eco2, sensor_type="SGP30")
        except Exception as e:
            logging.warning(f"SmokeGasDetector SGP30: read error: {e}")
            return None

    def stop(self):
        """Release SGP30 resources."""
        self._ready = False
        logging.info("SmokeGasDetector SGP30: stopped")


class _MockSmokeConnector:
    """
    Simulated smoke/gas connector for development and testing.

    Produces three scenarios: normal air quality, warning level, and danger level.
    """

    def __init__(self, scenario: str = "normal"):
        """
        Initialize mock smoke connector.

        Parameters
        ----------
        scenario : str
            Scenario to simulate: 'normal', 'warning', or 'danger'.
        """
        import random

        self._random = random
        self._scenario = scenario
        logging.info(
            f"SmokeGasDetector mock: simulation active (scenario='{scenario}'). "
            "No real hardware used."
        )

    async def read(self) -> Optional[SmokeGasReading]:
        """
        Return simulated smoke/gas reading.

        Returns
        -------
        Optional[SmokeGasReading]
            Simulated reading.
        """
        await asyncio.sleep(0)

        if self._scenario == "warning":
            smoke = 350.0 + self._random.uniform(-20.0, 20.0)
            gas = 320.0 + self._random.uniform(-20.0, 20.0)
        elif self._scenario == "danger":
            smoke = 750.0 + self._random.uniform(-30.0, 30.0)
            gas = 700.0 + self._random.uniform(-30.0, 30.0)
        else:
            smoke = 50.0 + self._random.uniform(-10.0, 10.0)
            gas = 40.0 + self._random.uniform(-10.0, 10.0)

        return SmokeGasReading(smoke_ppm=smoke, gas_ppm=gas, sensor_type="mock")

    def stop(self):
        """Stop mock connector."""
        logging.info("SmokeGasDetector mock: stopped")


class SmokeGasDetector(FuserInput[SmokeGasDetectorConfig, Optional[SmokeGasReading]]):
    """
    Universal Smoke and Gas Detector input plugin for OM1.

    Reads smoke and gas sensor data and converts readings into natural language
    context for the LLM. Detects normal air quality, warning levels, and
    dangerous smoke/gas concentrations for fire and hazard detection.

    Supports four hardware backends: Arduino MQ sensors via serial, ENS160,
    SGP30, and mock simulation.

    A ``cooldown`` parameter prevents the LLM context from being flooded
    with repeated identical alerts.

    Example config entry::

        {
          "type": "SmokeGasDetector",
          "config": {
            "connector": "serial",
            "port": "/dev/ttyUSB0",
            "cooldown": 5.0,
            "smoke_danger_threshold": 600,
            "gas_danger_threshold": 600
          }
        }
    """

    def __init__(self, config: SmokeGasDetectorConfig):
        """
        Initialize SmokeGasDetector.

        Parameters
        ----------
        config : SmokeGasDetectorConfig
            Plugin configuration.
        """
        super().__init__(config)

        self.descriptor_for_LLM = "Smoke and Gas Detector"
        self.io_provider = IOProvider()
        self.messages: list[Message] = []
        self._last_alert_time: float = 0.0

        connector = config.connector.lower()

        if connector == "serial":
            self._connector = _SerialSmokeConnector(
                port=config.port,
                baudrate=config.baudrate,
                timeout=config.serial_timeout,
            )
        elif connector == "i2c_ens160":
            self._connector = _ENS160Connector()
        elif connector == "i2c_sgp30":
            self._connector = _SGP30Connector()
        elif connector == "mock":
            self._connector = _MockSmokeConnector(scenario=config.mock_scenario)
        else:
            logging.error(
                f"SmokeGasDetector: unknown connector '{connector}'. "
                "Valid options: serial, i2c_ens160, i2c_sgp30, mock. "
                "Falling back to mock."
            )
            self._connector = _MockSmokeConnector(scenario=config.mock_scenario)

        logging.info(
            f"SmokeGasDetector initialized: connector='{connector}', "
            f"cooldown={config.cooldown}s"
        )

    async def _poll(self) -> Optional[SmokeGasReading]:
        """
        Poll the active connector for a smoke/gas reading.

        Returns
        -------
        Optional[SmokeGasReading]
            Smoke/gas reading if available, None otherwise.
        """
        await asyncio.sleep(0.5)
        return await self._connector.read()

    def _classify(self, reading: SmokeGasReading) -> str:
        """
        Classify a smoke/gas reading into a severity level.

        Parameters
        ----------
        reading : SmokeGasReading
            The reading to classify.

        Returns
        -------
        str
            Severity level: 'danger', 'warning', or 'normal'.
        """
        if (
            reading.smoke_ppm >= self.config.smoke_danger_threshold
            or reading.gas_ppm >= self.config.gas_danger_threshold
        ):
            return "danger"
        if (
            reading.smoke_ppm >= self.config.smoke_warning_threshold
            or reading.gas_ppm >= self.config.gas_warning_threshold
        ):
            return "warning"
        return "normal"

    async def _raw_to_text(
        self, raw_input: Optional[SmokeGasReading]
    ) -> Optional[Message]:
        """
        Convert a smoke/gas reading to a natural language message.

        Parameters
        ----------
        raw_input : Optional[SmokeGasReading]
            Smoke/gas reading from the sensor.

        Returns
        -------
        Optional[Message]
            Timestamped message for LLM context, or None if suppressed.
        """
        if raw_input is None:
            return None

        level = self._classify(raw_input)
        now = time.time()

        if level == "danger":
            if (now - self._last_alert_time) < self.config.cooldown:
                return None
            self._last_alert_time = now
            message = (
                f"SMOKE ALERT: Critical smoke/gas level detected. "
                f"Smoke: {raw_input.smoke_ppm:.0f} ppm, Gas: {raw_input.gas_ppm:.0f} ppm. "
                f"Immediate evacuation recommended. Possible fire or gas leak."
            )
        elif level == "warning":
            if (now - self._last_alert_time) < self.config.cooldown:
                return None
            self._last_alert_time = now
            message = (
                f"SMOKE WARNING: Elevated smoke/gas detected. "
                f"Smoke: {raw_input.smoke_ppm:.0f} ppm, Gas: {raw_input.gas_ppm:.0f} ppm. "
                f"Possible fire risk. Inspect area immediately."
            )
        else:
            message = (
                f"Smoke/gas detector: Air quality normal. "
                f"Smoke: {raw_input.smoke_ppm:.0f} ppm, Gas: {raw_input.gas_ppm:.0f} ppm."
            )

        return Message(timestamp=now, message=message)

    async def raw_to_text(self, raw_input: Optional[SmokeGasReading]):
        """
        Convert raw smoke/gas input to text and append to message buffer.

        Parameters
        ----------
        raw_input : Optional[SmokeGasReading]
            Raw smoke/gas reading from the sensor connector.
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
            f"\nINPUT: {self.descriptor_for_LLM}\n// START\n{latest.message}\n// END\n"
        )

        self.io_provider.add_input(
            self.__class__.__name__, latest.message, latest.timestamp
        )
        self.messages = []
        return result

    def stop(self):
        """Gracefully shut down the smoke/gas detector connector."""
        logging.info("SmokeGasDetector: stopping")
        if self._connector and hasattr(self._connector, "stop"):
            self._connector.stop()
        self.messages = []
