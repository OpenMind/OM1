import asyncio
import logging
import time
from typing import Optional

from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider

try:
    import serial as _serial

    _SERIAL_AVAILABLE = True
except ImportError:
    _serial = None  # type: ignore
    _SERIAL_AVAILABLE = False
    logging.warning(
        "pyserial not found. PIRMotionInput serial connector unavailable. "
        "Install with: pip install pyserial"
    )

try:
    import RPi.GPIO as _GPIO  # type: ignore

    _GPIO_LIB = _GPIO
    _GPIO_AVAILABLE = True
    logging.info("PIRMotionInput: using RPi.GPIO")
except ImportError:
    try:
        import Jetson.GPIO as _GPIO  # type: ignore

        _GPIO_LIB = _GPIO
        _GPIO_AVAILABLE = True
        logging.info("PIRMotionInput: using Jetson.GPIO")
    except ImportError:
        _GPIO_LIB = None
        _GPIO_AVAILABLE = False
        logging.warning(
            "Neither RPi.GPIO nor Jetson.GPIO found. "
            "PIRMotionInput GPIO connector unavailable."
        )

try:
    import zenoh as _zenoh  # type: ignore

    _ZENOH_AVAILABLE = True
except ImportError:
    _zenoh = None  # type: ignore
    _ZENOH_AVAILABLE = False
    logging.warning("zenoh not found. PIRMotionInput zenoh connector unavailable.")


class PIRMotionConfig(SensorConfig):
    """
    Configuration for PIR Motion Sensor (HC-SR501) input plugin.

    Supports four hardware backends via the ``connector`` field:

    - ``"serial"``  : Arduino or any USB microcontroller via pyserial.
    - ``"gpio"``    : Direct BCM GPIO on Raspberry Pi or Jetson.
    - ``"zenoh"``   : Network-distributed input via Zenoh pub/sub.
    - ``"mock"``    : Simulated events for development and testing (default).

    Parameters
    ----------
    connector : str
        Hardware backend to use. Default is ``"mock"``.
    port : str
        Serial device path. Used only when ``connector="serial"``.
        Examples: ``"/dev/ttyUSB0"`` (Linux), ``"COM3"`` (Windows),
        ``"/dev/cu.usbmodem1101"`` (macOS).
    baudrate : int
        Serial baudrate. Must match the Arduino sketch. Default 9600.
    serial_timeout : float
        Serial readline timeout in seconds. Default 1.0.
    gpio_pin : int
        BCM GPIO pin number connected to HC-SR501 OUT. Default 17.
    zenoh_topic : str
        Zenoh key expression to subscribe to. Default ``"om/sensors/pir"``.
    cooldown : float
        Minimum seconds between motion events forwarded to the LLM.
        Prevents context flooding when the sensor output stays HIGH
        (HC-SR501 can hold HIGH for up to ~200 s). Default 5.0.
    mock_trigger_interval : int
        Average number of poll cycles between simulated motion events.
        Only used when ``connector="mock"``. Default 10.
    """

    connector: str = Field(
        default="mock",
        description=(
            "Hardware backend: 'serial' (Arduino/USB), "
            "'gpio' (Raspberry Pi/Jetson), "
            "'zenoh' (network), "
            "'mock' (testing/simulation)"
        ),
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
        description="Serial read timeout in seconds. Used when connector='serial'.",
    )
    gpio_pin: int = Field(
        default=17,
        description="BCM GPIO pin number for PIR OUT signal. Used when connector='gpio'.",
    )
    zenoh_topic: str = Field(
        default="om/sensors/pir",
        description="Zenoh topic for PIR data. Used when connector='zenoh'.",
    )
    cooldown: float = Field(
        default=5.0,
        description=(
            "Minimum seconds between motion events forwarded to LLM. "
            "Prevents context flooding when sensor stays HIGH."
        ),
    )
    mock_trigger_interval: int = Field(
        default=10,
        description=(
            "Average poll cycles between simulated motion events. "
            "Used only when connector='mock'."
        ),
    )


class _SerialPIRConnector:
    """
    Read HC-SR501 state from an Arduino via USB serial.

    The paired Arduino sketch must send one line per reading::

        "MOTION:1"  →  PIR OUT is HIGH (motion detected)
        "MOTION:0"  →  PIR OUT is LOW  (no motion)

    Minimal Arduino sketch
    ----------------------
    .. code-block:: cpp

        const int PIR_PIN = 2;
        void setup() {
            Serial.begin(9600);
            pinMode(PIR_PIN, INPUT);
        }
        void loop() {
            Serial.println(digitalRead(PIR_PIN) ? "MOTION:1" : "MOTION:0");
            delay(500);
        }

    Wiring (HC-SR501 to Arduino Uno)
    ---------------------------------
    HC-SR501 VCC  →  Arduino 5V
    HC-SR501 GND  →  Arduino GND
    HC-SR501 OUT  →  Arduino D2
    """

    def __init__(self, port: str, baudrate: int, timeout: float):
        self._ser = None
        if not _SERIAL_AVAILABLE or _serial is None:
            logging.error("_SerialPIRConnector: pyserial not available.")
            return
        try:
            self._ser = _serial.Serial(port, baudrate, timeout=timeout)
            logging.info(
                f"PIRMotionInput serial: connected to {port} @ {baudrate} baud"
            )
        except _serial.SerialException as e:
            logging.error(f"PIRMotionInput serial: failed to open {port}: {e}")

    async def read(self) -> Optional[bool]:
        if self._ser is None:
            return None
        try:
            raw = self._ser.readline().decode("utf-8").strip()
        except Exception as e:
            logging.warning(f"PIRMotionInput serial: read error: {e}")
            return None
        if raw == "MOTION:1":
            return True
        if raw == "MOTION:0":
            return False
        if raw:
            logging.debug(f"PIRMotionInput serial: unrecognised line: '{raw}'")
        return None

    def stop(self):
        if self._ser and self._ser.is_open:
            self._ser.close()
            logging.info("PIRMotionInput serial: port closed")


class _GPIOPIRConnector:
    """
    Read HC-SR501 state directly from a BCM GPIO pin.

    Compatible with Raspberry Pi (RPi.GPIO) and Jetson (Jetson.GPIO).

    Wiring (HC-SR501 to Raspberry Pi)
    -----------------------------------
    HC-SR501 VCC  →  RPi Pin 2  (5V)
    HC-SR501 GND  →  RPi Pin 6  (GND)
    HC-SR501 OUT  →  RPi Pin 11 (GPIO17, BCM)

    .. note::
        HC-SR501 OUT is typically 3.3 V-safe, but verify your sensor's
        datasheet before connecting directly to a 3.3 V GPIO.
    """

    def __init__(self, pin: int):
        self._pin = pin
        self._ready = False
        if not _GPIO_AVAILABLE or _GPIO_LIB is None:
            logging.error("PIRMotionInput GPIO: GPIO library not available.")
            return
        try:
            _GPIO_LIB.setmode(_GPIO_LIB.BCM)
            _GPIO_LIB.setup(self._pin, _GPIO_LIB.IN)
            self._ready = True
            logging.info(f"PIRMotionInput GPIO: GPIO{pin} configured as input (BCM)")
        except Exception as e:
            logging.error(f"PIRMotionInput GPIO: setup failed: {e}")

    async def read(self) -> Optional[bool]:
        if not self._ready or _GPIO_LIB is None:
            return None
        try:
            return bool(_GPIO_LIB.input(self._pin))
        except Exception as e:
            logging.warning(f"PIRMotionInput GPIO: read error on GPIO{self._pin}: {e}")
            return None

    def stop(self):
        if self._ready and _GPIO_AVAILABLE and _GPIO_LIB is not None:
            try:
                _GPIO_LIB.cleanup(self._pin)
                logging.info(f"PIRMotionInput GPIO: GPIO{self._pin} cleaned up")
            except Exception as e:
                logging.warning(f"PIRMotionInput GPIO: cleanup error: {e}")


class _ZenohPIRConnector:
    """
    Read HC-SR501 state from a Zenoh topic.

    Enables distributed setups where the PIR sensor is attached to a
    remote device (e.g. a Raspberry Pi running a Zenoh publisher) and
    the OM1 runtime is on a different machine on the same network.

    Expected message payload (UTF-8)
    ----------------------------------
    ``"1"`` or ``"MOTION:1"``  →  motion detected
    ``"0"`` or ``"MOTION:0"``  →  no motion

    Example publisher (Python)
    ---------------------------
    .. code-block:: python

        import zenoh, time
        import RPi.GPIO as GPIO

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(17, GPIO.IN)
        session = zenoh.open()
        pub = session.declare_publisher("om/sensors/pir")
        while True:
            pub.put("1" if GPIO.input(17) else "0")
            time.sleep(0.5)
    """

    def __init__(self, topic: str):
        self._topic = topic
        self._session = None
        self._subscriber = None
        self._queue: asyncio.Queue = asyncio.Queue()

        if not _ZENOH_AVAILABLE or _zenoh is None:
            logging.error("PIRMotionInput zenoh: zenoh not available.")
            return
        try:
            self._session = _zenoh.open(_zenoh.Config())
            self._subscriber = self._session.declare_subscriber(topic, self._on_message)
            logging.info(f"PIRMotionInput zenoh: subscribed to '{topic}'")
        except Exception as e:
            logging.error(
                f"PIRMotionInput zenoh: failed to subscribe to '{topic}': {e}"
            )

    def _on_message(self, sample):
        try:
            text = sample.payload.decode("utf-8").strip()
            if text in ("1", "MOTION:1"):
                self._queue.put_nowait(True)
            elif text in ("0", "MOTION:0"):
                self._queue.put_nowait(False)
            else:
                logging.debug(f"PIRMotionInput zenoh: unrecognised payload: '{text}'")
        except Exception as e:
            logging.warning(f"PIRMotionInput zenoh: message parse error: {e}")

    async def read(self) -> Optional[bool]:
        if self._session is None:
            return None
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def stop(self):
        if self._subscriber:
            try:
                self._subscriber.undeclare()
            except Exception as e:
                logging.warning(f"PIRMotionInput zenoh: undeclare error: {e}")
        if self._session:
            try:
                self._session.close()
                logging.info("PIRMotionInput zenoh: session closed")
            except Exception as e:
                logging.warning(f"PIRMotionInput zenoh: session close error: {e}")


class _MockPIRConnector:
    """
    Simulated PIR connector for development and testing.

    Generates synthetic motion events without any physical hardware.
    This is the default connector, ensuring a safe out-of-box experience
    on any machine.

    A motion event (``True``) is produced approximately once every
    ``trigger_interval`` calls, with random jitter.
    """

    def __init__(self, trigger_interval: int = 10):
        import random

        self._trigger_interval = trigger_interval
        self._call_count = 0
        self._random = random
        logging.info(
            f"PIRMotionInput mock: simulation active "
            f"(trigger_interval={trigger_interval}). No real hardware used."
        )

    async def read(self) -> Optional[bool]:
        await asyncio.sleep(0)
        self._call_count += 1
        jitter = self._random.randint(
            max(1, self._trigger_interval // 2),
            self._trigger_interval + self._trigger_interval // 2,
        )
        if self._call_count >= jitter:
            self._call_count = 0
            logging.debug("PIRMotionInput mock: simulated motion event")
            return True
        return False

    def stop(self):
        logging.info("PIRMotionInput mock: stopped")


class PIRMotionInput(FuserInput[PIRMotionConfig, Optional[bool]]):
    """
    PIR Motion Sensor (HC-SR501) input plugin for OM1.

    Reads motion detection events from an HC-SR501 PIR sensor and converts
    them into natural language context for the LLM. Supports four hardware
    backends selectable via the ``connector`` config field.

    A ``cooldown`` parameter prevents the LLM context from being flooded
    when the sensor output stays HIGH (HC-SR501 can hold HIGH for up to
    ~200 seconds).

    Example config entry (``config/pir_guard.json5``)::

        {
          "type": "PIRMotionInput",
          "config": {
            "connector": "serial",
            "port": "/dev/ttyUSB0",
            "baudrate": 9600,
            "cooldown": 5.0
          }
        }
    """

    def __init__(self, config: PIRMotionConfig):
        """
        Initialize PIRMotionInput.

        Parameters
        ----------
        config : PIRMotionConfig
            Plugin configuration. ``connector`` selects the hardware backend.
        """
        super().__init__(config)

        self.descriptor_for_LLM = "PIR Motion Sensor"
        self.io_provider = IOProvider()
        self.messages: list[Message] = []
        self._last_motion_time: float = 0.0

        connector = config.connector.lower()

        if connector == "serial":
            self._connector = _SerialPIRConnector(
                port=config.port,
                baudrate=config.baudrate,
                timeout=config.serial_timeout,
            )
        elif connector == "gpio":
            self._connector = _GPIOPIRConnector(pin=config.gpio_pin)
        elif connector == "zenoh":
            self._connector = _ZenohPIRConnector(topic=config.zenoh_topic)
        elif connector == "mock":
            self._connector = _MockPIRConnector(
                trigger_interval=config.mock_trigger_interval
            )
        else:
            logging.error(
                f"PIRMotionInput: unknown connector '{connector}'. "
                "Valid options: serial, gpio, zenoh, mock. "
                "Falling back to mock."
            )
            self._connector = _MockPIRConnector(
                trigger_interval=config.mock_trigger_interval
            )

        logging.info(
            f"PIRMotionInput initialized: connector='{connector}', "
            f"cooldown={config.cooldown}s"
        )

    async def _poll(self) -> Optional[bool]:
        """
        Poll the active connector for a motion event.

        Applies cooldown logic to suppress repeated detections while
        the sensor output remains HIGH.

        Returns
        -------
        Optional[bool]
            ``True``  if motion detected and cooldown has elapsed.
            ``False`` if no motion detected.
            ``None``  if connector is unavailable or within cooldown window.
        """
        await asyncio.sleep(0.5)

        detected: Optional[bool] = await self._connector.read()

        if detected is True:
            now = time.time()
            if (now - self._last_motion_time) >= self.config.cooldown:
                self._last_motion_time = now
                logging.info("PIRMotionInput: motion detected (cooldown passed)")
                return True
            logging.debug(
                "PIRMotionInput: motion detected but within cooldown, suppressing"
            )
            return None

        return False

    async def _raw_to_text(self, raw_input: Optional[bool]) -> Optional[Message]:
        """
        Convert raw sensor state to a natural language message.

        Parameters
        ----------
        raw_input : Optional[bool]
            ``True``  → motion detected.
            ``False`` → no motion (not forwarded to LLM).
            ``None``  → suppressed or unavailable.

        Returns
        -------
        Optional[Message]
            Timestamped message for LLM context, or ``None``.
        """
        if raw_input is not True:
            return None

        message = (
            "Motion detected by PIR sensor. "
            "A person or moving object is present nearby. "
            "Consider alerting or investigating."
        )
        return Message(timestamp=time.time(), message=message)

    async def raw_to_text(self, raw_input: Optional[bool]):
        """
        Convert raw input to text and append to message buffer.

        Parameters
        ----------
        raw_input : Optional[bool]
            Raw boolean state from the sensor connector.
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
            Formatted context string, or ``None`` if no events buffered.
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
        """
        Gracefully shut down the sensor connector.
        """
        logging.info("PIRMotionInput: stopping")
        if self._connector and hasattr(self._connector, "stop"):
            self._connector.stop()
        self.messages = []
