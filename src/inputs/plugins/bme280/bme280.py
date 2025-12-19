"""BME280 environmental sensor input plugin."""

import asyncio
import logging
import time
from typing import Any, Dict, Optional

from inputs.base import Message
from inputs.base.loop import FuserInput
from inputs.plugins.bme280.config import BME280Config

# Type stubs for optional hardware dependencies
board = None
busio = None
adafruit_bme280 = None

try:
    import adafruit_bme280.advanced as adafruit_bme280  # type: ignore
    import board  # type: ignore
    import busio  # type: ignore

    BME280_AVAILABLE = True
except ImportError:
    BME280_AVAILABLE = False
    logging.warning(
        "BME280 libraries not available. "
        "Install with: pip install adafruit-circuitpython-bme280"
    )


class BME280Input(FuserInput[BME280Config, Dict[str, Any]]):
    """
    BME280 environmental sensor input.

    Reads temperature, humidity, and pressure from BME280 sensor via I2C.
    Falls back to mock data if hardware is not available.

    Attributes
    ----------
    config : BME280Config
        Configuration for the sensor
    sensor : Adafruit_BME280_I2C or None
        The BME280 sensor object if available

    Examples
    --------
    >>> config = BME280Config(i2c_address=0x76, sampling_rate=1.0)
    >>> sensor = BME280Input(config)
    >>> # Use with async for loop
    >>> async for reading in sensor.listen():
    ...     print(reading)
    """

    def __init__(self, config: BME280Config):
        """
        Initialize BME280 sensor.

        Parameters
        ----------
        config : BME280Config
            Configuration object containing I2C address and sampling rate
        """
        super().__init__(config)
        self.config: BME280Config = config
        self.sensor: Optional[Any] = None
        self._initialize_sensor()

    def _initialize_sensor(self) -> None:
        """Initialize the BME280 hardware sensor if available."""
        if not BME280_AVAILABLE or board is None or busio is None:
            logging.info(
                "BME280 libraries not available, using mock data. "
                "Install with: pip install adafruit-circuitpython-bme280"
            )
            return

        if adafruit_bme280 is None:
            logging.info("adafruit_bme280 module not available")
            return

        try:
            i2c = busio.I2C(board.SCL, board.SDA)
            self.sensor = adafruit_bme280.Adafruit_BME280_I2C(
                i2c, address=self.config.i2c_address
            )
            logging.info(
                f"BME280 sensor initialized at address "
                f"{hex(self.config.i2c_address)}"
            )
        except Exception as e:
            logging.error(f"Failed to initialize BME280 sensor: {e}")
            logging.info("Falling back to mock mode")
            self.sensor = None

    def _read_sensor(self) -> Dict[str, Any]:
        """
        Read current sensor values.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing temperature, humidity, pressure, and timestamp
        """
        if self.sensor is not None:
            try:
                return {
                    "temperature": round(self.sensor.temperature, 2),
                    "humidity": round(self.sensor.humidity, 2),
                    "pressure": round(self.sensor.pressure, 2),
                    "timestamp": time.time(),
                }
            except Exception as e:
                logging.error(f"Error reading BME280 sensor: {e}")
                return self._mock_data()
        else:
            return self._mock_data()

    def _mock_data(self) -> Dict[str, Any]:
        """
        Return mock data when sensor is not available.

        Returns
        -------
        Dict[str, Any]
            Mock sensor data with reasonable default values
        """
        return {
            "temperature": 25.0,
            "humidity": 50.0,
            "pressure": 1013.25,
            "timestamp": time.time(),
            "mock": True,
        }

    async def _raw_to_text(self, raw_input: Dict[str, Any]) -> Optional[Message]:
        """
        Convert raw sensor data to text message.

        Parameters
        ----------
        raw_input : Dict[str, Any]
            Raw sensor reading data

        Returns
        -------
        Optional[Message]
            Message object containing formatted sensor data
        """
        text = (
            f"BME280 Sensor Reading: "
            f"Temperature: {raw_input['temperature']}°C, "
            f"Humidity: {raw_input['humidity']}%, "
            f"Pressure: {raw_input['pressure']} hPa"
        )

        if raw_input.get("mock", False):
            text += " (mock data)"

        return Message(timestamp=raw_input["timestamp"], message=text)

    async def _listen_loop(self):
        """
        Continuous loop that reads sensor data at configured sampling rate.

        Yields
        ------
        Dict[str, Any]
            Sensor readings
        """
        logging.info(
            f"Starting BME280 sensor loop with sampling rate "
            f"{self.config.sampling_rate}s"
        )

        while True:
            try:
                reading = self._read_sensor()
                yield reading
                await asyncio.sleep(self.config.sampling_rate)
            except Exception as e:
                logging.error(f"Error in BME280 listen loop: {e}")
                await asyncio.sleep(self.config.sampling_rate)
