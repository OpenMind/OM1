import asyncio

from inputs.plugins.air_quality.connector.base import (
    AirQualityConnector,
    AirQualityData,
)


class BME680Connector(AirQualityConnector):
    """
    Air quality connector for BME680 environmental sensor via I2C.

    Reads temperature, humidity, pressure, and VOC gas resistance.
    Uses the bme680 Python library (pip install bme680).

    Wiring (I2C):
        VCC  → 3.3V
        GND  → GND
        SDA  → SDA (e.g. GPIO2 on Raspberry Pi)
        SCL  → SCL (e.g. GPIO3 on Raspberry Pi)

    Datasheet: https://www.bosch-sensortec.com/products/environmental-sensors/gas-sensors/bme680/
    Library: https://github.com/pimoroni/bme680-python
    """

    def __init__(self, config: dict):
        """
        Parameters
        ----------
        config : dict
            Must contain:
            - i2c_address (int, optional): I2C address, default 0x76
            - location (str, optional): location label, default 'Robot'
            - gas_baseline (float, optional): baseline gas resistance in Ohms
              for IAQ calculation. Calibrate by running sensor in clean air.
        """
        super().__init__(config)
        self.i2c_address: int = config.get("i2c_address", 0x76)
        self.location: str = config.get("location", "Robot")
        self.gas_baseline: float = config.get("gas_baseline", 50000.0)
        self._sensor = None

    async def connect(self) -> bool:
        """Initialize the BME680 sensor over I2C and configure oversampling."""
        try:
            import bme680

            loop = asyncio.get_event_loop()
            self._sensor = await loop.run_in_executor(
                None, lambda: bme680.BME680(self.i2c_address)  # type: ignore[attr-defined]
            )
            # Recommended oversampling settings from Bosch
            self._sensor.set_humidity_oversample(bme680.OS_2X)  # type: ignore[attr-defined]
            self._sensor.set_pressure_oversample(bme680.OS_4X)  # type: ignore[attr-defined]
            self._sensor.set_temperature_oversample(bme680.OS_8X)  # type: ignore[attr-defined]
            self._sensor.set_filter(bme680.FILTER_SIZE_3)  # type: ignore[attr-defined]
            self._sensor.set_gas_status(bme680.ENABLE_GAS_MEAS)  # type: ignore[attr-defined]
            self._sensor.set_gas_heater_temperature(320)
            self._sensor.set_gas_heater_duration(150)
            self._sensor.select_gas_heater_profile(0)

            self.logger.info(
                f"BME680Connector: connected at I2C address {hex(self.i2c_address)}"
            )
            return True

        except ImportError:
            self.logger.error(
                "BME680Connector: bme680 library not installed. Run: pip install bme680"
            )
            return False
        except Exception as e:
            self.logger.error(f"BME680Connector: failed to connect: {e}")
            return False

    async def disconnect(self) -> None:
        """Release the BME680 sensor reference."""
        self._sensor = None
        self.logger.info("BME680Connector: disconnected")

    async def read(self) -> AirQualityData | None:
        """
        Read one sample from BME680 sensor.

        Returns
        -------
        AirQualityData or None
            Environmental data, or None if read failed.
        """
        if self._sensor is None:
            self.logger.error("BME680Connector: not connected")
            return None

        try:
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(None, self._read_sensor)
            return data

        except Exception as e:
            self.logger.error(f"BME680Connector: read error: {e}")
            return None

    def _read_sensor(self) -> AirQualityData | None:
        """
        Blocking sensor read — runs in executor.

        Returns
        -------
        AirQualityData or None
        """
        if self._sensor is None or not self._sensor.get_sensor_data():
            self.logger.warning("BME680Connector: sensor data not ready")
            return None

        temperature = round(self._sensor.data.temperature, 1)
        humidity = round(self._sensor.data.humidity, 1)

        # IAQ (Indoor Air Quality) score 0-500 from gas resistance
        # Higher gas resistance = cleaner air
        aqi = None
        if self._sensor.data.heat_stable:
            gas_resistance = self._sensor.data.gas_resistance
            # Normalize: baseline is clean air (AQI ~25), degraded is AQI ~200+
            ratio = gas_resistance / self.gas_baseline
            aqi = max(0, min(500, round(25 / max(ratio, 0.01))))

        return AirQualityData(
            aqi=aqi,
            temperature=temperature,
            humidity=humidity,
            location=self.location,
            source="bme680",
        )
