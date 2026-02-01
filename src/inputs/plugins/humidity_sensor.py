import asyncio
import logging
import time
from typing import Optional

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider


class HumiditySensorConfig(SensorConfig):
    sensor_type: str = "dht22"
    pin: Optional[int] = None
    i2c_address: Optional[str] = None
    update_interval: float = 2.0
    calibration_offset: float = 0.0


class HumiditySensor(FuserInput[HumiditySensorConfig, Optional[dict]]):
    """
    Humidity sensor input handler that provides real-time humidity readings.
    Supports DHT22 sensors with temperature and humidity data.
    """
    
    def __init__(self, config: HumiditySensorConfig):
        super().__init__(config)
        self.sensor_type = config.sensor_type.lower()
        self.pin = config.pin
        self.i2c_address = config.i2c_address
        self.update_interval = config.update_interval
        self.calibration_offset = config.calibration_offset
        self.io_provider = IOProvider()
        self.humidity_sensor = None
        self.last_reading_time = 0
        
        self._setup_sensor()
        self.descriptor_for_LLM = "Humidity and Temperature Environment Sensor"

    def _setup_sensor(self):
        """Setup DHT22 sensor."""
        try:
            import adafruit_dht
            self.humidity_sensor = adafruit_dht.DHT22(self.pin)
            logging.info(f"DHT22 humidity sensor initialized on pin {self.pin}")
        except ImportError:
            logging.error("adafruit_dht library not available for DHT22")
            self.humidity_sensor = None

    async def _poll(self) -> Optional[dict]:
        """Poll humidity sensor for new data."""
        if not self.humidity_sensor:
            return None
            
        current_time = time.time()
        if current_time - self.last_reading_time < self.update_interval:
            return None
            
        try:
            humidity = self.humidity_sensor.humidity
            temperature = self.humidity_sensor.temperature
            data = {
                "humidity": round(humidity + self.calibration_offset, 2),
                "temperature": round(temperature + self.calibration_offset, 2),
                "sensor_type": "DHT22",
                "timestamp": current_time
            }
        except Exception as e:
            logging.error(f"Error reading {self.sensor_type}: {e}")
            return None
            
        self.last_reading_time = current_time
        return data

    async def _raw_to_text(self, raw_input: dict) -> Optional[Message]:
        """Convert raw humidity data to text format."""
        if not raw_input:
            return None
        
        humidity = raw_input.get("humidity", "N/A")
        temperature = raw_input.get("temperature", "N/A")
        sensor_type = raw_input.get("sensor_type", "Unknown")
        
        if humidity == "N/A":
            return None
        
        temp_f = temperature * 9/5 + 32
        msg = (
            f"Current humidity: {humidity:.1f}%\n"
            f"Current temperature: {temperature:.1f}C ({temp_f:.1f}F)\n"
            f"Sensor type: {sensor_type}\n"
            f"Reading status: Good\n"
            f"Timestamp: {raw_input.get('timestamp', 'N/A')}"
        )
        
        return Message(timestamp=raw_input.get('timestamp', time.time()), message=msg)

    async def raw_to_text(self, raw_input: dict) -> Optional[Message]:
        return await self._raw_to_text(raw_input)

    async def formatted_latest_buffer(self) -> str:
        """Get latest sensor reading as formatted buffer."""
        latest_data = await self._poll()
        if latest_data:
            return await self._raw_to_text(latest_data)
        return "No recent humidity data available"

    async def _listen_loop(self):
        """Main listening loop for humidity sensor."""
        while True:
            try:
                data = await self._poll()
                if data:
                    message = await self._raw_to_text(data)
                    self.messages.append(message)
                    self.io_provider.output_message(message)
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                logging.error(f"Error in humidity sensor loop: {e}")
                await asyncio.sleep(5)
