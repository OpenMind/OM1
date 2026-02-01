import asyncio
import logging
import time
from typing import Optional

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider


class LightSensorConfig(SensorConfig):
    sensor_type: str = "bh1750"
    i2c_address: Optional[str] = None
    update_interval: float = 1.0
    calibration_offset: float = 0.0
    threshold: float = 100.0


class LightSensor(FuserInput[LightSensorConfig, Optional[dict]]):
    """
    Light sensor input handler using BH1750 digital light sensor.
    Provides ambient light level measurements in lux.
    """
    
    def __init__(self, config: LightSensorConfig):
        super().__init__(config)
        self.sensor_type = config.sensor_type.lower()
        self.i2c_address = config.i2c_address
        self.update_interval = config.update_interval
        self.calibration_offset = config.calibration_offset
        self.threshold = config.threshold
        self.io_provider = IOProvider()
        self.light_sensor = None
        self.last_reading_time = 0
        
        self._setup_sensor()
        self.descriptor_for_LLM = "Ambient Light Level Sensor"

    def _setup_sensor(self):
        """Setup BH1750 light sensor."""
        try:
            import board
            import adafruit_bh1750
            i2c = board.I2C()
            if self.i2c_address:
                self.light_sensor = adafruit_bh1750.BH1750(i2c, address=self.i2c_address)
            else:
                self.light_sensor = adafruit_bh1750.BH1750(i2c)
            logging.info(f"BH1750 light sensor initialized on I2C")
        except ImportError:
            logging.error("adafruit_bh1750 library not available for BH1750")
            self.light_sensor = None

    async def _poll(self) -> Optional[dict]:
        """Poll light sensor for new data."""
        if not self.light_sensor:
            return None
            
        current_time = time.time()
        if current_time - self.last_reading_time < self.update_interval:
            return None
            
        try:
            # BH1750 provides light level in lux
            lux = self.light_sensor.lux
            data = {
                "light_level_lux": round(lux + self.calibration_offset, 2),
                "light_status": "Bright" if lux > self.threshold else "Dim",
                "sensor_type": "BH1750",
                "timestamp": current_time
            }
        except Exception as e:
            logging.error(f"Error reading {self.sensor_type}: {e}")
            return None
            
        self.last_reading_time = current_time
        return data

    async def _raw_to_text(self, raw_input: dict) -> Optional[Message]:
        """Convert raw light data to text format."""
        if not raw_input:
            return None
        
        lux = raw_input.get("light_level_lux", "N/A")
        light_status = raw_input.get("light_status", "Unknown")
        sensor_type = raw_input.get("sensor_type", "Unknown")
        
        if lux == "N/A":
            return None
        
        msg = (
            f"Current light level: {lux:.1f} lux\n"
            f"Light status: {light_status}\n"
            f"Sensor type: {sensor_type}\n"
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
        return "No recent light data available"

    async def _listen_loop(self):
        """Main listening loop for light sensor."""
        while True:
            try:
                data = await self._poll()
                if data:
                    message = await self._raw_to_text(data)
                    self.messages.append(message)
                    self.io_provider.output_message(message)
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                logging.error(f"Error in light sensor loop: {e}")
                await asyncio.sleep(5)
