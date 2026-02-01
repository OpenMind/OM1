"""
Temperature Sensor Plugin for OM1 - Simulation Version

Provides simulated temperature readings without requiring hardware sensors.
"""

import asyncio
import logging
import math
import time
from typing import Optional

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider


class TemperatureSensorConfig(SensorConfig):
    sensor_type: str = "simulation"
    pin: Optional[int] = None
    i2c_address: Optional[str] = None
    update_interval: float = 2.0
    calibration_offset: float = 0.0


class TemperatureSensor(FuserInput[TemperatureSensorConfig, Optional[dict]]):
    """
    Simulated temperature sensor for OM1 testing.
    Provides realistic temperature variations without hardware requirements.
    """
    
    def __init__(self, config: TemperatureSensorConfig):
        super().__init__(config)
        
        self.sensor_type = config.sensor_type.lower()
        self.update_interval = config.update_interval
        self.calibration_offset = config.calibration_offset
        self.io_provider = IOProvider()
        self.temp_value = 20.0
        self.humidity_value = 50.0
        self.last_reading_time = 0
        
        self.descriptor_for_LLM = "Simulated Temperature and Humidity Environment Sensor"

    async def _poll(self) -> Optional[dict]:
        """
        Poll simulated temperature sensor with realistic variations.
        """
        
        current_time = time.time()
        
        # Throttle polling based on update_interval
        if current_time - self.last_reading_time < self.update_interval:
            return None
        
        # Simulate realistic temperature variations
        elapsed = current_time - self.last_reading_time
        temp_variation = math.sin(elapsed * 0.5) * 3  # Sinusoidal variation
        humidity_variation = math.cos(elapsed * 0.3) * 10  # Humidity variation
        base_temp = 22.0
        
        temperature = round(base_temp + temp_variation + self.calibration_offset, 1)
        humidity = max(30, min(80, round(50.0 + humidity_variation)))
        
        data = {
            "temperature": temperature,
            "humidity": humidity,
            "sensor_type": "DHT22_SIM",
            "timestamp": current_time
        }
        
        self.temp_value = temperature
        self.humidity_value = humidity
        self.last_reading_time = current_time
        
        return data

    async def _raw_to_text(self, raw_input: dict) -> Optional[Message]:
        """
        Convert simulated sensor data to text format.
        """
        
        if not raw_input:
            return None
        
        temperature = raw_input.get("temperature", "N/A")
        humidity = raw_input.get("humidity", "N/A")
        sensor_type = raw_input.get("sensor_type", "Unknown")
        
        if temperature == "N/A":
            return None
        
        temp_f = temperature * 9/5 + 32
        msg = (
            f"🌡 Current temperature: {temperature:.1f}°C ({temp_f:.1f}°F)\n"
            f"💧 Current humidity: {humidity:.1f}%\n"
            f"🌡 Sensor type: {sensor_type}\n"
            f"✅ Simulation mode active\n"
            f"Timestamp: {raw_input.get('timestamp', time.time())}"
        )
        
        return Message(timestamp=raw_input.get('timestamp', time.time()), message=msg)

    async def raw_to_text(self, raw_input: dict) -> Optional[Message]:
        return await self._raw_to_text(raw_input)

    async def formatted_latest_buffer(self) -> str:
        """
        Get latest sensor reading as formatted buffer.
        """
        
        latest_data = await self._poll()
        if latest_data:
            return await self._raw_to_text(latest_data)
        return "No recent temperature data available (simulation mode)"

    async def _listen_loop(self):
        """
        Main listening loop for simulated temperature sensor.
        """
        
        while True:
            try:
                data = await self._poll()
                if data:
                    message = await self._raw_to_text(data)
                    self.messages.append(message)
                    self.io_provider.output_message(message)
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                logging.error(f"Error in temperature sensor loop: {e}")
                await asyncio.sleep(5)
