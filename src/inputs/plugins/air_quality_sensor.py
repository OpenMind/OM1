"""
Air Quality Sensor Plugin for OM1 - Simulation Version

Provides simulated air quality readings (CO2, VOC, PM2.5) without requiring hardware sensors.
"""

import asyncio
import logging
import math
import time
from typing import Optional

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider


class AirQualitySensorConfig(SensorConfig):
    sensor_type: str = "simulation"
    i2c_address: Optional[str] = None
    update_interval: float = 5.0
    calibration_offset: dict = {"co2": 0.0, "voc": 0.0, "pm25": 0.0}


class AirQualitySensor(FuserInput[AirQualitySensorConfig, Optional[dict]]):
    """
    Simulated air quality sensor for OM1 testing.
    Provides realistic CO2, VOC, and PM2.5 measurements without hardware requirements.
    """
    
    def __init__(self, config: AirQualitySensorConfig):
        super().__init__(config)
        self.sensor_type = config.sensor_type.lower()
        self.i2c_address = config.i2c_address
        self.update_interval = config.update_interval
        self.calibration_offset = config.calibration_offset
        self.io_provider = IOProvider()
        self.last_reading_time = 0
        
        self.descriptor_for_LLM = "Simulated Air Quality and Multi-Pollutant Environment Sensor"

    async def _poll(self) -> Optional[dict]:
        """
        Poll simulated air quality sensor with realistic variations.
        """
        current_time = time.time()
        if current_time - self.last_reading_time < self.update_interval:
            return None
            
        # Simulate realistic air quality variations
        elapsed = current_time - self.last_reading_time
        
        # Base values for typical indoor environment
        base_co2 = 450.0  # ppm
        base_voc = 100.0  # index
        base_pm25 = 12.0  # µg/m³
        
        # Add realistic variations
        co2_variation = math.sin(elapsed * 0.2) * 50 + math.cos(elapsed * 0.1) * 30
        voc_variation = math.sin(elapsed * 0.3) * 30 + math.cos(elapsed * 0.15) * 20
        pm25_variation = math.sin(elapsed * 0.25) * 5 + math.cos(elapsed * 0.12) * 3
        
        co2 = base_co2 + co2_variation
        voc = base_voc + voc_variation
        pm25 = base_pm25 + pm25_variation
        
        # Apply calibration offsets
        co2_calibrated = max(400, co2 + self.calibration_offset.get("co2", 0.0))
        voc_calibrated = max(0, voc + self.calibration_offset.get("voc", 0.0))
        pm25_calibrated = max(0, pm25 + self.calibration_offset.get("pm25", 0.0))
        
        # Simulate temperature and humidity (typical for air quality monitoring)
        temperature = 22.0 + math.sin(elapsed * 0.1) * 2
        humidity = 50.0 + math.cos(elapsed * 0.15) * 10
        
        data = {
            "co2_ppm": round(co2_calibrated, 2),
            "voc_index": round(voc_calibrated, 2),
            "pm25_ug_m3": round(pm25_calibrated, 2),
            "temperature_celsius": round(temperature, 2),
            "relative_humidity_percent": round(humidity, 2),
            "air_quality_level": self._calculate_air_quality(co2_calibrated, voc_calibrated, pm25_calibrated),
            "sensor_type": "AIR_QUALITY_SIM",
            "timestamp": current_time
        }
            
        self.last_reading_time = current_time
        return data

    def _calculate_air_quality(self, co2, voc, pm25):
        """
        Calculate overall air quality level based on pollutant levels.
        """
        # Normalize each pollutant to 0-1 scale based on health guidelines
        co2_score = min(co2 / 1000, 1.0)  # 1000ppm = moderate concern
        voc_score = min(voc / 200, 1.0)   # 200 = moderate
        pm25_score = min(pm25 / 35, 1.0)   # 35 µg/m³ = moderate (EPA standard)
        
        # Calculate average score
        overall_score = (co2_score + voc_score + pm25_score) / 3
        
        if overall_score < 0.3:
            return "Good"
        elif overall_score < 0.6:
            return "Moderate"
        elif overall_score < 0.8:
            return "Poor"
        else:
            return "Hazardous"

    async def _raw_to_text(self, raw_input: dict) -> Optional[Message]:
        """
        Convert raw air quality data to text format.
        """
        if not raw_input:
            return None
        
        co2 = raw_input.get("co2_ppm", "N/A")
        voc = raw_input.get("voc_index", "N/A")
        pm25 = raw_input.get("pm25_ug_m3", "N/A")
        air_quality = raw_input.get("air_quality_level", "Unknown")
        sensor_type = raw_input.get("sensor_type", "Unknown")
        
        if co2 == "N/A":
            return None
        
        msg = (
            f"🌬 Air Quality Status: {air_quality}\n"
            f"💨 CO2: {co2} ppm\n"
            f"🧪 VOC Index: {voc}\n"
            f"🌫 PM2.5: {pm25} µg/m³\n"
            f"🌡 Temperature: {raw_input.get('temperature_celsius', 'N/A')}°C\n"
            f"💧 Humidity: {raw_input.get('relative_humidity_percent', 'N/A')}%\n"
            f"📡 Sensor type: {sensor_type}\n"
            f"✅ Simulation mode active\n"
            f"Timestamp: {raw_input.get('timestamp', 'N/A')}"
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
        return "No recent air quality data available (simulation mode)"

    async def _listen_loop(self):
        """
        Main listening loop for simulated air quality sensor.
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
                logging.error(f"Error in air quality sensor loop: {e}")
                await asyncio.sleep(5)
