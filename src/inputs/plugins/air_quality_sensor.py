import asyncio
import logging
import time
from typing import Optional, Dict, Any

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider


class AirQualitySensorConfig(SensorConfig):
    sensor_type: str = "sht30"
    i2c_address: Optional[str] = None
    update_interval: float = 5.0
    calibration_offset: Dict[str, float] = {"co2": 0.0, "voc": 0.0, "pm25": 0.0}


class AirQualitySensor(FuserInput[AirQualitySensorConfig, Optional[Dict[str, Any]]):
    """
    Air quality sensor input handler using SHT30 multi-pollutant sensor.
    Provides CO2, VOC, PM2.5, temperature, and humidity measurements.
    """
    
    def __init__(self, config: AirQualitySensorConfig):
        super().__init__(config)
        self.sensor_type = config.sensor_type.lower()
        self.i2c_address = config.i2c_address
        self.update_interval = config.update_interval
        self.calibration_offset = config.calibration_offset
        self.io_provider = IOProvider()
        self.air_sensor = None
        self.last_reading_time = 0
        
        self._setup_sensor()
        self.descriptor_for_LLM = "Air Quality and Multi-Pollutant Environment Sensor"

    def _setup_sensor(self):
        """Setup SHT30 air quality sensor."""
        try:
            import board
            import adafruit_sht30
            self.air_sensor = adafruit_sht30.SHT30(self.i2c_address)
            logging.info(f"SHT30 air quality sensor initialized on I2C address {self.i2c_address}")
        except ImportError:
            logging.error("adafruit_sht30 library not available for SHT30")
            self.air_sensor = None

    async def _poll(self) -> Optional[Dict[str, Any]]:
        """Poll air quality sensor for new data."""
        if not self.air_sensor:
            return None
            
        current_time = time.time()
        if current_time - self.last_reading_time < self.update_interval:
            return None
            
        try:
            # SHT30 provides multiple measurements
            co2 = self.air_sensor.co2_eq_ppm
            voc = self.air_sensor.voc_index
            pm25 = self.air_sensor.pm25_ug_m3
            temperature = self.air_sensor.temperature
            humidity = self.air_sensor.relative_humidity
            
            # Apply calibration offsets
            co2_calibrated = max(0, co2 + self.calibration_offset["co2"])
            voc_calibrated = max(0, voc + self.calibration_offset["voc"])
            pm25_calibrated = max(0, pm25 + self.calibration_offset["pm25"])
            
            data = {
                "co2_ppm": round(co2_calibrated, 2),
                "voc_index": round(voc_calibrated, 2),
                "pm25_ug_m3": round(pm25_calibrated, 2),
                "temperature_celsius": round(temperature + self.calibration_offset.get("temperature", 0.0), 2),
                "relative_humidity_percent": round(humidity + self.calibration_offset.get("humidity", 0.0), 2),
                "air_quality_level": self._calculate_air_quality(co2_calibrated, voc_calibrated, pm25_calibrated),
                "sensor_type": "SHT30",
                "timestamp": current_time
            }
        except Exception as e:
            logging.error(f"Error reading {self.sensor_type}: {e}")
            return None
            
        self.last_reading_time = current_time
        return data

    def _calculate_air_quality(self, co2, voc, pm25):
        """Calculate overall air quality level based on pollutant levels."""
        # Simple air quality calculation (can be enhanced with more sophisticated algorithms)
        co2_score = min(co2 / 400, 1.0)  # Normalize CO2 (400ppm = moderate)
        voc_score = min(voc / 200, 1.0)   # Normalize VOC (200 = moderate)
        pm25_score = min(pm25 / 35, 1.0)   # Normalize PM2.5 (35 ug/m3 = moderate)
        
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

    async def _raw_to_text(self, raw_input: Dict[str, Any]) -> Optional[Message]:
        """Convert raw air quality data to text format."""
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
            f"Air Quality Status: {air_quality}\n"
            f"CO2: {co2} ppm\n"
            f"VOC Index: {voc}\n"
            f"PM2.5: {pm25} ug/m³\n"
            f"Temperature: {raw_input.get('temperature_celsius', 'N/A')}°C\n"
            f"Humidity: {raw_input.get('relative_humidity_percent', 'N/A')}%\n"
            f"Sensor type: {sensor_type}\n"
            f"Timestamp: {raw_input.get('timestamp', 'N/A')}"
        )
        
        return Message(timestamp=raw_input.get('timestamp', time.time()), message=msg)

    async def raw_to_text(self, raw_input: Dict[str, Any]) -> Optional[Message]:
        return await self._raw_to_text(raw_input)

    async def formatted_latest_buffer(self) -> str:
        """Get latest sensor reading as formatted buffer."""
        latest_data = await self._poll()
        if latest_data:
            return await self._raw_to_text(latest_data)
        return "No recent air quality data available"

    async def _listen_loop(self):
        """Main listening loop for air quality sensor."""
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
