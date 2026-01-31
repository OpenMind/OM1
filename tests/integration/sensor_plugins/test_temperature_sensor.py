import asyncio
import pytest
import time
from unittest.mock import MagicMock, patch
from typing import Dict, Any

from inputs.plugins.temperature_sensor import TemperatureSensor, TemperatureSensorConfig


class TestTemperatureSensor:
    """Test suite for TemperatureSensor plugin."""
    
    @pytest.fixture
    def config(self):
        return TemperatureSensorConfig(
            sensor_type="dht22",
            pin=4,
            update_interval=1.0,
            calibration_offset=0.5
        )
    
    @pytest.mark.asyncio
    async def test_dht22_reading(self, config):
        """Test DHT22 sensor reading functionality."""
        with patch('adafruit_dht.DHT22') as mock_dht22:
            # Mock sensor instance
            mock_instance = MagicMock()
            mock_instance.temperature = 25.0
            mock_instance.humidity = 60.0
            
            mock_dht22.return_value = mock_instance
            
            sensor = TemperatureSensor(config)
            
            # Test successful reading
            data = await sensor._poll()
            
            assert data is not None
            assert data["temperature"] == 25.5  # 25.0 + 0.5 calibration
            assert data["humidity"] == 60.0
            assert data["sensor_type"] == "DHT22"
            assert "timestamp" in data
            
            # Test message formatting
            message = await sensor._raw_to_text(data)
            assert "25.5C" in message
            assert "60.0%" in message
            assert "DHT22" in message
    
    @pytest.mark.asyncio
    async def test_ds18b20_reading(self, config):
        """Test DS18B20 sensor reading functionality."""
        with patch('adafruit_ds18x20.DS18X20') as mock_ds18b20:
            # Mock sensor instance
            mock_instance = MagicMock()
            mock_instance.temperature = 22.0
            mock_instance.humidity = 55.0
            
            mock_ds18b20.return_value = mock_instance
            
            sensor = TemperatureSensor(config)
            sensor.sensor_type = "ds18b20"  # Override for test
            
            data = await sensor._poll()
            
            assert data is not None
            assert data["temperature"] == 22.5  # 22.0 + 0.5 calibration
            assert data["sensor_type"] == "DS18B20"
            
            # Test formatted latest buffer
            buffer = await sensor.formatted_latest_buffer()
            assert buffer is not None
            assert "22.5C" in buffer
    
    @pytest.mark.asyncio
    async def test_sensor_error_handling(self, config):
        """Test sensor error handling."""
        with patch('adafruit_dht.DHT22') as mock_dht22:
            mock_dht22.side_effect = Exception("Sensor error")
            
            sensor = TemperatureSensor(config)
            
            # Should handle gracefully
            data = await sensor._poll()
            assert data is None
            
            # Test formatted buffer returns appropriate message
            buffer = await sensor.formatted_latest_buffer()
            assert buffer == "No recent temperature data available"
    
    @pytest.mark.asyncio
    async def test_message_buffering(self, config):
        """Test message buffering functionality."""
        with patch('adafruit_dht.DHT22') as mock_dht22:
            mock_instance = MagicMock()
            mock_instance.temperature = 20.0
            mock_instance.humidity = 50.0
            
            mock_dht22.return_value = mock_instance
            
            sensor = TemperatureSensor(config)
            
            # Generate multiple readings
            readings = []
            for i in range(3):
                data = {
                    "temperature": 20.0 + i,
                    "humidity": 50.0 + i,
                    "sensor_type": "DHT22",
                    "timestamp": time.time() + i
                }
                readings.append(data)
            
            # Test that messages are properly buffered
            for reading in readings:
                message = await sensor._raw_to_text(reading)
                sensor.messages.append(message)
            
            assert len(sensor.messages) == 3
            
            # Test formatted buffer contains latest data
            buffer = await sensor.formatted_latest_buffer()
            assert "22.0C" in buffer  # Latest reading
