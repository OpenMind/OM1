#!/usr/bin/env python3
"""
Setup script for testing new sensor plugins with OM1.
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from inputs.plugins.temperature_sensor import TemperatureSensor, TemperatureSensorConfig
from inputs.plugins.humidity_sensor import HumiditySensor, HumiditySensorConfig
from inputs.plugins.light_sensor import LightSensor, LightSensorConfig
from inputs.plugins.air_quality_sensor import AirQualitySensor, AirQualitySensorConfig


async def test_temperature_sensor():
    """Test temperature sensor plugin."""
    print("Testing Temperature Sensor...")
    
    config = TemperatureSensorConfig(
        sensor_type="dht22",
        pin=4,
        update_interval=2.0,
        calibration_offset=0.0
    )
    
    sensor = TemperatureSensor(config)
    
    try:
        # Start the sensor
        async for message in sensor._listen_loop():
            print(f"Temperature: {message.message}")
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\nTemperature sensor test stopped.")
    except Exception as e:
        print(f"Error: {e}")


async def test_humidity_sensor():
    """Test humidity sensor plugin."""
    print("Testing Humidity Sensor...")
    
    config = HumiditySensorConfig(
        sensor_type="dht22",
        pin=4,
        update_interval=2.0,
        calibration_offset=0.0
    )
    
    sensor = HumiditySensor(config)
    
    try:
        # Start the sensor
        async for message in sensor._listen_loop():
            print(f"Humidity: {message.message}")
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\nHumidity sensor test stopped.")
    except Exception as e:
        print(f"Error: {e}")


async def test_light_sensor():
    """Test light sensor plugin."""
    print("Testing Light Sensor...")
    
    config = LightSensorConfig(
        sensor_type="bh1750",
        i2c_address=0x23,
        update_interval=1.0,
        threshold=100.0
    )
    
    sensor = LightSensor(config)
    
    try:
        # Start the sensor
        async for message in sensor._listen_loop():
            print(f"Light: {message.message}")
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\nLight sensor test stopped.")
    except Exception as e:
        print(f"Error: {e}")


async def test_air_quality_sensor():
    """Test air quality sensor plugin."""
    print("Testing Air Quality Sensor...")
    
    config = AirQualitySensorConfig(
        sensor_type="sht30",
        i2c_address=0x44,
        update_interval=5.0,
        calibration_offset={"co2": 0.0, "voc": 0.0, "pm25": 0.0}
    )
    
    sensor = AirQualitySensor(config)
    
    try:
        # Start the sensor
        async for message in sensor._listen_loop():
            print(f"Air Quality: {message.message}")
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\nAir quality sensor test stopped.")
    except Exception as e:
        print(f"Error: {e}")


async def main():
    """Main function to test all sensor plugins."""
    print("OM1 Sensor Plugin Test Suite")
    print("============================")
    
    if len(sys.argv) > 1:
        sensor_type = sys.argv[1].lower()
        
        if sensor_type == "temperature":
            await test_temperature_sensor()
        elif sensor_type == "humidity":
            await test_humidity_sensor()
        elif sensor_type == "light":
            await test_light_sensor()
        elif sensor_type == "air_quality":
            await test_air_quality_sensor()
        else:
            print(f"Unknown sensor type: {sensor_type}")
            print("Available: temperature, humidity, light, air_quality")
    else:
        print("Testing all sensors sequentially...")
        await test_temperature_sensor()
        await asyncio.sleep(2)
        await test_humidity_sensor()
        await asyncio.sleep(2)
        await test_light_sensor()
        await asyncio.sleep(2)
        await test_air_quality_sensor()


if __name__ == "__main__":
    asyncio.run(main())
