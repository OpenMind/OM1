#!/usr/bin/env python3
"""
OM1 Sensor Plugin Demo - Creates visual demonstration of new sensor capabilities
Records screen while showing real-time sensor data from multiple sensors
"""

import asyncio
import time
import os
from datetime import datetime
from typing import List, Dict, Any

from inputs.plugins.temperature_sensor import TemperatureSensor, TemperatureSensorConfig
from inputs.plugins.humidity_sensor import HumiditySensor, HumiditySensorConfig  
from inputs.plugins.light_sensor import LightSensor, LightSensorConfig
from inputs.plugins.air_quality_sensor import AirQualitySensor, AirQualitySensorConfig


class SensorDemo:
    """Demo class for showcasing OM1 sensor plugins."""
    
    def __init__(self):
        self.running = False
        self.start_time = None
        self.sensor_data = {
            'temperature': [],
            'humidity': [],
            'light': [],
            'air_quality': []
        }
    
    def print_header(self):
        """Print demo header."""
        print("\n" + "="*60)
        print("🤖 OM1 SENSOR PLUGIN DEMO 🤖")
        print("="*60)
        print("📊 Real-time Environmental Monitoring System\n")
        print(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
    
    def print_sensor_status(self, sensor_name: str, data: Dict[str, Any]):
        """Print formatted sensor data."""
        print(f"\n📋 {sensor_name.upper()} SENSOR READING:")
        print("-" * 30)
        
        for key, value in data.items():
            if isinstance(value, (int, float)):
                if 'temperature' in key.lower() or 'temp' in key.lower():
                    print(f"🌡 {key.replace('_', ' ').title()}: {value}°C")
                elif 'humidity' in key.lower():
                    print(f"💧 {key.replace('_', ' ').title()}: {value}%")
                elif 'lux' in key.lower() or 'light' in key.lower():
                    print(f"💡 {key.replace('_', ' ').title()}: {value} lux")
                elif 'co2' in key.lower():
                    print(f"🫧 {key.replace('_', ' ').title()}: {value} ppm")
                elif 'voc' in key.lower():
                    print(f"🌬 {key.replace('_', ' ').title()}: {value}")
                elif 'pm25' in key.lower():
                    print(f"🫠 {key.replace('_', ' ').title()}: {value} μg/m³")
                elif 'air_quality' in key.lower():
                    print(f"🌤 {key.replace('_', ' ').title()}: {value}")
                else:
                    print(f"📊 {key.replace('_', ' ').title()}: {value}")
            else:
                print(f"📋 {key.replace('_', ' ').title()}: {value}")
        
        print("-" * 30)
        timestamp = data.get('timestamp', time.time())
        print(f"⏰ Timestamp: {datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')}")
    
    async def run_demo(self, duration_minutes: int = 30):
        """Run the sensor demo for specified duration."""
        self.running = True
        self.start_time = time.time()
        duration_seconds = duration_minutes * 60
        
        print(f"\n🎬 DEMO MODE: {duration_minutes} minutes")
        print("📹 Recording sensor data...")
        print("⏹ Press Ctrl+C to stop early\n")
        
        # Initialize sensors (simulation mode)
        sensors = []
        
        # Temperature sensor
        temp_config = TemperatureSensorConfig(
            sensor_type="dht22",
            pin=4,
            update_interval=2.0,
            calibration_offset=0.0
        )
        temp_sensor = TemperatureSensor(temp_config)
        sensors.append(('Temperature', temp_sensor))
        
        # Humidity sensor  
        humidity_config = HumiditySensorConfig(
            sensor_type="dht22",
            pin=4, 
            update_interval=2.0,
            calibration_offset=0.0
        )
        humidity_sensor = HumiditySensor(humidity_config)
        sensors.append(('Humidity', humidity_sensor))
        
        # Light sensor
        light_config = LightSensorConfig(
            sensor_type="bh1750",
            update_interval=1.0,
            threshold=100.0
        )
        light_sensor = LightSensor(light_config)
        sensors.append(('Light', light_sensor))
        
        # Air quality sensor
        air_config = AirQualitySensorConfig(
            sensor_type="sht30",
            update_interval=5.0,
            calibration_offset={"co2": 0.0, "voc": 0.0, "pm25": 0.0}
        )
        air_sensor = AirQualitySensor(air_config)
        sensors.append(('Air Quality', air_sensor))
        
        print("\n🔧 All sensors initialized (simulation mode)")
        
        try:
            # Run demo for specified duration
            while self.running and (time.time() - self.start_time) < duration_seconds:
                iteration_start = time.time()
                
                # Collect data from all sensors
                for sensor_name, sensor in sensors:
                    try:
                        data = await sensor._poll()
                        if data:
                            self.sensor_data[sensor_name.lower()].append(data)
                            self.print_sensor_status(sensor_name, data)
                    except Exception as e:
                        print(f"❌ {sensor_name} sensor error: {e}")
                
                # Update display every 2 seconds
                if time.time() - self.start_time > 0:
                    print("\n" + "-"*40)
                    print(f"🕐 Elapsed: {int((time.time() - self.start_time) / 60)}:{int((time.time() - self.start_time) % 60)} | 📊 Total Readings: {sum(len(readings) for readings in self.sensor_data.values())}")
                    print("-"*40)
                
                await asyncio.sleep(2)
        
        except KeyboardInterrupt:
            print("\n\n🛑 Demo stopped by user\n")
        
        finally:
            self.running = False
            self.print_summary()
    
    def print_summary(self):
        """Print demo summary."""
        duration = time.time() - self.start_time
        
        print("\n" + "="*60)
        print("📈 DEMO SUMMARY 📈")
        print("="*60)
        print(f"🕐 Total Duration: {int(duration // 60)}:{int(duration % 60)} minutes")
        
        for sensor_name, readings in self.sensor_data.items():
            if readings:
                print(f"\n📋 {sensor_name.title()} Sensor:")
                print(f"   📊 Total Readings: {len(readings)}")
                
                # Calculate some stats
                if readings and 'temperature' in readings[0]:
                    temps = [r['temperature'] for r in readings if 'temperature' in r]
                    if temps:
                        print(f"   🌡 Temp Range: {min(temps):.1f}°C - {max(temps):.1f}°C")
                        print(f"   🌡 Avg Temp: {sum(temps)/len(temps):.1f}°C")
                
                if readings and 'humidity' in readings[0]:
                    humidities = [r['humidity'] for r in readings if 'humidity' in r]
                    if humidities:
                        print(f"   💧 Humidity Range: {min(humidities):.1f}% - {max(humidities):.1f}%")
                        print(f"   💧 Avg Humidity: {sum(humidities)/len(humidities):.1f}%")
        
        print("="*60)
        print("\n🎬 Ready for bounty submission!")
        print("💡 Demo script records sensor data in terminal format for video recording")


async def main():
    """Main demo function."""
    demo = SensorDemo()
    
    if len(os.sys.argv) > 1:
        try:
            duration = int(os.sys.argv[1])
        except ValueError:
            duration = 30
    else:
        duration = 30
    
    demo.print_header()
    await demo.run_demo(duration_minutes=duration)


if __name__ == "__main__":
    asyncio.run(main())
