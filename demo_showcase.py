#!/usr/bin/env python3
"""
Standalone Demo Showcase for OM1 Sensor Plugins
Shows what the sensor plugins would look like when running
"""

import time
import random
from datetime import datetime


class SensorSimulator:
    """Simulates sensor readings for demonstration."""
    
    def __init__(self, sensor_type, color_emoji, unit):
        self.sensor_type = sensor_type
        self.color_emoji = color_emoji
        self.unit = unit
        self.readings = []
        self.start_time = time.time()
    
    def get_reading(self):
        """Generate realistic sensor reading."""
        elapsed = time.time() - self.start_time
        
        if self.sensor_type == "temperature":
            # Simulate temperature with small variations
            base_temp = 22.0
            variation = random.uniform(-2, 3)
            temperature = base_temp + variation + (elapsed * 0.1)
            return {
                "temperature": round(temperature, 1),
                "sensor_type": "DHT22",
                "timestamp": time.time()
            }
        
        elif self.sensor_type == "humidity":
            # Simulate humidity readings
            base_humidity = 55.0
            variation = random.uniform(-5, 8)
            humidity = max(20, min(80, base_humidity + variation + (elapsed * 0.2)))
            return {
                "humidity": round(humidity, 1),
                "temperature": round(22.0 + random.uniform(-1, 2), 1),
                "sensor_type": "DHT22",
                "timestamp": time.time()
            }
        
        elif self.sensor_type == "light":
            # Simulate light sensor with day/night cycle
            base_light = 150
            cycle = int(elapsed / 10) % 2
            variation = random.uniform(-20, 50)
            lux = base_light + variation + (cycle * 100)
            return {
                "light_level_lux": round(max(10, lux), 0),
                "light_status": "Bright" if lux > 100 else "Dim",
                "sensor_type": "BH1750",
                "timestamp": time.time()
            }
        
        elif self.sensor_type == "air_quality":
            # Simulate air quality readings
            base_co2 = 400
            base_voc = 50
            base_pm25 = 15
            
            co2 = base_co2 + random.uniform(-50, 100)
            voc = base_voc + random.uniform(-10, 30)
            pm25 = base_pm25 + random.uniform(-5, 15)
            
            # Calculate air quality level
            avg_normalized = ((co2/400) + (voc/200) + (pm25/35)) / 3
            
            if avg_normalized < 0.3:
                air_quality = "Good"
            elif avg_normalized < 0.6:
                air_quality = "Moderate"
            elif avg_normalized < 0.8:
                air_quality = "Poor"
            else:
                air_quality = "Hazardous"
            
            return {
                "co2_ppm": round(co2, 0),
                "voc_index": round(voc, 0),
                "pm25_ug_m3": round(pm25, 0),
                "air_quality_level": air_quality,
                "sensor_type": "SHT30",
                "timestamp": time.time()
            }


def print_header():
    """Print demo header."""
    print("\n" + "="*80)
    print("🤖 OM1 SENSOR PLUGIN DEMO SHOWCASE 🤖")
    print("="*80)
    print("📊 Real-time Environmental Monitoring System")
    print(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print("🎬 Simulating 4 New Sensor Plugins for Bounty #365")
    print("="*80)


def print_sensor_data(sensor, data):
    """Print formatted sensor data."""
    print(f"\n{sensor.color_emoji} {sensor.sensor_type.upper()} SENSOR:")
    print("-" * 40)
    
    if sensor.sensor_type == "temperature":
        temp = data.get('temperature', 0)
        temp_f = temp * 9/5 + 32
        print(f"🌡 Temperature: {temp}°C ({temp_f}°F)")
    
    elif sensor.sensor_type == "humidity":
        humidity = data.get('humidity', 0)
        temp = data.get('temperature', 0)
        temp_f = temp * 9/5 + 32
        print(f"💧 Humidity: {humidity}%")
        print(f"🌡 Temperature: {temp}°C ({temp_f}°F)")
    
    elif sensor.sensor_type == "light":
        lux = data.get('light_level_lux', 0)
        status = data.get('light_status', 'Unknown')
        print(f"💡 Light Level: {lux} lux ({status})")
    
    elif sensor.sensor_type == "air_quality":
        co2 = data.get('co2_ppm', 0)
        voc = data.get('voc_index', 0)
        pm25 = data.get('pm25_ug_m3', 0)
        quality = data.get('air_quality_level', 'Unknown')
        print(f"🫧 CO2: {co2} ppm")
        print(f"🌬 VOC Index: {voc}")
        print(f"🫠 PM2.5: {pm25} μg/m³")
        print(f"🌤 Air Quality: {quality}")
    
    print("-" * 40)
    timestamp = data.get('timestamp', time.time())
    print(f"⏰ Time: {datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')}")
    print("="*80)


def main():
    """Main demo function."""
    print_header()
    
    # Create sensors
    sensors = [
        SensorSimulator("temperature", "🌡", "°C"),
        SensorSimulator("humidity", "💧", "%"),
        SensorSimulator("light", "💡", "lux"),
        SensorSimulator("air_quality", "🌤", "")
    ]
    
    print("\n🔄 Starting real-time simulation (30 seconds)...")
    print("⏹ Press Ctrl+C to stop early\n")
    
    try:
        start_time = time.time()
        readings_count = 0
        
        while time.time() - start_time < 30:
            iteration_start = time.time()
            
            print("\n" + "-"*60)
            print(f"🕐 Elapsed: {int((time.time() - start_time) / 60)}:{int((time.time() - start_time) % 60)} | 📊 Total Readings: {readings_count}")
            print("-"*60)
            
            # Show data from all sensors
            for sensor in sensors:
                data = sensor.get_reading()
                print_sensor_data(sensor, data)
                readings_count += 1
            
            # Update every 2 seconds
            elapsed = time.time() - iteration_start
            if elapsed < 2:
                sleep_time = 2.0 - elapsed
                time.sleep(sleep_time)
        
        print("\n" + "="*80)
        print("📈 SIMULATION SUMMARY 📈")
        print("="*80)
        print(f"🕐 Total Duration: 30 seconds")
        print(f"📊 Total Readings: {readings_count}")
        print(f"🎯 Demo Success: All sensor simulations completed")
        print("="*80)
        print("\n✅ Ready for OM1 integration and bounty submission!")
        
    except KeyboardInterrupt:
        print("\n\n🛑 Demo stopped by user\n")
    
    print("\n💡 This demonstrates the sensor plugin capabilities:")
    print("   🌡 Temperature sensor with calibration support")
    print("   💧 Humidity sensor with DHT22 integration")
    print("   💡 Light sensor with lux measurements")
    print("   🌤 Air quality sensor with multi-pollutant detection")
    print("\n🎯 All ready for integration with OM1!")


if __name__ == "__main__":
    main()
