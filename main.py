from src.plugins.sensors.temperature_plugin import TemperaturePlugin
from src.plugins.sensors.humidity_plugin import HumidityPlugin
from src.plugins.sensors.light_plugin import LightSensorPlugin
from src.plugins.sensors.air_quality_plugin import AirQualityPlugin
from src.plugins.sensors.sound_sensor_plugin import SoundSensorPlugin
import time
from os import system

temp = TemperaturePlugin()
hum = HumidityPlugin()
light = LightSensorPlugin()
air = AirQualityPlugin()
sound = SoundSensorPlugin()

snap = 0

while True:
    snap += 1
    # extract data
    t = temp.get_data()
    h = hum.get_data()
    l = light.get_data()
    a = air.get_data()
    s = sound.get_data()

    comfort_index = (
        t["comfort"],
        h["comfort"],
        l["description"],
        a["air_quality"],
        s["environment"]
    )

    # clear screen
    system("cls")   # use "clear" for Linux/Mac

    print(f"\n===== Environment Snapshot #{snap} =====\n")
    print(f"{'TEMP':<12}{'HUMIDITY':<12}{'LIGHT':<15}{'CO₂':<10}{'SOUND':<10}{'COMFORT INDEX'}")
    print("-" * 85)
    print(
        f"{t['temperature']:<12.1f}"
        f"{h['humidity']:<12.1f}"
        f"{l['lux']:<15.1f}"
        f"{a['co2_ppm']:<10}"
        f"{s['sound_db']:<10.1f}"
        f"{comfort_index}"
    )

    time.sleep(1)
