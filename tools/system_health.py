import psutil
import platform
from datetime import datetime

def system_health_check():
    print(f"OM1 System Health - {datetime.now()}")
    print(f"System: {platform.system()} {platform.release()}")
    print(f"CPU Usage: {psutil.cpu_percent()}%")
    try:
        battery = psutil.sensors_battery()
        print(f"Battery: {battery.percent}%" if battery else "No Battery")
    except:
        print("Battery info unavailable")

if __name__ == "__main__":
    system_health_check()