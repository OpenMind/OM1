import asyncio
import json
import logging
import time
import psutil
from dataclasses import dataclass
from typing import List, Optional

from inputs.base import SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider

@dataclass
class Message:
    timestamp: float
    message: str

class SystemHealthInput(FuserInput[str]):
    """
    Monitors system vitals (CPU, RAM, Disk, Battery) using psutil.
    Acts as the robot's 'proprioception' (internal body awareness).
    """

    def __init__(self, config: SensorConfig = SensorConfig()):
        super().__init__(config)
        
        self.messages: List[Message] = []
        # Sensor name visible to the LLM
        self.descriptor_for_LLM = getattr(self.config, "input_name", "System Health Monitor")
        self.io_provider = IOProvider()
        
        # Polling interval (default: 5.0s)
        self.interval = getattr(self.config, "interval", 5.0)
        
        logging.info(f"[*] System Health Input initialized. Interval: {self.interval}s")

    async def _poll(self) -> Optional[str]:
        """
        Poll system statistics via psutil.
        """
        await asyncio.sleep(self.interval)

        try:
            # 1. CPU Usage
            cpu_pct = psutil.cpu_percent(interval=None)
            
            # 2. Memory Usage
            mem = psutil.virtual_memory()
            ram_pct = mem.percent
            
            # 3. Disk Usage (Root)
            disk = psutil.disk_usage('/')
            disk_pct = disk.percent
            
            # 4. Battery (if available)
            battery_info = "AC Power"
            battery_pct = 100
            try:
                bat = psutil.sensors_battery()
                if bat:
                    battery_pct = round(bat.percent, 1)
                    battery_info = "Charging" if bat.power_plugged else f"Discharging ({bat.secsleft/60:.0f} min left)"
            except Exception:
                pass 

            stats = {
                "cpu_usage_percent": cpu_pct,
                "ram_usage_percent": ram_pct,
                "disk_usage_percent": disk_pct,
                "power_status": battery_info,
                "battery_level": battery_pct
            }
            
            return json.dumps(stats)

        except Exception as e:
            logging.error(f"[-] Error reading system health: {e}")
            return None

    async def _raw_to_text(self, raw_input: str) -> Message:
        return Message(timestamp=time.time(), message=raw_input)

    async def raw_to_text(self, raw_input: Optional[str]):
        if raw_input is None:
            return

        pending_message = await self._raw_to_text(raw_input)
        if pending_message is not None:
            self.messages.append(pending_message)

    def formatted_latest_buffer(self) -> Optional[str]:
        if len(self.messages) == 0:
            return None

        latest_msg = self.messages[-1]
        
        result = f"""
INPUT: {self.descriptor_for_LLM}
// START
System Status Report (Internal Telemetry):
{latest_msg.message}
// END
"""
        self.io_provider.add_input(
            self.descriptor_for_LLM, latest_msg.message, time.time()
        )
        
        self.messages = []
        return result