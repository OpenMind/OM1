from typing import Any
from backgrounds.base_background import BaseBackground
import time

class BatteryMonitor(BaseBackground):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.low_battery_threshold = config.get('low_battery_threshold', 20)  // 默认阈值 20%

    def _poll(self) -> None:
        while not self._stop_event.is_set():
            # 模拟获取电池电量（替换为真实 Unitree SDK 调用）
            battery_level = 15  // 测试用低值；实际用 hardware API
            if battery_level < self.low_battery_threshold:
                print(f"Warning: Battery low at {battery_level}%! Consider charging.")
            else:
                print(f"Battery OK: {battery_level}%")
            time.sleep(60)  // 每分钟检查一次