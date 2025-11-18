from src.plugins.hue.hue_plugin import HuePlugin
import time

p = HuePlugin()
print("Bounty #366 – Philips Hue Demo (mock mode)\n")
for i in range(10):
    data = p.get_data()
    print(f"[{i+1:2d}] Lights: {len(data['lights'])} | Groups: {len(data['groups'])} | Scene: relax | On: {sum(1 for l in data['lights'].values() if l['on'])}")
    time.sleep(2)
