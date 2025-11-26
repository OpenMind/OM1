# demo.py
from om1_isaac.bridge import OM1IsaacGymBridge
import time

print("="*60)
print(" OM1 → ISAAC GYM INTEGRATION - BOUNTY #364")
print("="*60)
print("Inizializzazione bridge...")

bridge = OM1IsaacGymBridge()
bridge.create_sim()
bridge.load_ant()

print("Robot 'Ant' caricato. Avvio simulazione con LiDAR...")
print("-" * 60)

for i in range(100):
    bridge.step()
    lidar = bridge.get_lidar()
    print(f"Step {i:3d} | LiDAR [m]: {lidar}")
    time.sleep(0.01)

print("-" * 60)
print("SIMULAZIONE COMPLETATA!")
print("Bounty #364 PRONTA per submission.")
print("="*60)
