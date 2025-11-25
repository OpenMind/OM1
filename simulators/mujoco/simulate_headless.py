import mujoco
import numpy as np

print("=== TEST INTEGRAZIONE MUJOCO - BOUNTY #362 ===")
model = mujoco.MjModel.from_xml_path('pendulum.xml')
data = mujoco.MjData(model)

for i in range(1000):
    data.ctrl[0] = np.sin(i * 0.01) * 5.0
    mujoco.mj_step(model, data)

print(f"Simulazione completata!")
print(f"Posizione finale: {data.qpos[0]:.3f} rad")
print(f"Velocità finale: {data.qvel[0]:.3f} rad/s")
print("Bounty #362: INTEGRAZIONE MUJOCO COMPLETATA!")
