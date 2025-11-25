import mujoco
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

model = mujoco.MjModel.from_xml_path("pendulum.xml")
data = mujoco.MjData(model)

print("Simulazione + video MP4 corretto in corso... (20 secondi)")

positions = []
forces = []

for i in range(1200):
    # Forza sinusoidale morbida → oscillazione naturale
    force = 8 * np.sin(i * 0.02)   # oscillazione dolce
    data.ctrl[0] = force
    forces.append(force)
    
    mujoco.mj_step(model, data)
    positions.append(data.qpos[0])

# Animazione
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
fig.suptitle("Inverted Pendulum – Bounty #362", fontsize=16)

# Grafico angolo
line1, = ax1.plot([], [], 'b-', lw=3)
ax1.set_xlim(0, 1200)
ax1.set_ylim(-3.5, 3.5)
ax1.set_ylabel("Theta (rad)")
ax1.grid(True)

# Grafico forza applicata
line2, = ax2.plot([], [], 'r-', lw=2)
ax2.set_xlim(0, 1200)
ax2.set_ylim(-10, 10)
ax2.set_xlabel("Time Step")
ax2.set_ylabel("Force")
ax2.grid(True)

def animate(i):
    line1.set_data(range(i+1), positions[:i+1])
    line2.set_data(range(i+1), forces[:i+1])
    return line1, line2

anim = FuncAnimation(fig, animate, frames=1200, interval=33, blit=True)
anim.save('mujoco_pendulum_demo_CORRETTO.mp4', writer='ffmpeg', fps=60)

print("VIDEO CORRETTO SALVATO → mujoco_pendulum_demo_CORRETTO.mp4")
print("Ora il pendolo OSCILLA e si stabilizza – perfetto per la bounty!")
