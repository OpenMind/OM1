# MuJoCo Simulator Integration - Bounty #362

Complete MuJoCo physics simulator integration for OpenMind OM1 with inverted pendulum example and Gymnasium environment wrapper.

## Features

- **MuJoCo Physics Engine** - High-performance physics simulation
- **Inverted Pendulum Model** - XML-defined model with sensors and actuators
- **Gymnasium Integration** - RL-ready environment wrapper
- **Interactive Viewer** - Real-time visualization with keyboard controls
- **Headless Mode** - Automated testing and training support

## Files Structure
simulators/mujoco/
├── pendulum.xml              # Inverted pendulum MuJoCo model
├── gym_env.py                # Gymnasium environment wrapper
├── simulate.py               # Interactive simulation with viewer
├── simulate_headless.py      # Headless mode for automation
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── *.mp4                     # Demo videos


## Requirements

- Python 3.8+
- MuJoCo 2.3.0+
- Gymnasium

## Installation

### Step 1: Install Dependencies

```bash
cd simulators/mujoco
pip install -r requirements.txt


Or manually:
pip install mujoco>=2.3.0
pip install gymnasium[mujoco]


Step 2: Verify Installation
python simulate.py

**Controls:**

- ESC: Exit simulation
- Mouse: Rotate camera
- Scroll: Zoom in/out


### Headless Mode

Run without GUI for testing or training:
python simulate_headless.py

### Gymnasium Environment

Use as a standard Gym environment for reinforcement learning:
from gym_env import PendulumEnv

env = PendulumEnv()
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()

## Model Details

### Inverted Pendulum (`pendulum.xml`)

- **Base**: Fixed body (box)
- **Pole**: Cylindrical pendulum with hinge joint
- **Sensors**: Joint position and velocity
- **Actuator**: Motor at hinge joint
- **Ground**: Checkerboard plane


### State Space

- Position: Joint angle (radians)
- Velocity: Joint angular velocity (rad/s)


### Action Space

- Continuous: Torque applied to joint [-1, 1]


## Testing

### Test Interactive Mode
python simulate.py

Expected: Window opens with pendulum simulation

### Test Gymnasium Environment
python gym_env.py

Expected: 1000 steps executed with random actions

### Test Headless Mode
python simulate_headless.py

Expected: Simulation runs without GUI, outputs to console

## Demo Videos

Two demonstration videos are included:

- `mujoco_pendulum_demo.mp4` - Interactive simulation
- `mujoco_pendulum_demo_CORRETTO.mp4` - Corrected demo


## Troubleshooting

### "ImportError: No module named 'mujoco'"
pip install mujoco>=2.3.0

### "GLEW initialization error"

This is normal for headless environments. Use `simulate_headless.py` instead.

### Model not loading

Verify `pendulum.xml` is in the same directory as the Python scripts.

## Integration with OM1

This MuJoCo backend can be integrated into OM1's simulator orchestrator for multi-simulator support.

Future integration points:

- Add to `src/simulators/plugins/`
- Register with simulator orchestrator
- Implement OM1 API interface


## References

- MuJoCo Documentation: [https://mujoco.readthedocs.io/](https://mujoco.readthedocs.io/)
- Gymnasium: [https://gymnasium.farama.org/](https://gymnasium.farama.org/)
- OM1 Repository: [https://github.com/OpenMindAGI/OM1](https://github.com/OpenMindAGI/OM1)


## Credits

**Author**: lau90eth**Bounty**: #362 - MuJoCo Simulator Integration**Date**: November 2025
