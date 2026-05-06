---
title: Gazebo Simulator
description: "Quadruped Simulation and Control"
icon: robot
---

## System Requirements

| Component | Minimum | Good/Recommended | Ideal |
|-----------|---------|------------------|-------|
| **CPU** | Intel i7 (10th gen) or AMD Ryzen 7-8 cores minimum | Intel i9 (12th gen+) or AMD Ryzen 9-12 cores | AMD Ryzen 9 7950X or Intel i9-13900K 16+ cores (24+ threads) |
| **RAM** | 16 GB | 32 GB | 64 GB|
| **GPU** | NVIDIA GTX 1660 Ti 6 GB VRAM | NVIDIA RTX 3070 or RTX 4060 Ti 8-12 GB VRAM | NVIDIA RTX 4080/4090 16+ GB VRAM with CUDA 11.8+ |
| **OS** | Ubuntu 22.04 | Ubuntu 22.04 | Ubuntu 22.04 |

It's ideal to have at least 128 GB SSD storage for the setup to run smoothly.

## Simulation Instructions

To get started with **Gazebo** and **Unitree SDK**, please install cyclonedds and **ROS2 Humble** first. You can find the installation steps [here](../developing/middleware.md).

To install compilers and other tools to build ROS packages, run

```bash
sudo apt install ros-dev-tools
```

Install the following additional dependencies

```bash
sudo apt install ros-humble-rmw-cyclonedds-cpp
sudo apt install ros-humble-rosidl-generator-dds-idl
```

Set up your environment by sourcing the following file.

```bash
source /opt/ros/humble/setup.bash
```

If you don't have uv installed, use the following command to install it on your system.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Check if you have rosdep installed by running `rosdep` or `rosdep --version`. If it is not installed, run the following:

```bash
sudo apt-get update
sudo apt-get install python3-rosdep
```

Once you've successfully completed above steps, follow the following steps to start the gazebo simulation, generate SLAM map of the surrounding and start navigation.

Step 1: Clone the [OM1-sim](https://github.com/OpenMind/OM1-sim) repository:

```bash
git clone https://github.com/OpenMind/OM1-sim.git
```

Step 2:  Initialize rosdep by setting up the source list (this is only needed once per machine)

> **Note:**  Run 'sudo rosdep init' only if rosdep has not been initialized on this machine before. If it has, skip this line and start from 'rosdep update'.

```bash
cd OM1-sim
sudo rosdep init
rosdep update
rosdep install --from-paths . --ignore-src -r -y
```

It automatically installs all system dependencies needed by the ROS packages in your current directory.

Step 3: Build all the packages:

```bash
colcon build
```

Once the build is successful, create a virtual environment and install the dependencies

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install .
```

Now you should be able to launch the **Gazebo Simulator**.

Step 4: Open a terminal and run the following commands. You'll now be able to see the Gazebo and RViZ windows launch on your system.

```bash
source install/setup.bash
ros2 launch go2_gazebo_sim go2_launch.py
```

Step 5: Run Zenoh Ros2 Bridge

To run the Zenoh bridge for the Unitree Go2, you need to have the Zenoh ROS 2 bridge installed. You can find the installation instructions in the [Zenoh ROS 2 Bridge documentation](https://github.com/eclipse-zenoh/zenoh-plugin-ros2dds)

After installing the Zenoh ROS 2 bridge, you can run it with the following command:

```bash
zenoh-bridge-ros2dds -c ./zenoh/zenoh_bridge_config.json5
```

Step 6: Start OM1

Refer to the [Installation Guide](../developing/1_get-started.md) for detailed instructions.

Then add the optional Python CycloneDDS module to OM1, run

```bash
uv pip install -r pyproject.toml --extra dds
```

Setup your API key in `.bashrc` file and run your simulation agent:

Get your API key from the [portal](https://portal.openmind.com), and add it to `bashrc`

```bash
vi ~/.bashrc
```

```bash
export OM_API_KEY="<your_api_key>"
```

Now, run the simulation agent

```bash
uv run src/run.py simulation
```

Step 7: Teleoperate the robot in simulation

You can also use teleoperation to control the robot through your keyboard.

Open `OM1-sim` directory in a new terminal and run the following commands

```bash
source install/setup.bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

Use the keyboard controls displayed in the terminal to move the robot:
```
i - Move forward
, - Move backward
j - Turn left
l - Turn right
k - Stop
U/O/M/> - Move diagonally
```

> **Note**:
> 1. Auto charging feature is not supported with Gazebo but it will be launched soon. Stay tuned!
> 2. SLAM Map generation and Navigation are only offered as part of Premium features and would require an active subscription to Enterprise Plan on [OpenMind Portal](https://portal.openmind.com).
