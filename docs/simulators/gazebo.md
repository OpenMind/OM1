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

Checkout the video walkthrough [here](https://assets.openmind.org/education-video/Gazebo%20Setup%20Tutorial%20%28Part%201%29.mp4). More video tutorials coming soon.

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

Step 1: Clone the [OM1-ros2-sdk](https://github.com/OpenMind/OM1-ros2-sdk) repository:

```bash
git clone https://github.com/OpenMind/OM1-ros2-sdk.git
```

Step 2: Install all the necessary dependencies:

```bash
cd OM1-ros2-sdk
uv venv --python 3.10
sudo rosdep init
rosdep update
rosdep install --from-paths . --ignore-src -r -y
source .venv/bin/activate
uv pip install .
```

Step 3: Build all the packages:

```bash
colcon build
```

Now you should be able to launch the **Gazebo Simulator**.

Step 4: Open a terminal and run the following commands. You'll now be able to see the Gazebo and RViZ windows launch on your system.

```bash
source install/setup.bash
ros2 launch go2_gazebo_sim go2_launch.py
```

Step 5: Open a new terminal and run:

```bash
source install/setup.bash
ros2 launch go2_sdk sensor_launch.py use_sim:=true
```

This will bring up the `om/path` topic, enabling OM1 to understand the surrounding environment.

Step 6: Open a new terminal and run:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/.venv/lib/python3.10/site-packages
source install/setup.bash
ros2 launch orchestrator orchestrator_launch.py use_sim:=true
```

This will bring up the `orchestrator`, to consume data collected by `om1_sensor` for SLAM and Navigation.

Step 7: Run Zenoh Ros2 Bridge

To run the Zenoh bridge for the Unitree Go2, you need to have the Zenoh ROS 2 bridge installed. You can find the installation instructions in the [Zenoh ROS 2 Bridge documentation](https://github.com/eclipse-zenoh/zenoh-plugin-ros2dds)

After installing the Zenoh ROS 2 bridge, you can run it with the following command:

```bash
zenoh-bridge-ros2dds -c ./zenoh/zenoh_bridge_config.json5
```

Step 8: Start OM1

Refer to the [Installation Guide](../developing/1_get-started.md) for detailed instructions.

Then add the optional Python CycloneDDS module to OM1, run

```bash
uv pip install -r pyproject.toml --extra dds
```

Setup your API key in `.bashrc` file and run your simulation agent:

Get your API key from the [portal](https://portal.openmind.org), and add it to `bashrc`

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

Step 9: Teleoperate the robot in simulation

You can also use teleoperation to control the robot through your keyboard.

Switch back to `OM1-ros2-sdk` in a new terminal and run the following commands

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

> **Note**: We don't have auto charging feature supported with Gazebo but it will be launched soon. Stay tuned!
