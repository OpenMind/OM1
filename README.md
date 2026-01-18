![OM_Banner_X2 (1)](https://github.com/user-attachments/assets/853153b7-351a-433d-9e1a-d257b781f93c)

<p align="center">
<a href="https://arxiv.org/abs/2412.18588">Technical Paper</a> |
<a href="https://docs.openmind.org/">Documentation</a> |
<a href="https://x.com/openmind_agi">X</a>
</p>

**OpenMind's OM1 is a modular AI runtime that empowers developers to create and deploy multimodal AI agents across digital environments and physical robots**, including Humanoids, Phone Apps, websites, Quadrupeds, and educational robots such as TurtleBot 4. OM1 agents can process diverse inputs like web data, social media, camera feeds, and LIDAR, while enabling physical actions including motion, autonomous navigation, and natural conversations. The goal of OM1 is to make it easy to create highly capable human-focused robots, that are easy to upgrade and (re)configure to accommodate different physical form factors.

## Capabilities of OM1

* **Modular Architecture**: Designed with Python for simplicity and seamless integration.
* **Data Input**: Easily handles new data and sensors.
* **Hardware Support via Plugins**: Supports new hardware through plugins for API endpoints and specific robot hardware connections to `ROS2`, `Zenoh`, and `CycloneDDS`. (We recommend `Zenoh` for all new development).
* **Web-Based Debugging Display**: Monitor the system in action with WebSim (available at http://localhost:8000/) for easy visual debugging.
* **Pre-configured Endpoints**: Supports Text-to-Speech, multiple LLMs from OpenAI, xAI, DeepSeek, Anthropic, Meta, Gemini, NearAI, Ollama (local), and multiple Visual Language Models (VLMs) with pre-configured endpoints for each service.

## Architecture Overview
![Artboard 1@4x 1 (1)](https://github.com/user-attachments/assets/dd91457d-010f-43d8-960e-d1165834aa58)


## Getting Started

To get started with OM1, let's run the Spot agent. Spot uses your webcam to capture and label objects. These text captions are then sent to the LLM, which returns `movement`, `speech` and `face` action commands. These commands are displayed on WebSim along with basic timing and other debugging information.

### Package Management and VENV

You will need the [`uv` package manager](https://docs.astral.sh/uv/getting-started/installation/).

### Clone the Repo

```bash
git clone https://github.com/OpenMind/OM1.git
cd OM1
git submodule update --init
uv venv
```

### Install Dependencies

For MacOS
```bash
brew install portaudio ffmpeg
```

For Linux
```bash
sudo apt-get update
sudo apt-get install portaudio19-dev python3-dev ffmpeg
```

### Obtain an OpenMind API Key

Obtain your API Key at [OpenMind Portal](https://portal.openmind.org/). Copy it to `config/spot.json5`, replacing the `openmind_free` placeholder. Or, `cp env.example .env` and add your key to the `.env`.

### Launching OM1

Run
```bash
uv run src/run.py spot
```

After launching OM1, the Spot agent will interact with you and perform (simulated) actions. For more help connecting OM1 to your robot hardware, see [getting started](https://docs.openmind.org/developing/1_get-started).

Note: This is just an example agent configuration.
If you want to interact with the agent and see how it works, make sure ASR and TTS are configured in spot.json5.

## What's Next?

* Try out some [examples](https://docs.openmind.org/examples)
* Add new `inputs` and `actions`.
* Design custom agents and robots by creating your own `json5` config files with custom combinations of inputs and actions.
* Change the system prompts in the configuration files (located in `/config/`) to create new behaviors.

## Interfacing with New Robot Hardware

OM1 assumes that robot hardware provides a high-level SDK that accepts elemental movement and action commands such as `backflip`, `run`, `gently pick up the red apple`, `move(0.37, 0, 0)`, and `smile`. An example is provided in `src/actions/move/connector/ros2.py`:

```python
...
elif output_interface.action == "shake paw":
    if self.sport_client:
        self.sport_client.Hello()
...
```

If your robot hardware does not yet provide a suitable HAL (hardware abstraction layer), traditional robotics approaches such as RL (reinforcement learning) in concert with suitable simulation environments (Unity, Gazebo), sensors (such as hand mounted ZED depth cameras), and custom VLAs will be needed for you to create one. It is further assumed that your HAL accepts motion trajectories, provides battery and thermal management/monitoring, and calibrates and tunes sensors such as IMUs, LIDARs, and magnetometers.

OM1 can interface with your HAL via USB, serial, ROS2, CycloneDDS, Zenoh, or websockets. For an example of an advanced humanoid HAL, please see [Unitree's C++ SDK](https://github.com/unitreerobotics/unitree_sdk2/blob/adee312b081c656ecd0bb4e936eed96325546296/example/g1/high_level/g1_loco_client_example.cpp#L159). Frequently, a HAL, especially ROS2 code, will be dockerized and can then interface with OM1 through DDS middleware or websockets.

## Recommended Development Platforms

OM1 is developed on:

* Nvidia Thor (running JetPack 7.0) - full support
* Jetson AGX Orin 64GB (running Ubuntu 22.04 and JetPack 6.1) - limited support
* Mac Studio with Apple M2 Ultra with 48 GB unified memory (running MacOS Sequoia)
* Mac Mini with Apple M4 Pro with 48 GB unified memory (running MacOS Sequoia)
* Generic Linux machines (running Ubuntu 22.04)

OM1 _should_ run on other platforms (such as Windows) and microcontrollers such as the Raspberry Pi 5 16GB.

## Introduction to BrainPack

From research to real-world autonomy, a platform that learns, moves, and builds with you.
We'll shortly be releasing the **BOM** and details on **DIY** for the BrainPack.
Stay tuned!

## Full Autonomy Guidance

We're excited to introduce **full autonomy** for Unitree Go2 and G1 with the BrainPack. Full autonomy has five services that work together in a loop without manual intervention:

- **om1**
- **unitree_sdk** – A ROS 2 package that provides SLAM (Simultaneous Localization and Mapping) capabilities for the Unitree Go2 robot using an RPLiDAR sensor, the SLAM Toolbox and the Nav2 stack.
- **om1-avatar** – A modern React-based frontend application that provides the user interface and avatar display system for OM1 robotics software.
- **om1-video-processor** - The OM1 Video Processor is a Docker-based solution that enables real-time video streaming, face recognition, and audio capture for OM1 robots.
- **om1-system-setup** - To setup wifi, and, monitor and manage docker containers.

Clone the following repositories -
- git clone https://github.com/OpenMind/OM1.git
- git clone https://github.com/OpenMind/unitree-sdk.git
- git clone https://github.com/OpenMind/OM1-avatar.git
- git clone https://github.com/OpenMind/OM1-video-processor.git
- git clone https://github.com/OpenMind/OM1-system-setup.git

## Starting the system
To start all services, run the following commands:
- For OM1

Setup the API key

For Bash: vim ~/.bashrc or ~/.bash_profile.

For Zsh: vim ~/.zshrc.

Add

```bash
export OM_API_KEY="your_api_key"
```

Update the docker-compose file. Replace "unitree_go2_autonomy_advance" with the agent you want to run.
```bash
command: ["unitree_go2_autonomy_advance"]
```

```bash
cd OM1
docker-compose up om1 -d --no-build
```

- For unitree_sdk
```bash
cd unitree_sdk
docker-compose up orchestrator -d --no-build
docker-compose up om1_sensor -d --no-build
docker-compose up watchdog -d --no-build
docker-compose up zenoh_bridge -d --no-build
```

- For OM1-avatar
```bash
cd OM1-avatar
docker-compose up om1_avatar -d --no-build
```

- For OM1-video-processor
```bash
cd OM1-video-processor
docker-compose up -d
```

- For OM1-system-setup
```bash
cd OM1-system-setup
docker-compose up -d ota_agent
docker-compose up -d ota_updater
docker-compose up -d om1_monitor
```

## Detailed Documentation

More detailed documentation can be accessed at [docs.openmind.org](https://docs.openmind.org/).

## Contributing

Please make sure to read the [Contributing Guide](./CONTRIBUTING.md) before making a pull request.

## License

This project is licensed under the terms of the MIT License, which is a permissive free software license that allows users to freely use, modify, and distribute the software. The MIT License is a widely used and well-established license that is known for its simplicity and flexibility. By using the MIT License, this project aims to encourage collaboration, modification, and distribution of the software.
