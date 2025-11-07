![OM_Banner_X2 (1)](https://github.com/user-attachments/assets/853153b7-351a-433d-9e1a-d257b781f93c)

<p align="center">
  <a href="https://arxiv.org/abs/2412.18588">Technical Paper</a> |
  <a href="https://docs.openmind.org/">Documentation</a> |
  <a href="https://x.com/openmind_agi">X</a> |
  <a href="https://discord.gg/openmind">Discord</a>
</p>

---

**OpenMind’s OM1 is a modular AI runtime that empowers developers to create and deploy multimodal AI agents across digital environments and physical robots** — including humanoids, mobile apps, websites, quadrupeds, and educational robots like TurtleBot 4.

OM1 agents process diverse inputs (web data, social media, camera feeds, LIDAR) while enabling physical actions (motion, navigation, and conversation). The goal: to make it easy to build highly capable, human-focused robots that are simple to upgrade and reconfigure across different form factors.

---

## 🚀 Capabilities of OM1

* **Modular Architecture:** Built in Python for simplicity and seamless integration.  
* **Flexible Data Input:** Easily handles new sensors and data streams.  
* **Hardware Plugin Support:** Integrates hardware APIs through plugins supporting `ROS2`, `Zenoh`, and `CycloneDDS`.  
  - 🧠 Recommended middleware: `Zenoh` for all new development.  
* **Web-Based Debug Display:** Use **WebSim** (http://localhost:8000/) for visual debugging and real-time monitoring.  
* **Preconfigured AI Endpoints:** Compatible with OpenAI, xAI, DeepSeek, Anthropic, Meta, Gemini, and NearAI, with built-in Text-to-Speech and Visual Language Model (VLM) support.  

---

## 🧩 Architecture Overview

![Architecture Overview](https://github.com/user-attachments/assets/14e9b916-4df7-4700-9336-2983c85be311)

---

## 🧠 Getting Started

Let’s run the **Spot agent**. It uses your webcam to capture and label objects, sending these captions to an LLM that returns commands for `movement`, `speech`, and `face` actions. These are visualized in **WebSim** with timing and debug info.

### 📦 Prerequisites

You’ll need the [`uv` package manager](https://docs.astral.sh/uv/getting-started/installation/).

### 🧰 Clone the Repository

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
sudo apt-get install portaudio19-dev python-dev ffmpeg
```

### Obtain an OpenMind API Key

Obtain your API Key at [OpenMind Portal](https://portal.openmind.org/). Copy it to `config/spot.json5`, replacing the `openmind_free` placeholder. Or, `cp env.example .env` and add your key to the `.env`. 

### Launching OM1

Run
```bash
uv run src/run.py spot
```

After launching OM1, the Spot agent will interact with you and perform (simulated) actions. For more help connecting OM1 to your robot hardware, see [getting started](https://docs.openmind.org/getting-started).

Note: This is just an example agent configuration.
If you want to interact with the agent and see how it works, make sure ASR and TTS are configured in spot.json5.

## What's Next?

* Try out some [examples](https://docs.openmind.org/examples)
* Add new `inputs` and `actions`.
* Design custom agents and robots by creating your own `json5` config files with custom combinations of inputs and actions.
* Change the system prompts in the configuration files (located in `/config/`) to create new behaviors.

## Interfacing with New Robot Hardware

OM1 assumes that robot hardware provides a high-level SDK that accepts elemental movement and action commands such as `backflip`, `run`, `gently pick up the red apple`, `move(0.37, 0, 0)`, and `smile`. An example is provided in `actions/move_safe/connector/ros2.py`:

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

* Jetson AGX Orin 64GB (running Ubuntu 22.04 and JetPack 6.1)
* Mac Studio with Apple M2 Ultra with 48 GB unified memory (running MacOS Sequoia)
* Mac Mini with Apple M4 Pro with 48 GB unified memory (running MacOS Sequoia)
* Generic Linux machines (running Ubuntu 22.04)

OM1 _should_ run on other platforms (such as Windows) and microcontrollers such as the Raspberry Pi 5 16GB.


## Full Autonomy Guidance

We're excited to introduce **full autonomy mode**, where four services work together in a loop without manual intervention:

- **om1**
- **unitree_sdk** – A ROS 2 package that provides SLAM (Simultaneous Localization and Mapping) capabilities for the Unitree Go2 robot using an RPLiDAR sensor, the SLAM Toolbox and the Nav2 stack.
- **om1-avatar** – A modern React-based frontend application that provides the user interface and avatar display system for OM1 robotics software.
- **om1-video-processor** - The OM1 Video Processor is a Docker-based solution that enables real-time video streaming, face recognition, and audio capture for OM1 robots.

## Intro to BrainPack?
From research to real-world autonomy, a platform that learns, moves, and builds with you.
We'll shortly be releasing the **BOM** and details on **DIY** for the it. 
Stay tuned!

Clone the following repos -
- https://github.com/OpenMind/OM1.git
- https://github.com/OpenMind/unitree-sdk.git
- https://github.com/OpenMind/OM1-avatar.git
- https://github.com/OpenMind/OM1-video-processor.git

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
```
- For OM1-avatar
```bash
cd OM1-avatar
docker-compose up om1_avatar -d --no-build
```
## Detailed Documentation

More detailed documentation can be accessed at [docs.openmind.org](https://docs.openmind.org/).

## Contributing

Please make sure to read the [Contributing Guide](./CONTRIBUTING.md) before making a pull request.

## License

This project is licensed under the terms of the MIT License, which is a permissive free software license that allows users to freely use, modify, and distribute the software. The MIT License is a widely used and well-established license that is known for its simplicity and flexibility. By using the MIT License, this project aims to encourage collaboration, modification, and distribution of the software.
