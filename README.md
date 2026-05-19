![om1_banner_w](https://github.com/user-attachments/assets/86590615-018d-4443-b345-8d224227e83f)

<p align="center">
<a href="https://arxiv.org/abs/2412.18588">Technical Paper</a> |
<a href="https://docs.openmind.com/">Documentation</a> |
<a href="https://x.com/openmind_agi">X</a>
</p>

**OpenMind's OM1 is a modular AI runtime that empowers developers to create and deploy multimodal AI agents across digital environments and physical robots**, including Humanoids, Phone Apps, Quadrupeds, educational robots such as TurtleBot 4, and simulators like Gazebo and Isaac Sim. OM1 agents can process diverse inputs like web data, social media, camera feeds, and LIDAR, while enabling physical actions including motion, autonomous navigation, and natural conversations. The goal of OM1 is to make it easy to create highly capable human-focused robots, that are easy to upgrade and (re)configure to accommodate different physical form factors.

## Capabilities of OM1

* **Modular Architecture**: Designed with Python for simplicity and seamless integration.
* **Data Input**: Easily handles new data and sensors.
* **Hardware Support via Plugins**: Supports new hardware through plugins for API endpoints and specific robot hardware connections to `ROS2`, `Zenoh`, and `CycloneDDS`. (We recommend `Zenoh` for all new development).
* **Pre-configured Endpoints**: Supports Text-to-Speech, multiple LLMs from OpenAI, xAI, DeepSeek, Anthropic, Meta, Gemini, NearAI, Ollama (local), and multiple Visual Language Models (VLMs) with pre-configured endpoints for each service.
* **Metrics & Observability**: Includes a pre-configured Prometheus and Grafana stack to monitor real-time AI pipeline metrics like LLM and ASR latencies.

## Architecture Overview
![Artboard 1@4x 1 (1)](https://github.com/user-attachments/assets/0c482257-e4db-4a0a-8d83-d4548ac4beaf)

## Getting Started

If you are new to OM1, this is the fastest path to a successful first run using the `spot` agent.

Spot uses your webcam to detect objects and sends those observations to the LLM. The model then returns move/speak/emotion outputs as logs and API responses.

Spot in this quick start is the default starter configuration to help you understand the OM1 pipeline. It visualizes state updates in the terminal and does not execute robot hardware actions.

### Quick Start (5 Minutes)

1. Install system dependencies.
2. Clone the repository.
3. Add your OpenMind API key.
4. Launch OM1 and verify it is running.

### Prerequisites

- Python 3.10+
- [`uv` package manager](https://docs.astral.sh/uv/getting-started/installation/)
- Webcam access (recommended if configuring VLM)

Install system packages:

### 1. Install System Dependencies

For macOS:
```bash
brew install portaudio ffmpeg
```

For Linux:
```bash
sudo apt-get update
sudo apt-get install -y portaudio19-dev python3-dev ffmpeg
```

### 2. Clone

```bash
git clone https://github.com/OpenMind/OM1.git
cd OM1
git submodule update --init
uv venv
```

### 3. Configure API Key

Get your API key from [OpenMind Portal](https://portal.openmind.com/).

1. Sign in to OpenMind Portal.
2. Open the dashboard and create a new API key.
3. Copy the generated key.

Recommended (shell profile):
```bash
export OM_API_KEY="<your_api_key>"
```

Alternative (project-local):
```bash
cp .env.example .env
```

Then set:
```bash
OM_API_KEY=<your_api_key>
```
in `.env`.

You can also verify or adjust the fallback key location in `config/spot.json5`.

### 4. Launch Spot

```bash
uv run src/run.py spot
```

#### Verify It Is Working

Your setup is successful if:

- The terminal shows the Spot agent has started successfully.
- You see input processing and LLM responses logged in the terminal.
- The agent responds to camera input and generates appropriate outputs.

### 5. Monitor with Grafana (Optional)

If you have Docker installed, you can launch the included Prometheus and Grafana stack to monitor real-time AI pipeline metrics such as LLM and ASR latencies:

```bash
docker-compose up -d grafana prometheus
```

Navigate to <http://localhost:3000> in your browser (default login: `admin`/`admin`). The **OM1 Latency Monitoring** dashboard is automatically provisioned and ready to use.

### Troubleshooting

- `Authentication` errors: confirm `OM_API_KEY` is set and not expired.
- `No module` errors: run the command with `uv run` from the repo root.
- `Camera` access issues: grant terminal/IDE camera permissions in OS settings.
- `Address already in use` on port `8000`: stop the conflicting process or free the port.

### OMCU

OMCU is the computational unit for billing on OpenMind's platform. The free plan provides 50 OMCU renewed monthly.

Upgrade your plan [here](https://portal.openmind.com/) for additional credits.

For more help connecting OM1 to your robot hardware, see [getting started](https://docs.openmind.com/developing/1_get-started).

> **Note:** This quick start uses the Spot starter configuration. For voice interactions, ensure ASR and TTS are configured in `config/spot.json5`.

## What's Next?

* Try out some [examples](https://docs.openmind.com/developer-cookbook/examples)
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

OM1 can interface with your HAL via USB, serial, ROS2, CycloneDDS, Zenoh, or websockets. For an example of an advanced humanoid HAL, see [Unitree's C++ SDK](https://github.com/unitreerobotics/unitree_sdk2/blob/adee312b081c656ecd0bb4e936eed96325546296/example/g1/high_level/g1_loco_client_example.cpp#L159). Frequently, a HAL, especially ROS2 code, is dockerized and interfaces with OM1 through DDS middleware or websockets.

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

The BrainPack is designed to be mounted directly onto a robot to bring together mapping, object recognition, remote control, and self charging, giving humanoids and quadrupeds what they need to navigate, remember, and act with purpose.

## Full Autonomy Guidance

OM1 supports **full autonomy** for Unitree Go2 and G1 with BrainPack. The following features are supported with BrainPack:

- **Navigation** - Autonomous path planning and movement.
- **SLAM** - Simultaneous Localization and Mapping for persistent map-based operation.
- **Auto Charging** - Automated docking and battery charging workflows.
- **Face Detection and Anonymization** - Real-time perception and privacy-aware processing.

For more details, see [Full Autonomy](docs/full_autonomy_guidelines/architecture_overview.md).

The BrainPack is open-source and you can refer to the guidelines to build your own [here](https://github.com/OpenMind/brainpack).

## Simulator Support

OM1 integrates with popular robotics simulators to enable rapid prototyping and testing without physical hardware.

### Gazebo

Open source Gazebo support is designed for rapid prototyping, conversational interaction, and behavior testing. Gazebo integration with OM1 is supported for Unitree Go2.

See [Gazebo](docs/simulators/gazebo.md) to get started.

### Isaac Sim

NVIDIA Isaac Sim support enables physics-accurate simulation with GPU acceleration. Isaac Sim integration with OM1 is supported for Unitree Go2 and G1.

Requires NVIDIA GPU and CUDA support. See [Isaac Sim Setup](docs/simulators/isaac-sim.md) to get started.

## Detailed Documentation

More detailed documentation can be accessed at [docs.openmind.com](https://docs.openmind.com/).

## Contributing

Please make sure to read the [Contributing Guide](./CONTRIBUTING.md) before making a pull request.

## License

This project is licensed under the terms of the MIT License, which is a permissive free software license that allows users to freely use, modify, and distribute the software. The MIT License is a widely used and well-established license that is known for its simplicity and flexibility. By using the MIT License, this project aims to encourage collaboration, modification, and distribution of the software.
