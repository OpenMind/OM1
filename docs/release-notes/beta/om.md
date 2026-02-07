---
title: OM1 Beta Release
description: "v1.0.0-beta.4"
icon: rectangle-beta
---

## What's included

Beta release for the Docker image openmindagi/om1, which provides the full setup of OM1 service in one go without having the user to install dependencies separately.

## Features

### [v1.0.0-beta.4](https://github.com/OpenMind/OM1/releases/tag/v1.0.0-beta.4)
- Openrouter support for LLama and Anthropic: Added compatibility with OpenRouter API, enabling seamless access to more AI providers, including Meta’s LLaMA and Anthropic Claude models. This allows flexible model selection for natural language processing, reasoning, and control tasks depending on performance or cost preferences.
- Support multiple modes: We now support 5 different modes with Unitree Go2 full autonomy.
    Welcome mode - Initial greeting and user information gathering
    Conversation - Focused conversation and social interaction mode
    Slam - Autonomous navigation and mapping mode
    Navigation - Autonomous navigation mode
    Guard - Patrol and security monitoring mode
- Support face blurring and detection: The OpenMind Privacy System is a real-time, on-device face detection and blurring module designed to protect personal identity during video capture and streaming.
    It runs entirely on the Unitree Go2 robot’s edge device, requiring no cloud or network connectivity.
    All frame processing happens locally — raw frames never leave the device. Only the processed, blurred output is stored or streamed.
    The module operates offline and maintains low latency suitable for real-time applications
- Support multiple RTSP inputs: The OpenMind RTSP Ingest Pipeline manages multiple RTSP inputs, supporting three camera feeds and one microphone input for synchronized streaming. The top camera feed is processed through the OpenMind face recognition module for detection, overlay, and FPS monitoring, while the microphone (default_mic_aec) handles audio capture and streaming. All processed video and audio streams are ingested through the OpenMind API RTSP endpoint, enabling multi-source real-time data flow within the system.
- Support echo cancellation and remote video streaming: Use our portal to remotely display your face in our dog backpack and talk to people directly.
- Support navigation and mapping: The Navigation and Mapping enables OM1 to move intelligently within its environment using two core modes: Navigation Mode and Slam Mode.
    In Slam Mode, the robot explores its surroundings autonomously, using onboard sensors to build and continuously update internal maps for spatial awareness and future navigation. This mode is typically used during initial setup or when operating in new or changing environments.
    In Navigation Mode, the robot travels between predefined points within an existing mapped area, leveraging maps generated in Slam Mode for path planning, obstacle avoidance, and safe movement to target locations.
- Refactor AI control messaging: We now use function calls for taking actions.
    Here's our new flow - Actions -> Function calls params -> LLM -> Function calls -> Json Structure (CortexOutputModel).
- Support Nvidia Thor: We now support Nvidia Thor for Unitree Go2 full autonomy.
- Added release notes to our docs: The official documentation now includes a dedicated Release Notes section, making it easier to track feature updates, improvements, and bug fixes over time. This also improves transparency for developers and users integrating new releases.
- Introducing Lifecycle
    Each operational mode in OM1 follows a defined lifecycle, representing the complete process from entry to exit of that mode. A mode lifecycle ensures predictable behavior, safe transitions, and consistent data handling across all system states.

### [v1.0.0-beta.3](https://github.com/OpenMind/OM1/releases/tag/v1.0.0-beta.3)
- Downgraded Python to 3.10 for better Jetson support.
- Integrated Nav2 for state feedback and target publishing, with auto AI-mode disable after localization.
- Zenoh configs/sessions moved to zenoh_msgs, now preferring local network before multicast.
- Added avatar background server to communicate with the OM1-avatar
- Improved avatar animation with thinking behavior and ASR response injection into prompts.
- Added support for long range control of humanoids and quadrupeds using the TBS_TANGO2 radios.
- Added sleep mode for ASR, if there's no voice input for 5 min, it goes to sleep.

### [v1.0.0-beta.2](https://github.com/OpenMind/OM1/releases/tag/v1.0.0-beta.2)
- Support for custom camera indices and enables both microphone and speaker functionality in Docker.

### [v1.0.0-beta.1](https://github.com/OpenMind/OM1/releases/tag/v1.0.0-beta.1)
- Multiple LLM provider integrations(OpenAI, Gemini, Deepseek, xAI).
- GoogleASR model for speech to text.
- Riva and Eleven Labs for TTS.
- Preconfigured support for Unitree Go2, G1, TurtleBot, Ubtech Yanshee.
- Simulator support with Gazebo for Go2.
- Multi-arch support - AMD64 and ARM64.

## Docker image

### Setup the API key

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

The OM1 service is provided as a Docker image for easy setup:
```bash
cd OM1
docker-compose up om1 -d --no-build
```

The docker image is also available at [Docker Hub](https://hub.docker.com/layers/openmindagi/om1/v1.0.0-beta.4).

For more technical details, please refer to the [docs](https://docs.openmind.org/full_autonomy_guidelines/om).
