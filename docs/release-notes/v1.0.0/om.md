---
title: OM1 Production Ready Release
description: "v1.0.0"
---

## What's included

This release significantly expands OM1’s autonomy, simulation, and deployment capabilities.
Key highlights include full Gazebo simulation support, full autonomy for Unitree G1, over-the-air (OTA) updates, improved localization accuracy, and major runtime and ML stack upgrades. Developer experience has been enhanced through hot reload, configuration versioning, and expanded documentation.

## Features

### [v1.0.0](https://github.com/OpenMind/OM1/releases/tag/v1.0.0)

### Development Section

A new **Development** section has been added to the documentation to help developers quickly get started with:

- Environment setup
- Building the runtime
- Running and testing OM1 locally

This reduces onboarding time and standardizes development workflows.

### Hot Reload

Hot reload support has been added to accelerate development cycles:

- Resolved dependencies are reused after the first run
- Runtime configuration is persisted in `.runtime.json5`
- Enables seamless restarts and agent switching without full reinitialization

### Pydantic-Based Configuration

Refactored the action connector architecture to improve type safety, extensibility, and maintainability by introducing Pydantic-based configurations and stronger generic typing. Added connector-specific config models and enhanced documentation with detailed docstrings for improved clarity and validation.

## Simulation & Testing

### Gazebo Simulation Support

OM1 now provides full **Gazebo** simulation support for **Unitree Go2**, including:

- SLAM
- Navigation
- Auto charging

This allows users to test OM1 without physical hardware, enabling faster iteration and safer experimentation.

## Autonomy & Navigation

### Unitree G1 – Full Autonomy Support

OM1 now supports full autonomy for **Unitree G1**, including:

- Facial detection and anonymisation
- 3D SLAM map generation
- Autonomous navigation

### Context-Aware Mode Transitions

OM1 now supports **context-aware mode transitions**, enabling autonomous switching between modes without human intervention.

### LiDAR Localization Improvements

Localization accuracy has been improved through enhanced **LiDAR-based localization**.

## AI & Machine Learning

### Local LLM Support on Thor

Added support for **local large language models (LLMs)** on **Thor**, featuring:

- OM1 now supports Qwen3-30B local LLM
- 3.2-second response timeout
- Arbitration between cloud and local responses
- Local LLM determines the final response when both are available

### ML Stack Migration

The machine learning stack has been migrated to **Thor (Jetson 7.0)**:

- **AGX remains supported**, but with limited capabilities

## Deployment & Operations

### Over-the-Air (OTA) Updates

Introduced full support for over-the-air updates, enabling users to seamlessly upgrade to the latest runtime versions. Configurations now allow smooth version management and automated deployment of updates.

### Configuration Version Management

Configuration files now include a mandatory **`version`** field to ensure compatibility as the runtime evolves.

### Runtime Version Upgrade

The runtime has been upgraded to the latest version, improving:

- Performance
- Stability
- Maintainability

## Known Issues

Avoid running the OM1 container when exploring full autonomy on Unitree Go2. Run via `uv run src/run.py unitree_go2_autonomy_advance`.

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

The docker image is also available at [Docker Hub](https://hub.docker.com/layers/openmindagi/om1/v1.0.0).

For more technical details, please refer to the [docs](https://docs.openmind.org/full_autonomy_guidelines/om).
