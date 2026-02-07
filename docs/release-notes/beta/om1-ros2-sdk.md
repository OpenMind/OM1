---
title: OM1 ROS2 SDK Beta Release
description: "v1.0.1-beta.1"
icon: rectangle-beta
---

## What's included

Beta release for the Docker image openmindagi/om1_ros2_sdk, which provides the full ROS2 system for running the Unitree Go2, G1 and Limx Tron SDK.

## Features

## [v1.0.1-beta.1](https://github.com/OpenMind/OM1-ros2-sdk/releases/tag/v1.0.1-beta.1)
- Support for the LimX Tron robot model, enabling advanced robotics applications with enhanced capabilities.
- The Docker image and repository have been renamed back to om1-ros2-sdk for consistency with other OM1 components.
- Restructure of the codebase to support multiple robot models with a unified architecture, allowing for easier expansion and maintenance.
- Improved configuration management, enabling users to easily switch between different robot models and customize settings.

### [v1.0.0-beta.3](https://github.com/OpenMind/OM1-ros2-sdk/releases/tag/v1.0.0-beta.3)
- The Docker image and repository have been renamed to unitree_sdk, reflecting the unified SDK architecture and simplifying deployment workflows.
- Smart Auto-Charging System
  - The robot now supports AprilTag-based visual docking integrated with Nav2 navigation.
  - When the robot requires charging, it autonomously navigates to the charging station’s vicinity using the Nav2 stack.
  - Upon reaching the target area, it switches to precision docking mode, using onboard cameras to detect AprilTags mounted near or on the charging dock.
  - This hybrid approach ensures robust and accurate alignment for seamless autonomous charging.
- The CYCLONE_INTERFACE has been renamed to CYCLONEDDS_INTERFACE for consistency with the underlying communication standard.
- New Remote Control Feature
  Added the ability to remotely control the robot, enabling manual operation for testing, navigation overrides, and precision movement in complex environments.
- We added monitoring to WatchSensor for audio RTSP.
- Reorganized the orchestrator package by splitting core logic, API, and cloud nodes into 'core', HTTP/ROS handlers into 'handlers', and process, map, location, and charging logic into 'managers'. Added new service and utility modules, moved WebSocket clients/servers to 'utils', and updated imports accordingly. This modular structure improves maintainability, separation of concerns, and scalability for future development.
- Users can now enable or disable TTS mode directly from the web portal for greater flexibility in communication and operation.
- New topics have been added for fetching, enabling, and disabling AI mode, allowing dynamic AI state management via ROS or other interfaces.
- Agile mode of the robot is unstable due to the payload on its back. Operate in Classic Mode for optimal stability and performance is recommended.

### [v1.0.0-beta.2](https://github.com/OpenMind/OM1-ros2-sdk/releases/tag/v1.0.0-beta.2)
- Added CRSF Protocol Support
- Added turbo mode for the xbox controller
- Added listening and scouting configuration to zenoh_bridge_config for disabling multicast to prevent receiving messages from other robots
- Introduced Pydantic models for location and pose data
- New endpoints to add and list map locations, and ensure locations are stored in a dedicated directory.
- docker-compose is updated to mount a new 'locations' volume. Also extended the orchestrator API to handle 'add_location' and 'list_locations' actions.
- Added MediaMTX watcher: MediaMTX is a zero-dependency media server and proxy used to publish, read, proxy, record, and playback live video and audio streams. It serves as a central "media router," handling multiple streaming protocols like RTSP, WebRTC, RTMP, and HLS.

### [v1.0.0-beta.1](https://github.com/OpenMind/OM1-ros2-sdk/releases/tag/v1.0.0-beta.1)
- Real-time SLAM: Simultaneous localization and mapping using SLAM Toolbox
- RPLiDAR Integration: Support for RPLiDAR A1/A2/A3 series sensors
- Navigation: Integration with Nav2 for autonomous navigation
- Robot Control: Direct integration with Unitree Go2 movement commands
- Visualization: Pre-configured RViz setup for monitoring
- Transform Management: Automatic handling of coordinate frame transforms

## Component Overview

### Watchdog

- Monitors ROS2 topics and sensor health.
- Automatically restarts `om1_sensor` if any topics or sensors stop publishing data.
- Ensures system stability during long-running sessions.

### om1_sensor

- Manages all low-level sensor drivers:
  - Intel RealSense D435 (depth camera)
  - RPLidar (LiDAR scanning)
- Publishes ROS2 topics for system consumption:
  - `/om/paths` — Processed path and localization data
  - `/scan` — Raw LiDAR scan data

### Orchestrator

- Provides API endpoints and cloud service integration.
- Manages:
  - SLAM (Simultaneous Localization and Mapping)
  - Navigation (Nav2)
  - Map storage and loading
- Allows interaction via REST APIs.

### Zenoh Bridge

zenoh_bridge acts as a bridge between OM1 and OM1_sensor to publish and subscribe to/from ROS2 topics.

## Docker image

The OM1-ros2-sdk is provided as a Docker image for easy setup.
```bash
git clone https://github.com/OpenMind/OM1-ros2-sdk.git
```
```bash
cd OM1-ros2-sdk
docker-compose up orchestrator -d --no-build
docker-compose up om1_sensor -d --no-build
docker-compose up watchdog -d --no-build
docker-compose up zenoh_bridge -d --no-build
```

The docker image is also available at [Docker Hub](https://hub.docker.com/layers/openmindagi/OM1-ros2-sdk/v1.0.1-beta.1).

For more technical details, please refer to the [docs](https://docs.openmind.org/full_autonomy_guidelines/om1_ros2_sdk).
