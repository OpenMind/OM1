---
title: OM1 ROS2 SDK Production Ready Release
description: "v1.0.0"
icon: tarp
---

## What's included

Production ready release for the Docker image openmindagi/unitree_sdk, which provides the full ROS2 system for running the Unitree SDK.

## Features

### [v1.0.0](https://github.com/OpenMind/OM1-ros2-sdk/releases/tag/v1.0.0)

### Go2 LiDAR Localization
Introduced a custom LiDAR localization node into the navigation launch pipeline, replacing the previous AMCL-based localization. Package dependencies and entry points have been updated to support the new implementation.

### Localization via Zenoh
Enhanced the localization service and configuration to support additional topics and service endpoints. Updated service paths and expanded the Zenoh bridge configuration to improve localization data handling.

### NumPy Compatibility Update
Downgraded NumPy to maintain compatibility with tf_transformations.

### Risk-Aware Area Avoidance
Added a risk-aware local map and path selection mechanism to help the robot avoid unsafe areas such as holes, drops, steep terrain, and unknown regions. This improves navigation safety, simplifies tuning, and enables easier debugging through RViz.

### Enhanced Localization Parameters

- Reduced global localization particles from 10,000 to 5,000
- Added parameters for minimum prediction confidence, maximum consecutive failures, and failure quality thresholds
- Implemented consecutive failure tracking with automatic re-localization
- Updated logging to expose confidence and failure metrics
- Reduced Localization CPU Usage
- Optimized obstacle expansion logic in go2_lidar_localization.py by replacing a custom gradient mask loop with scipy.ndimage.maximum_filter, improving performance, maintainability, and code clarity.

### LiDAR Rotation Fix
Fixed the issue affecting LiDAR rotation handling to improve localization accuracy.

### Adjusted Movement Policy
Updated movement policy logic to improve navigation behavior and stability.

### Launch & System Architecture

- Odom Relay Node Refactor
- Moved the odometry relay node from the navigation and SLAM launch files into the sensor launch file. This centralizes relay functionality and improves reliability by enabling automatic restarts on failure.

### Unitree G1 Support Added
Added full support for Unitree G1, including:

- Navigation and robot behavior configuration
- SLAM configuration
- Robot state publishing and visualization

### Gazebo Integration
Fully integrated Gazebo simulation with OM1. Added joystick control, SLAM simulation support, and a simulated RealSense D435 camera for the Go2 robot, along with corresponding Gazebo sensors and ROS bridge topics.

### Frontier-Based Exploration
Added a new ROS 2 package, frontier_explorer, implementing a frontier-based exploration algorithm. This includes core exploration logic, costmap management, configuration files, launch files, and ROS 2 package setup. The package is now integrated with OM1, with updated simulation and configuration parameters and removal of an obsolete submodule reference.

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
docker-compose up zenoh-bridge -d --no-build
```

The docker image is also available at [Docker Hub](https://hub.docker.com/layers/openmindagi/unitree_sdk/v1.0.0).

For more technical details, please refer to the [docs](https://docs.openmind.org/full_autonomy_guidelines/unitree_sdk).
