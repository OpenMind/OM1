# Gazebo Environment Improvements - Bounty #363

## Overview

Enhanced Gazebo simulation environment for OM1 AGI robot with realistic obstacles, advanced sensors, and full OM1 API integration.

## Features

- **Office Environment**: Realistic 3D office space with static obstacles
- **Enhanced Robot Model**: OM1 robot with RGB camera, 360° LiDAR, and IMU
- **OM1 API Integration**: Real-time WebSocket bridge for sensor streaming
- **ROS 2 Support**: Full integration with ROS 2 Humble

## Quick Start

```bash
# Launch Gazebo with office environment
cd ~/OM1
gz sim gazebo_worlds/enhanced/office_environment.world

# In another terminal, run OM1 bridge
python3 scripts/om1_gazebo_bridge.py



Architecture

Gazebo Simulation
    ↓ (sensor data)
ROS 2 Bridge Nodes
    ↓ (topics: /camera, /lidar, /imu)
OM1 Gazebo Bridge
    ↓ (WebSocket)
OM1 API (wss://api.openmind.org)



## Environment Components

- **Ground Plane**: 100x100m gray surface
- **Box Obstacle**: 1x1x1m red cube at (2, 0, 0.5)
- **Cylinder Obstacle**: 0.5m radius, 1m height blue cylinder at (-2, 2, 0.5)
- **Sphere Obstacle**: 0.5m radius green sphere at (0, -3, 0.5)
- **OM1 Robot**: 0.5x0.3x0.2m blue robot at origin


## Sensors

- **RGB Camera**: 640x480@30Hz, 60° FOV
- **LiDAR**: 360° scan, 30m range, 10Hz
- **IMU**: 3-axis accelerometer + gyroscope, 100Hz


## Configuration

Edit `config/om1_config.yaml` to configure API endpoint and sensors.

## Troubleshooting

- **Robot not visible**: Check model path in world file
- **Bridge connection fails**: Verify API key in config
- **ROS topics not publishing**: Install ros-humble-ros-gz-bridge
