# Bounty #363: Enhanced Gazebo Environment

## Overview

This implementation enhances the existing Gazebo simulator environment for OM1 with:
- Rich test environment (warehouse + office sections)
- Multiple obstacle types for navigation testing
- Sensor integration (Camera, IMU, Depth)
- Demo robot with differential drive
- Example navigation script

## Features

### Enhanced Gazebo World

**Environment Sections:**
- **Warehouse Area**: Boxes, pallets, barrels for obstacle testing
- **Office Area**: Desk, chair, cabinet for indoor navigation
- **Test Arena**: Walls, cylinders, ramps for challenging scenarios
- **Boundary Walls**: Enclosed environment for safe testing

**Lighting:**
- Natural sunlight
- Additional point lights for indoor areas
- Proper ambient and diffuse lighting

### Robot Model: OM1 Test Robot

**Specifications:**
- Base: 0.5m x 0.4m x 0.3m mobile platform
- Drive: Differential drive (2 wheels + caster)
- Mass: 10kg base + 1kg wheels

**Sensors:**
- **RGB Camera**: 640x480 @ 30Hz, 60° FOV
  - Topic: `/om1_test_robot/camera/image_raw`
- **Depth Camera**: 640x480 @ 20Hz, 0.1-10m range
  - Topic: `/om1_test_robot/depth_camera/depth`
- **IMU**: 100Hz update rate, gaussian noise
  - Topic: `/om1_test_robot/imu/data`
- **Odometry**: 50Hz, publishes position and velocity
  - Topic: `/om1_test_robot/odom`

**Control:**
- Command Topic: `/om1_test_robot/cmd_vel`
- Type: `geometry_msgs/Twist`

## Installation

### Prerequisites
```bash
# Ubuntu 20.04/22.04
sudo apt update
sudo apt install -y \
    gazebo11 \
    libgazebo11-dev \
    ros-humble-desktop \
    ros-humble-gazebo-ros-pkgs \
    python3-pip

# Python dependencies
pip3 install rclpy
```

### Setup
```bash
# Clone repository (if not already done)
git clone https://github.com/OpenMind/OM1.git
cd OM1

# Source ROS2
source /opt/ros/humble/setup.bash

# Set Gazebo model path
export GAZEBO_MODEL_PATH=$(pwd)/models:$GAZEBO_MODEL_PATH
```

## Usage

### Option 1: Launch Gazebo World Only
```bash
# Using the launch script
./scripts/launch_demo.sh

# Or manually
gazebo worlds/bounty363.world
```

### Option 2: Launch with ROS2 Demo
```bash
# Terminal 1: Launch Gazebo
./scripts/launch_demo.sh

# Terminal 2: Run demo navigation script
source /opt/ros/humble/setup.bash
python3 scripts/demo_navigation.py
```

### Option 3: Manual Robot Control
```bash
# Terminal 1: Launch Gazebo
./scripts/launch_demo.sh

# Terminal 2: Control robot manually
source /opt/ros/humble/setup.bash

# Move forward
ros2 topic pub /om1_test_robot/cmd_vel geometry_msgs/Twist \
  '{linear: {x: 0.5, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'

# Turn
ros2 topic pub /om1_test_robot/cmd_vel geometry_msgs/Twist \
  '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.5}}'

# Stop
ros2 topic pub /om1_test_robot/cmd_vel geometry_msgs/Twist \
  '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'
```

## Monitoring Sensor Data
```bash
# View camera image
ros2 run rqt_image_view rqt_image_view /om1_test_robot/camera/image_raw

# View depth data
ros2 run rqt_image_view rqt_image_view /om1_test_robot/depth_camera/depth

# Monitor IMU
ros2 topic echo /om1_test_robot/imu/data

# Monitor odometry
ros2 topic echo /om1_test_robot/odom

# List all topics
ros2 topic list
```

## Demo Navigation Script

The `demo_navigation.py` script demonstrates:
1. Sensor data collection (Camera, Depth, IMU)
2. Basic movement patterns
3. Data statistics and reporting

**Demo Sequence:**
1. Initialize and wait for sensors (3s)
2. Move forward (3s)
3. Turn left (2s)
4. Move forward (2s)
5. Turn right (2s)
6. Return (2s)
7. Report statistics

## File Structure
```
OM1/
├── worlds/
│   └── bounty363.world              # Enhanced Gazebo world
├── models/
│   └── om1_test_robot/
│       ├── model.config             # Model metadata
│       ├── model.sdf                # Robot model with sensors
│       ├── meshes/                  # (Future: 3D meshes)
│       └── materials/               # (Future: textures)
├── scripts/
│   ├── gazebo_setup.sh              # Environment setup
│   ├── gazebo_run_deterministic.sh  # Deterministic launcher
│   ├── launch_demo.sh               # Simple demo launcher
│   └── demo_navigation.py           # Python demo script
└── docs/
    └── bounty363_README.md          # This file
```

## Deterministic Setup

For reproducible simulations:
```bash
# Run with fixed seed
./scripts/gazebo_run_deterministic.sh

# Or set manually
export GAZEBO_SEED=12345
gazebo worlds/bounty363.world
```

## Troubleshooting

### Issue: Robot not spawning

**Solution:**
```bash
# Check model path
echo $GAZEBO_MODEL_PATH

# Should include your models directory
export GAZEBO_MODEL_PATH=$(pwd)/models:$GAZEBO_MODEL_PATH
```

### Issue: No sensor data

**Solution:**
```bash
# Check if topics exist
ros2 topic list | grep om1_test_robot

# Check Gazebo plugins loaded
gz model -m om1_robot -i
```

### Issue: Gazebo crashes

**Solution:**
```bash
# Check Gazebo version
gazebo --version

# Reinstall if needed
sudo apt install --reinstall gazebo11 libgazebo11-dev
```

### Issue: Demo script not working

**Solution:**
```bash
# Check ROS2 sourced
echo $ROS_DISTRO

# Re-source
source /opt/ros/humble/setup.bash

# Check Python dependencies
pip3 install rclpy
```

## Performance Tips

1. **Reduce sensor rates** for slower machines:
   - Edit `models/om1_test_robot/model.sdf`
   - Change `<update_rate>` values

2. **Disable visualization**:
   - Set `<visualize>false</visualize>` in sensors

3. **Run headless**:
```bash
   gzserver worlds/bounty363.world
```

## Integration with OM1

To integrate with OM1 core system:

1. Subscribe to sensor topics in OM1 nodes
2. Use robot control interface via cmd_vel
3. Process sensor data for decision making
4. Stream data to OM1 cortex

Example integration point:
```python
# In your OM1 node
self.camera_sub = self.create_subscription(
    Image,
    '/om1_test_robot/camera/image_raw',
    self.process_vision_data,
    10
)
```

## Future Enhancements

- [ ] Add LiDAR sensor
- [ ] Dynamic obstacles (moving objects)
- [ ] Multiple robot scenarios
- [ ] Textured materials and meshes
- [ ] More complex navigation challenges
- [ ] Integration with OM1 planning system

## Credits

- **Author**: Wanbogang
- **Bounty**: #363
- **Date**: December 2024
- **License**: Apache 2.0 (same as OM1)

## Support

For issues or questions:
- Open issue on GitHub: https://github.com/OpenMind/OM1/issues
- Reference: Bounty #363
- Tag: @Wanbogang
