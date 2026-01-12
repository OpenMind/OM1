#!/bin/ash
export ROS_DISTRO=humble
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
unset CYCLONEDDS_URI
export ROS_DOMAIN_ID=0

/zenoh-bridge-ros2dds -c /app/config_rpi.json5

echo "zenoh-bridge-ros2dds that captures RPi DDS messages has started."
