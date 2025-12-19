#!/bin/bash

###############################################################################
# OM1 Test Robot Demo Launcher
# Bounty #363 - Enhanced Gazebo Environment
###############################################################################

set -e

echo "========================================"
echo "OM1 Test Robot Demo - Bounty #363"
echo "Enhanced Gazebo Environment with Sensors"
echo "========================================"
echo ""

# Check if Gazebo is installed
if ! command -v gazebo &> /dev/null; then
    echo "ERROR: Gazebo not found. Please install Gazebo first."
    echo "  Ubuntu: sudo apt install gazebo11 libgazebo11-dev"
    exit 1
fi

# Check if ROS2 is sourced
if [ -z "$ROS_DISTRO" ]; then
    echo "WARNING: ROS2 not sourced. Attempting to source..."
    if [ -f "/opt/ros/humble/setup.bash" ]; then
        source /opt/ros/humble/setup.bash
        echo "✓ ROS2 Humble sourced"
    elif [ -f "/opt/ros/foxy/setup.bash" ]; then
        source /opt/ros/foxy/setup.bash
        echo "✓ ROS2 Foxy sourced"
    else
        echo "ERROR: ROS2 not found. Please install ROS2 first."
        exit 1
    fi
fi

# Set Gazebo model path
export GAZEBO_MODEL_PATH=$(pwd)/models:$GAZEBO_MODEL_PATH
echo "✓ GAZEBO_MODEL_PATH set to: $(pwd)/models"

# Launch options
WORLD_FILE="$(pwd)/worlds/bounty363.world"
VERBOSE=${VERBOSE:-0}

echo ""
echo "Configuration:"
echo "  World file: $WORLD_FILE"
echo "  ROS Distro: $ROS_DISTRO"
echo "  Verbose: $VERBOSE"
echo ""

# Check if world file exists
if [ ! -f "$WORLD_FILE" ]; then
    echo "ERROR: World file not found: $WORLD_FILE"
    exit 1
fi

echo "Starting Gazebo with enhanced world..."
echo ""
echo "Robot will spawn at position (0, -5, 0)"
echo "Press Ctrl+C to stop"
echo ""

# Launch Gazebo
if [ "$VERBOSE" = "1" ]; then
    gazebo --verbose "$WORLD_FILE"
else
    gazebo "$WORLD_FILE"
fi
