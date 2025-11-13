#!/usr/bin/env bash
# ------------------------------------------------------------------
# Script: check_and_run.sh
#
# Description:
#   - Checks if the system is Ubuntu x86_64.
#   - Prints a warning if the system is not Ubuntu x86_64 but continues execution.
#   - Runs the start_simulation script inside the docker directory.
#
# ------------------------------------------------------------------

# Get system information
OS_NAME=$(uname -s)
ARCH=$(uname -m)
DISTRO=$(lsb_release -is 2>/dev/null || cat /etc/*release 2>/dev/null | grep '^NAME=')

# Check if the system is Ubuntu x86_64
if [[ "$OS_NAME" != "Linux" || "$ARCH" != "x86_64" || ! "$DISTRO" =~ "Ubuntu" ]]; then
    echo "Warning: This script is designed for Ubuntu x86_64."
    echo "You are running on: OS=$OS_NAME, Architecture=$ARCH, Distro=$DISTRO"
fi

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Define the simulation script path
SIM_SCRIPT="$SCRIPT_DIR/docker/start_simulation.sh"

# Check if the script exists before running
if [[ ! -f "$SIM_SCRIPT" ]]; then
    echo "Error: Simulation script not found at $SIM_SCRIPT"
    exit 1
fi

# Run the simulation script
echo "Running: bash $SIM_SCRIPT"
bash "$SIM_SCRIPT"
