#!/usr/bin/env bash
# ------------------------------------------------------------------
# Script: run_gazebo.sh
# Description: Cross-platform Gazebo Harmonic launcher for Ubuntu and macOS
# Usage: ./run_gazebo.sh [model_file.sdf]
# ------------------------------------------------------------------

set -e  # Exit on error

# Detect operating system
OS="$(uname)"
if [[ "$OS" == "Darwin" ]]; then
    PLATFORM="macos"
elif [[ "$OS" == "Linux" ]]; then
    PLATFORM="linux"
else
    echo "Error: Unsupported platform: $OS"
    echo "This script only supports macOS and Linux/Ubuntu"
    exit 1
fi

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Function to cleanup all Gazebo processes
cleanup() {
    echo -e "\nStopping all Gazebo processes..."
    pkill -f "gz sim" 2>/dev/null
    sleep 0.5
    pkill -9 -f "gz sim" 2>/dev/null
    echo "All Gazebo processes stopped."
    exit 0
}

# Check if gz is installed
if ! command -v gz &> /dev/null; then
    echo "Error: Gazebo Harmonic (gz) is not installed."
    if [[ "$PLATFORM" == "macos" ]]; then
        echo "To install on macOS, run:"
        echo "  brew tap osrf/simulation"
        echo "  brew install gz-harmonic"
        echo ""
        read -p "Would you like to install Gazebo Harmonic now? (y/n): " REPLY
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Installing Gazebo Harmonic..."
            brew tap osrf/simulation
            brew install gz-harmonic
            echo "Installation complete!"
        else
            echo "Installation cancelled."
            exit 1
        fi
    else
        echo "Gazebo Harmonic is not installed on Ubuntu."
        echo ""
        read -p "Would you like to install Gazebo Harmonic now? (y/n): " REPLY
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Installing dependencies..."
            sudo apt-get update && sudo apt-get install -y curl lsb-release gnupg

            echo "Adding Gazebo repository..."
            sudo curl https://packages.osrfoundation.org/gazebo.gpg --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
            echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] https://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

            echo "Installing Gazebo Harmonic..."
            sudo apt-get update && sudo apt-get install -y gz-harmonic

            echo "Installation complete!"
        else
            echo "Installation cancelled."
            echo ""
            echo "For installation guide: https://gazebosim.org/docs/harmonic/install_ubuntu"
            exit 1
        fi
    fi
fi

# Set resource path
export GZ_SIM_RESOURCE_PATH="$SCRIPT_DIR"
echo "GZ_SIM_RESOURCE_PATH: $GZ_SIM_RESOURCE_PATH"

# Get model file
FILE_NAME="${1:-model.sdf}"
MODEL_FILE="$SCRIPT_DIR/gz/models/$FILE_NAME"

if [ ! -f "$MODEL_FILE" ]; then
    echo "Error: Model file not found at $MODEL_FILE"
    exit 1
fi

# Start simulation server
echo "Starting Gazebo simulation with: $FILE_NAME"
gz sim "$MODEL_FILE" -s &

# Start GUI in new terminal (platform-specific)
echo "Opening new terminal for GUI..."

if [[ "$PLATFORM" == "macos" ]]; then
    # macOS: Use AppleScript to open Terminal
    osascript <<EOF
tell application "Terminal"
    do script "export GZ_SIM_RESOURCE_PATH='$SCRIPT_DIR'; gz sim -g"
end tell
EOF

elif [[ "$PLATFORM" == "linux" ]]; then
    # Linux: Try common terminal emulators
    if command -v gnome-terminal &> /dev/null; then
        gnome-terminal -- bash -c "export GZ_SIM_RESOURCE_PATH='$SCRIPT_DIR'; gz sim -g; exec bash"
    elif command -v x-terminal-emulator &> /dev/null; then
        x-terminal-emulator -e bash -c "export GZ_SIM_RESOURCE_PATH='$SCRIPT_DIR'; gz sim -g; exec bash"
    else
        echo "Warning: No suitable terminal found. Run this in a new terminal:"
        echo "  export GZ_SIM_RESOURCE_PATH='$SCRIPT_DIR'; gz sim -g"
    fi
fi

echo "Press Ctrl+C to stop all Gazebo processes"

# Set up cleanup on exit
trap cleanup INT TERM EXIT

# Keep script running
wait
