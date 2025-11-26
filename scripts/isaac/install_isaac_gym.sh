#!/bin/bash
# Installation script for Isaac Gym + OM1 Integration
# Bounty #364

echo "============================================"
echo "OM1 Isaac Gym Integration Setup"
echo "Bounty #364"
echo "============================================"

# Check for NVIDIA GPU
echo -e "\n[1/6] Checking NVIDIA GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. NVIDIA drivers required."
    echo "Install drivers: sudo apt install nvidia-driver-535"
    exit 1
fi

nvidia-smi
echo "✓ NVIDIA GPU detected"

# Check CUDA
echo -e "\n[2/6] Checking CUDA installation..."
if ! command -v nvcc &> /dev/null; then
    echo "WARNING: CUDA toolkit not found. Installing..."
    sudo apt update
    sudo apt install -y nvidia-cuda-toolkit
fi

nvcc --version
echo "✓ CUDA toolkit installed"

# Python environment
echo -e "\n[3/6] Setting up Python environment..."
if ! command -v python3.8 &> /dev/null; then
    echo "Installing Python 3.8..."
    sudo apt install -y python3.8 python3.8-venv python3.8-dev
fi

python3.8 -m venv isaac_gym_env
source isaac_gym_env/bin/activate

echo "✓ Python environment created"

# Install dependencies
echo -e "\n[4/6] Installing Python dependencies..."
pip install --upgrade pip
pip install -r isaac_gym_integration/requirements.txt

echo "✓ Dependencies installed"

# Isaac Gym installation
echo -e "\n[5/6] Isaac Gym installation..."
echo "Isaac Gym requires manual download from NVIDIA:"
echo "  1. Visit: https://developer.nvidia.com/isaac-gym"
echo "  2. Register (free) and download Isaac Gym Preview 4"
echo "  3. Extract to ~/isaacgym"
echo ""
echo "After downloading, run:"
echo "  cd ~/isaacgym/python"
echo "  pip install -e ."

if [ -d "$HOME/isaacgym" ]; then
    echo "Found isaacgym directory, installing..."
    cd ~/isaacgym/python
    pip install -e .
    cd -
    echo "✓ Isaac Gym installed"
else
    echo "⚠ Isaac Gym not found at ~/isaacgym"
    echo "Please download and install manually"
fi

# Test installation
echo -e "\n[6/6] Testing installation..."
python3 -c "import numpy; import torch; import yaml; import websockets; print('✓ All Python packages working')"

echo -e "\n============================================"
echo "Setup complete!"
echo "============================================"
echo ""
echo "To run the OM1 Isaac Gym integration:"
echo "  source isaac_gym_env/bin/activate"
echo "  python3 scripts/isaac/run_om1_isaac.py"
echo ""
echo "Note: Ensure Isaac Gym is downloaded and installed"
echo "============================================"
