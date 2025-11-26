# Isaac Gym Setup Guide - Step by Step

Complete installation guide for OM1 Isaac Gym integration on Ubuntu 22.04.

## Prerequisites Checklist

- [ ] Ubuntu 20.04 or 22.04
- [ ] NVIDIA GPU (GTX 1650 or better)
- [ ] 4GB+ VRAM
- [ ] Internet connection

## Step 1: Verify GPU

```bash
# Check GPU
lspci | grep -i nvidia

# Expected: NVIDIA Corporation Device... GTX/RTX...

Step 2: Install NVIDIA Drivers
# Check current driver
nvidia-smi

# If not working, install
sudo apt update
sudo apt install nvidia-driver-535 -y
sudo reboot

# After reboot, verify
nvidia-smi

Expected output:
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.xx       Driver Version: 535.xx       CUDA Version: 12.2   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0 On  |                  N/A |


Step 3: Install CUDA Toolkit
sudo apt install nvidia-cuda-toolkit -y

# Verify
nvcc --version

# Expected: cuda_12.x

Step 4: Install Python 3.8
sudo apt install python3.8 python3.8-venv python3.8-dev -y

# Verify
python3.8 --version

## Step 5: Download Isaac Gym

1. Open browser: [https://developer.nvidia.com/isaac-gym](https://developer.nvidia.com/isaac-gym)
2. Click "Join now" (free registration)
3. Fill form and verify email
4. Login and download "Isaac Gym Preview 4"
5. Save to `~/Downloads`


## Step 6: Extract Isaac Gym
cd ~/Downloads
tar -xvf IsaacGym_Preview_4_Package.tar.gz -C ~/

# Verify
ls ~/isaacgym
# Expected: python/ docs/ ...

Step 7: Install Isaac Gym
cd ~/isaacgym/python
python3.8 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -e .


Step 8: Test Isaac Gym
cd ~/isaacgym/python/examples
python joint_monkey.py

Expected: Window opens showing physics simulation with monkeys

**Press ESC to close**

## Step 9: Install OM1 Integration
cd ~/OM1
git checkout bounty-364-isaac-gym

# Create virtual environment
python3.8 -m venv isaac_gym_env
source isaac_gym_env/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r isaac_gym_integration/requirements.txt

# Install Isaac Gym in this env too
cd ~/isaacgym/python
pip install -e .
cd ~/OM1


## Step 10: Configure OM1 API

Edit API key if needed:
nano isaac_gym_integration/cfg/om1_robot.yaml

Your API key is already configured:
om1:
  api_key: "om1_live_482c8015..."

Step 11: Run OM1 Isaac Gym
source isaac_gym_env/bin/activate
python3 scripts/isaac/run_om1_isaac.py

Expected output:
============================================================
OM1 Isaac Gym Integration - Bounty #364
============================================================

[1/4] Creating Isaac Gym environment...
[2/4] Setting up ground plane and obstacles...
[3/4] Connecting to OM1 API...
✓ Connected to OM1 API at wss://api.openmind.org
[4/4] Starting simulation with sensor streaming...

Controls:
  - ESC: Exit
  - Sensor data streaming to OM1 API in real-time

============================================================

**Window opens with 4 robots in parallel environments**

## Verification

### Check GPU Usage

In another terminal:
watch -n 1 nvidia-smi

You should see:

- GPU Utilization: 60-90%
- Memory Used: 2000-3000 MB


### Check Sensor Streaming

OM1 API should receive:

- LiDAR scans @ 10 Hz
- IMU data @ 100 Hz
- Camera frames @ 30 Hz (if enabled)


## Common Issues

### "ImportError: No module named 'isaacgym'"

Solution:
cd ~/isaacgym/python
pip install -e .

### "RuntimeError: CUDA out of memory"

Solution: Reduce parallel environments
nano isaac_gym_integration/cfg/om1_robot.yaml
# Change: num_envs: 1


### Window doesn't open

Solution: Check X11 forwarding
echo $DISPLAY
# Should show: :0 or :1

# If empty:
export DISPLAY=:0

## Next Steps

- Experiment with different num_envs (1, 2, 4, 8...)
- Modify robot parameters in config
- Add custom obstacles
- Implement navigation algorithms
- Train RL policies


## Success!

You now have OM1 running in Isaac Gym with full sensor integration and OM1 API streaming.

Bounty #364 complete!
