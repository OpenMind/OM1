---
title: Troubleshooting
description:
icon: gear
---

## Gazebo Specific Issues

### colcon build fails

**Problem:** Build process exits with errors.

**Solution:**
1. Ensure you're not running `colcon build` inside a virtual environment
2. Deactivate any active environments (including conda)
3. Open a fresh terminal and retry the build

### Gazebo and RViz fail to launch

**Problem:** Simulation environment doesn't start properly.

**Solution:**
- Verify CycloneDDS is configured correctly on your system
- Check that ROS 2 middleware is properly initialized

### Robot stops moving unexpectedly

**Problem:** Robot is unresponsive to movement commands.

**Solution:**
- Manually reposition the robot in Gazebo using translate or rotate mode
- Alternatively, use RViz to send a 2D Nav Goal pose to the robot

### Orchestrator throws errors

**Problem:** API communication or initialization fails.

**Solution:**
1. Verify environment variables are set in your `~/.bashrc`:

   ```bash
   export OM_API_KEY=<your_key>
   export OM_API_KEY_ID=<your_key_id>
   ```

2. Confirm your virtual environment is active with Python 3.10:

   ```bash
   python --version
   which python
   ```

**Problem:** Packages not found.

**Solution:**
1. Confirm your virtual environment is active with Python 3.10:

   ```bash
   python --version
   which python
   ```

2. Confirm you installed the dependencies using `uv pip install`, during the setup. If you still face issues, try deleting and creating a new virtual environment. Make sure to export `PYTHONPATH` to correct location.

## Audio Issues

### Robot doesn't respond to voice commands

**Problem:** Audio input/output is not working.

**Solution:**
- Check system audio settings
- Verify the correct microphone is selected as input
- Verify the correct speaker is selected as output
- Test audio with a simple recording to confirm functionality
