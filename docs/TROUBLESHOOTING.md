# OM1 Troubleshooting Guide

This guide helps you diagnose and resolve common issues with OM1.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Configuration Problems](#configuration-problems)
- [Runtime Errors](#runtime-errors)
- [API Issues](#api-issues)
- [Performance Problems](#performance-problems)
- [Hardware Issues](#hardware-issues)

---

## Installation Issues

### uv: command not found

**Problem**: `uv` package manager is not installed or not in PATH.

**Solution**:
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH (add to ~/.bashrc or ~/.zshrc)
export PATH="$HOME/.local/bin:$PATH"

# Reload shell
source ~/.bashrc  # or source ~/.zshrc
```

---

### Python version too old

**Problem**: OM1 requires Python 3.7+, but system has older version.

**Solution**:
```bash
# Check Python version
python3 --version

# If version < 3.7, install newer Python:
# macOS (using Homebrew):
brew install python@3.11

# Linux (using deadsnakes PPA):
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.11
```

---

### Module not found errors

**Problem**: Import errors for required packages.

**Solution**:
```bash
# Reinstall dependencies
uv pip install -e .

# Or install specific package
uv pip install package_name
```

---

## Configuration Problems

### JSON parse errors

**Problem**: Configuration file has syntax errors.

**Solution**:
```bash
# Validate configuration
python scripts/validate_config.py config/your_config.json5

# Common JSON5 issues:
# - Trailing commas (allowed in JSON5)
# - Comments (allowed in JSON5)
# - Missing quotes around keys (not allowed)
# - Unmatched brackets/braces
```

**Example Fix**:
```json5
// Wrong
{name: "robot",}  // Missing quotes, trailing comma

// Right
{"name": "robot"}  // JSON5 allows both
```

---

### Port 8000 already in use

**Problem**: WebSim fails to start because port 8000 is occupied.

**Solution**:
```json5
{
  "simulators": [
    {
      "type": "WebSim",
      "config": {
        "port": 8001  // Use different port
      }
    }
  ]
}
```

**Find what's using port 8000**:
```bash
# macOS/Linux
lsof -i :8000

# Kill the process
kill -9 <PID>
```

---

### API key errors

**Problem**: `unauthorized` or `invalid API key` errors.

**Solution**:

1. **Check .env file**:
```bash
cat .env
# Should contain:
# OM_API_KEY=om1_live_...
# URID=...
```

2. **Verify API key is active**:
   - Visit https://fabric.openmind.org
   - Check your API key status
   - Ensure URID matches API key

3. **Add API key to config** (alternative to .env):
```json5
{
  "api_key": "om1_live_your_actual_key_here",
  "URID": "your_urid_here"
}
```

---

## Runtime Errors

### Robot doesn't respond

**Problem**: OM1 starts but robot doesn't react to inputs.

**Possible Causes**:

1. **No inputs configured**:
```json5
{
  "agent_inputs": []  // Empty - only responds to audio
}
```

2. **Hertz too low**:
```json5
{
  "hertz": 0.01  // Too slow - increase to 0.1
}
```

3. **No actions configured**:
```json5
{
  "agent_actions": []  // Empty - no actions available
}
```

**Solution**: Add inputs and actions to configuration.

---

### Camera not working

**Problem**: VLM input shows black screen or errors.

**Solution**:

1. **Check camera index**:
```json5
{
  "agent_inputs": [
    {
      "type": "VLM_COCO_Local",
      "config": {"camera_index": 0}  // Try 0, 1, 2, etc.
    }
  ]
}
```

2. **Test camera**:
```bash
# macOS
system_profiler SP CamerasDataType

# Linux
ls /dev/video*
```

3. **Check permissions**:
```bash
# macOS: Allow camera access in System Preferences
# Linux: Add user to video group
sudo usermod -a -G video $USER
```

---

### Audio not working

**Problem**: Robot doesn't hear or speak.

**Solution**:

1. **Check audio devices**:
```bash
# List audio devices
python3 -c "import pyaudio; p = pyaudio.PyAudio(); [print(i, p.get_device_info_by_index(i)['name']) for i in range(p.get_device_count())]"
```

2. **Install ffmpeg** (required for audio playback):
```bash
# macOS
brew install ffmpeg

# Linux
sudo apt install ffmpeg
```

3. **Test audio**:
```bash
# Record test
arecord test.wav

# Playback test
aplay test.wav
```

---

## API Issues

### 402 Payment Required

**Problem**: API returns 402 error (insufficient balance).

**Solution**:

1. **Check balance**:
   - Visit https://fabric.openmind.org
   - Check your account balance
   - Add funds if needed

2. **Reduce API usage**:
```json5
{
  "hertz": 0.05,  // Reduce frequency
  "cortex_llm": {
    "config": {
      "history_length": 1  // Reduce context
    }
  }
}
```

3. **Use APILimiter**:
```json5
{
  "backgrounds": [
    {
      "type": "APILimiter",
      "config": {
        "max_requests_per_minute": 30,
        "max_cost_per_hour": 0.5
      }
    }
  ]
}
```

---

### API rate limiting

**Problem**: Too many requests, getting rate-limited.

**Solution**:

1. **Reduce hertz**:
```json5
{"hertz": 0.1}  // Instead of 1.0 or higher
```

2. **Add APILimiter background** (see above)

3. **Remove unused inputs**:
```json5
{
  "agent_inputs": [
    // Only keep inputs you actually need
    {"type": "VLM_COCO_Local"}
    // Remove audio if not used, etc.
  ]
}
```

---

## Performance Problems

### High CPU usage

**Problem**: OM1 using too much CPU.

**Solution**:

1. **Reduce hertz**:
```json5
{"hertz": 0.05}  // Lower frequency
```

2. **Remove resource-intensive inputs**:
```json5
{
  "agent_inputs": [
    // VLM_COCO_Local uses significant CPU
    // Consider removing if not needed
  ]
}
```

3. **Check with HealthCheck**:
```json5
{
  "backgrounds": [
    {
      "type": "HealthCheck",
      "config": {
        "check_interval": 60,
        "alert_threshold_cpu": 80.0
      }
    }
  ]
}
```

---

### High memory usage

**Problem**: Memory usage growing over time.

**Possible Causes**:

1. **Large history_length**:
```json5
{
  "cortex_llm": {
    "config": {
      "history_length": 10  // Too high
    }
  }
}
```

**Solution**: Reduce to 3 or lower.

2. **Memory leak in plugin**:
   - Check custom background plugins
   - Look for growing data structures
   - Use HealthCheck to monitor

---

### Slow response time

**Problem**: Long delay between input and robot response.

**Solution**:

1. **Check API latency**:
   - High hertz = more API calls
   - Reduce hertz if needed

2. **Reduce history_length**:
```json5
{
  "cortex_llm": {
    "config": {
      "history_length": 1  // Faster processing
    }
  }
}
```

3. **Use faster LLM** (if available):
```json5
{
  "cortex_llm": {
    "type": "FasterLLM",  // If available
    "config": {...}
  }
}
```

---

## Hardware Issues

### Robot not connecting

**Problem**: Real robot hardware not connecting.

**Solution**:

1. **Check robot_ip in config**:
```json5
{
  "robot_ip": "192.168.1.100"  // Your robot's IP
}
```

2. **Test connection**:
```bash
ping 192.168.1.100
```

3. **Check firewall**:
```bash
# Allow robot ports
sudo ufw allow 7447/tcp  # Zenoh
sudo ufw allow from 192.168.1.0/24
```

---

### Sensor data not appearing

**Problem**: GPS, IMU, or other sensors not working.

**Solution**:

1. **Check sensor is enabled**:
```json5
{
  "agent_inputs": [
    {
      "type": "GPS",  // or "IMU", "Odometry", etc.
      "config": {...}
    }
  ]
}
```

2. **Check robot is actually connected** (see above)

3. **Verify sensor topic**:
   - Use WebSim to check available topics
   - Check ROS2 topics: `ros2 topic list`

---

## Portal Issues

### Device shows OFF in Portal

**Problem**: Device appears offline at portal.openmind.org.

**Solution**:

1. **Ensure TeleopsConnection is configured**:
```json5
{
  "backgrounds": [
    {
      "type": "TeleopsConnection",
      "config": {
        "api_key": "om1_live_YOUR_KEY"
      }
    }
  ]
}
```

2. **Check API key**:
   - Must start with `om1_live_`
   - Must match your URID
   - Must be active

3. **Verify status updates**:
   - Check logs for "✓ Teleops status update sent"
   - Should appear every 5 seconds

---

### Can't control robot from Portal

**Problem**: Device shows online but can't control.

**Solution**:

1. **Check video connection**:
```json5
{
  "backgrounds": [
    {
      "type": "TeleopsConnection",
      "config": {
        "api_key": "...",
        "video_connected": true  // Ensure video is enabled
      }
    }
  ]
}
```

2. **Verify robot is actually running**:
   - Check OM1 logs
   - Ensure robot is not crashed

3. **Check WebSocket connection**:
   - Logs should show "Connected to wss://..."
   - No disconnect errors

---

## Getting Help

If you're still stuck:

1. **Check logs**:
```bash
# OM1 logs are printed to console
# Look for ERROR or WARNING messages
```

2. **Search existing issues**:
   - [GitHub Issues](https://github.com/OpenMind/OM1/issues)
   - [Discussions](https://github.com/OpenMind/OM1/discussions)

3. **Ask for help**:
   - Telegram: [OpenMind Dev Group](https://t.me/openminddev)
   - Create a new GitHub issue with:
     - Configuration file (sanitized)
     - Error messages
     - System information (OS, Python version)

4. **Use utility scripts**:
```bash
# Validate config
python scripts/validate_config.py config/your_config.json5

# Estimate costs
python scripts/api_cost_estimator.py config/your_config.json5
```

---

## Common Error Messages

### `ImportError: No module named 'om1_utils'`

**Cause**: OM1 not installed or installed incorrectly.

**Solution**:
```bash
cd /path/to/OM1
uv pip install -e .
```

---

### `KeyError: 'api_key'`

**Cause**: API key not configured.

**Solution**: Add to .env or config file (see [API key errors](#api-key-errors)).

---

### `AttributeError: 'NoneType' object has no attribute...`

**Cause**: Required dependency not loaded or failed to initialize.

**Solution**:
1. Check logs for earlier errors
2. Verify all dependencies installed
3. Check configuration is valid

---

### `RuntimeWarning: coroutine was never awaited`

**Cause**: Async/await mismatch in code (usually in WebSim).

**Solution**: This is usually a warning, not an error. If it causes problems:
- Update to latest OM1 version
- Check WebSim configuration

---

## Best Practices to Avoid Issues

1. **Always validate configuration**:
   ```bash
   python scripts/validate_config.py config/your_config.json5
   ```

2. **Estimate costs before running**:
   ```bash
   python scripts/api_cost_estimator.py config/your_config.json5
   ```

3. **Start with minimal config**:
   - Use `examples/minimal_config.json5` as starting point
   - Add features incrementally

4. **Use HealthCheck for monitoring**:
   ```json5
   {
     "backgrounds": [
       {"type": "HealthCheck", "config": {...}}
     ]
   }
   ```

5. **Keep API key secure**:
   - Never commit API keys to git
   - Use .env file (add to .gitignore)
   - Rotate compromised keys

---

## Additional Resources

- [Configuration Examples](../examples/README.md)
- [Background Plugins Guide](BACKGROUND_PLUGINS.md)
- [Utility Scripts](../scripts/README.md)
- [Main README](../README.md)
