# OM1 Troubleshooting Guide

This guide provides solutions to the most common errors encountered when installing, configuring, or running the OM1 robotics runtime across macOS, Linux, and Docker environments.

Its purpose is to help developers and contributors resolve issues quickly and reduce duplicated bug reports across the community.

---

## 1. Python Version Mismatch (macOS)

### Symptoms
Unable to connect to any of [tcp/127.0.0.1:7447]
ModuleNotFoundError
python: error while loading shared libraries

### Cause
OM1 currently supports **Python 3.10** on macOS.  
Using Python 3.11 may cause:
- Zenoh failing to start  
- Module import errors  
- Native library issues  

### Fix
Install Python 3.10:

```bash
uv python install 3.10
uv venv --python 3.10
```

Or with pyenv:

```bash
pyenv install 3.10.14
pyenv local 3.10.14
```

## 2. Zenoh Unable to Connect

### Error
Unable to connect to any of [tcp/127.0.0.1:7447]

### Cause
- Running OM1 without a robot
- GovernanceEthereum enabled in configuration
- Incorrect runtime environment

### Fix
In your agent config (config/*.json5), comment out:

```json5
{
  "type": "GovernanceEthereum"
}
```

Example:

```json5
"agent_inputs": [
  // {
  //   "type": "GovernanceEthereum"
  // },
  {
    "type": "VLM_COCO_Local",
    "config": {
      "camera_index": 0
    }
  }
]
```
## 3. FFmpeg Not Installed (Audio / TTS Fails)

### Error

```arduino
ffmpeg not found
Audio backend initialization failed
```

### Cause
FFmpeg is required for:
- Text-to-Speech
- Audio playback
- Media pipeline

### Fix
macOS:

```bash
brew install ffmpeg
```

Ubuntu:

```bash
sudo apt update
sudo apt install ffmpeg
```

## 4. PortAudio Missing (Microphone Errors)

### Error

```makefile
OSError: PortAudio library not found
```

### Cause
PortAudio is needed for microphone capture when using audio inputs

### Fix
macOS:

```bash
brew install portaudio
```

Ubuntu:

```bash
sudo apt-get install portaudio19-dev python3-dev
```

## 5. Docker Permission Denied

### Error

```pgsql
permission denied while trying to connect to docker daemon
```

### Cause
User account does not have permission to access Docker

### Fix (Linux)

```bash
sudo usermod -aG docker $USER
newgrp docker
```

## 6. WebSim / Debug UI Not Loading

### Symptoms
- http://localhost:8000 not loading
- Blank page
- WebSim service down

### Fix
Check if service is running:

```bash
docker ps | grep om1_avatar
```

If container not running:

```bash
docker compose up om1_avatar
```

## 7. Model File or VLM Not Found

### Error

```lua
Model file missing
Could not load VLM_COCO_Local
```

### Cause
- Model not downloaded
- Wrong model path
- Missing permissions

### Fix
Run:

```bash
uv pip install --upgrade pillow torchvision torch
```

Ensure your model path in config:

```bash
"model_path": "./models/vlm_coco"
```

## 8. Docker Build Fails on macOS ARM (M1/M2/M3)

### Cause
Some base images default to x86.

### Fix
Force platform:

```bash
docker build --platform linux/amd64 -t om1 .
```

Or use multi-arch:

```bash
docker buildx build --platform linux/arm64,linux/amd64 .
```

## 9. Missing Submodules After Clone

### Symptoms
- src empty
- unitree_sdk missing
- build errors

### Fix
Run:

```bash
git submodule update --init --recursive
```

## 10. Common macOS System Dependencies
If anything unexpected fails:

```bash
brew install cmake pkg-config openssl
brew upgrade
```

## Need More Help?
- Review the OM1 README
- Check active GitHub issues
- Join the OpenMind Discord community
- Search OM1 related repositories listed under the ecosystem









