---
title: Troubleshooting Guide
description: "Guide to troubleshoot some common issues"
icon: wrench
---

### Common Issues & Quick Fixes

| Issue                   | Likely Cause                | Quick Fix                                                                 |
|-------------------------|-----------------------------|---------------------------------------------------------------------------| 
| No Speech               | Permission issues           | Check the settings                                                        |
| No speech recognition   | Microphone not configured   | Check audio input settings                                                |
| Robot not moving        | Connection issue/Network issue            | Restart OM1/Robot and check your internet connection                                        |
| OpenSSL certificate issue | Security certificate not found | `uv pip install certifi` `export SSL_CERT_FILE=$(python3 -m certifi)` `export REQUESTS_CA_BUNDLE=$(python3 -m certifi)` |
| Error message during build: `fatal error: portaudio.h: No such file or directory compilation terminated. error: command '/usr/bin/gcc' failed with exit code 1` | The issue is due to python-all-dev being deprecated and unavailable in non standard Ubuntu installations. | Installing only PortAudio development headers fixes the problem: `sudo apt-get update` `sudo apt-get install portaudio19-dev`|

### Running OM1 Headless or on VMs

When running OM1 on headless servers or virtual machines, you may encounter the following non-fatal warnings. These are expected and generally safe to ignore:

| Warning | Cause | Resolution |
|---------|-------|------------|
| `ALSA lib confmisc.c:855:(parse_cards) cannot find card '0'` | No audio hardware present | Safe to ignore if audio input/output is not required. Set `TTS` and `ASR` to `null` in your config. |
| `ALSA lib pcm.c:2707:(snd_pcm_open_nofree) Unknown PCM cards.pcm.rear` | ALSA cannot find specific audio devices | Install `pulseaudio` or configure ALSA dummy driver: `sudo modprobe snd-dummy` |
| `jack server is not running or cannot be started` | JACK audio server not available | Safe to ignore unless using JACK. Configure PulseAudio instead if needed. |
| `WARNING: No display found. Disabling WebSim.` | No X11 display available | Expected in headless mode. WebSim requires a display. Use `--no-websim` flag or set `DISPLAY=:0` with Xvfb. |
| `libGL error: No matching fbConfigs or visuals found` | OpenGL libraries expect display hardware | Install mesa libraries: `sudo apt-get install libgl1-mesa-glx` or use software rendering. |
| `PulseAudio: Unable to connect: Connection refused` | PulseAudio daemon not running | Start PulseAudio: `pulseaudio --start` or configure for system mode. |

**Running with Virtual Display (Xvfb)**

If you need WebSim in a headless environment:

```bash
# Install Xvfb
sudo apt-get install xvfb

# Start virtual display
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99

# Now run OM1
uv run src/run.py spot
```

**Disabling Unused Features**

To suppress warnings, disable unused features in your config:

```json5
{
  "agent_inputs": [
    // Remove or comment out audio-related inputs
    // { "type": "GoogleASRInput" }
  ],
  "agent_actions": [
    // Remove or comment out audio-related actions
    // { "name": "speak", "connector": "elevenlabs_tts" }
  ]
}
```

