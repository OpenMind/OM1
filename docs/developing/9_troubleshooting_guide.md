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
| Build errors            | Missing dependencies        | Run `make deps` to download zenoh-c and Go dependencies |
| PortAudio error during build | Missing PortAudio headers | `sudo apt-get update` `sudo apt-get install portaudio19-dev`|
