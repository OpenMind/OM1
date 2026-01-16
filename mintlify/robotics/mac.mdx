---
title: Apple Mac for Robotics
description: "Apple Mac for Robotics"
---

## Hardware

For the Mac Mini, add a 12V input directly to the power rail. You can figure this out yourself, or, find instructions on YouTube.

## Apple Remote Desktop (ARD)

Apple Remote Desktop is a good solution for developing, except that the audio is automatically routed to your development computer when you connect, which then prevents you from remotely debugging on-robot audio hardware issues.

**Solution 1** Stop ARD completely, connect via SSH, and in `system_hw_test`, run:

```bash
uv run test_audio_mac.py
```

**Solution 2** Stop ARD completely, connect via SSH, and run:

```bash
osascript -e 'set volume without output muted'
osascript -e 'set volume output volume 20' # adjust int to whatever value you want
```

**Solution 3** The Mac _might_ automatically unmute the default audio device once the ARD session is terminated.

## FileVault / Auto Connect to Wifi

FileVault blocks auto connect to WiFi, making it impossible to remote connect to your Mac after it boots (unless you have a screen/keyboard connected to it), which sadly defeats the entire purpose of ARD.

**Solution** Turn off FileVault, or, do not provide AppleID credentials, in which case FileVault will be off by default.
