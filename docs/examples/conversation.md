---
title: Conversation
description: "Using Cloud Endpoints for Voice Inputs and Text to Speech"
icon: comment
---

This section provides various examples for integrating and using multiple cloud-based AI endpoints, such as OpenAI, DeepSeek, and others, for voice input processing, text-to-speech (TTS) and emotion detection. Whether you need to convert spoken language into text (ASR) or generate natural-sounding speech from text, these examples will help you interact with different cloud providers seamlessly.

## Voice to Text Processing with OpenAI

This example uses your `default` audio in (microphone) and your `default` audio output (speaker). Please test both your microphone and speaker in your system settings to make sure they are connected and working. On a Mac, the system may request permission to access your audio - Allow permissions.

```bash
make run CONFIG=conversation
```

Especially on Linux, such as on Ubuntu 20.04 on the Nvidia Orin, audio support can be marginal. Expect some audio inputs and outputs to not work correctly, or to advertise incorrect hardware capabilities, such as USB microphones that report zero input channels etc.

## Selecting and testing your audio devices

OM1 uses your operating system's default microphone and speaker (via `portaudio`). Set the correct defaults in your OS sound settings (macOS: **System Settings → Sound**; Linux: `pavucontrol` or `alsamixer`) and test them there before running the agent.

## Debugging audio

Run in dev mode to get verbose logs, including the ASR input and TTS playback state, which help diagnose audio issues:

```bash
make dev CONFIG=conversation
```

The ASR input's capture settings are controlled by its `config` block in the agent config — for example `GoogleASRInput` accepts `rate` (e.g. `16000`) and `chunk` (e.g. `1600`). Tune these if the microphone reports unexpected capabilities.
