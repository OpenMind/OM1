---
title: Face Detection & Anonymization
description: "Privacy-aware perception: detect and anonymize faces in the robot's video."
icon: user-shield
---

Put a camera-equipped robot in a public space or a workplace and privacy stops being optional. OM1 can detect faces in the live video feed and **anonymize** them — obscuring people who aren't part of an interaction — before the footage is used or streamed. The agent can still reason about what it sees ("a person is present") without keeping identifiable images.

This is a managed capability of the OM1 autonomy stack: you enable and configure it through the [OpenMind portal](https://portal.openmind.com), not the robot's local API, and it runs on the GPU-accelerated video pipeline on supported platforms. Because it touches how footage is captured and retained, it's worth aligning with your OpenMind contact on compliance and retention when you turn it on.

For the bigger picture, see the [Autonomy Overview](../architecture_overview.md).
