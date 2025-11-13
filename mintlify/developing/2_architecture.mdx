---
title: Core Architecture
description: "Core Architecture and Runtime Flow"
---

## Architecture Overview

This system diagram illustrates some of OM1's layers and modules.

![Architecture Diagram](../assets/om1-architecture.png)

### Raw Sensor Layer

The sensors provide raw inputs:

- Vision: Cameras for visual perception.
- Sound: Microphones capturing audio data.
- Battery/System: Monitoring battery and system health.
- Location/GPS: Positioning information.
- LIDAR: Laser-based sensing for 3D mapping and navigation.

### AI Captioning and Compression Layer

These models convert raw sensor data into meaningful descriptions:

- VLM (Vision Language Model): Converts visual data to natural language descriptions (e.g., human activities, object interactions).
- ASR (Automatic Speech Recognition): Converts audio data into text.
- Platform State: Describes internal system status (e.g battery percentage, odometry readings).
- Spatial/NAV: Processes location and navigation data.
- 3D environments: Interprets 3D environmental data from sensors like LIDAR.

### Natural Language Data Bus (NLDB)

A centralized bus that collects and manages natural language data generated from various captioning/compression modules, ensuring structured data flow between components.

Example messages might include:

```bash
Vision: “You see a human. He looks happy and is smiling and pointing to a chair.”
Sound: “You just heard: Bits, run to the chair.”
Odom: 1.3, 2.71, 0.32
Power: 73%
```

### State Fuser

This module combines short inputs from the NLDB into one paragraph, providing context and situational awareness to subsequent decision-making modules. It fuses spatial data (e.g. the number and relative location of proximal humans and robots), audio commands, and visual cues into a unified, compact, description of the robot's current world.

Example fuser output:

```bash
137.0270: You see a human, 3.2 meters to your left. He looks happy and is smiling. He is pointing to a chair. You just heard: Bits run to the chair.
139.0050: You see a human, 1.5 meters in front of you. He is showing you a flat hand. You just heard: Bits, stop.
```

### Multi AI Planning/Decision Layer

Uses fused data to make decisions through one or more AI models. A typical multi-agent endpoint wraps three of more LLMs:

- Fast Action LLM (Local or Cloud): A small LLM that quickly processes immediate or time-critical actions without significant latency. Expected token response time - 300 ms.
- Cognition ("Core") LLM (Cloud): Cloud-based LLM for complex reasoning, long-term planning, and high-level cognitive tasks, leveraging more computational resources. Expected token response time - 2 s.
- Mentor/Coach LLM (Cloud): Cloud-based LLM for 3rd person view critique of the robot-human interaction. Generates full critique every 30 seconds and provides it to the Core LLM.

These LLMs are constrained by natural language rules provided in the configuration files, or, downloaded from immutable public ledgers (blockchains) such as Ethereum. Storing robot constitutions/guardrails on immutable public ledgers facilitates transparency, traceability, decentralized coordination, and logging for accountability.

Feedback Loop:

- Adjustments based on performance metrics or environmental conditions (e.g., adjusting vision frame rates for efficiency).

### Hardware Abstraction Layer (HAL)

This layer translates high-level AI decisions into actionable commands for robot hardware. It's responsible for converting a high level decision such as "pick up the red apple with your left hand" into the succession of gripper arm servo commands that results in the apple being picked up. Typical `action` modules handle:

- Move: Controls robot movement.
- Sound: Generates auditory signals.
- Speech: Handles synthesized voice outputs.
- Wallet: Digital wallet for economic transactions or cryptographic operations for identity verification.

In many cases, this is where AI decisions are mapped onto existing ROS2 functionalities, and/or CycloneDDS or Zenoh middleware.

### Overall System Data Flow

Raw Sensors → AI Captioning/Compression (Audio, LIDAR, Spatial RAG, Vision models) → NLDB → Data Fuser → AI Decision Layer (Emergency Responder LLM, Core LLM, Coach LLM) → HAL → Robot Actions ("Foundational" models, ROS2 code, movement policies, action models)
