---
title: Full Autonomy
description: "Full autonomy architecture and introduction"
icon: sitemap
---

This section describes the full autonomy architecture and deployment model for OM1 with BrainPack.

OM1 is a modular robotics intelligence platform that connects perception, language, and motion into a single runtime. In full autonomy mode, the robot operates independently — navigating its environment, responding to speech, streaming video, and making decisions — without requiring constant human input. All services run as containerized processes on the BrainPack, communicating over well-defined interfaces.

### Platform Support

| Platform | Support Level | Notes |
|----------|---------------|-------|
| **NVIDIA AGX Orin** | Limited | Suitable for deployments that do not require heavy ML inference workloads |
| **NVIDIA Thor** | Full | Recommended for ML-heavy autonomy workloads. Leverages GPU and DLA across navigation, vision, and audio processing |

### Robot Support

- Unitree Go2
- Unitree G1
- LimX Tron

---

## Architecture Overview

The full autonomy stack is built from modular, containerized services that communicate through well-defined interfaces. Each service has a single responsibility and can be updated, restarted, or replaced independently without affecting the rest of the system.

![ ](../.gitbook/assets/full-autonomy-assets/full_autonomy_architecture.png)

At a high level, sensor data flows from hardware into the ROS2 SDK, which publishes it as structured topics. OM1 consumes those topics alongside user input, runs them through the LLM, and emits action commands back to the robot. The video processor handles media as a parallel pipeline, and the avatar renders robot state on the display throughout.

---

## Open Source Components

### OM1 (`om1`)

OM1 is the central intelligence of the system. It acts as the orchestration layer between the robot's hardware, its sensors, and the language model — translating perception and user intent into physical action.

At runtime, OM1 maintains a continuous loop: it listens for speech via ASR, passes the transcription (along with relevant context and system state) to the configured LLM, receives a response, and dispatches the appropriate action — whether that is speaking a reply, issuing a movement command, or triggering a downstream service. This loop runs in real time, keeping the robot responsive to its environment and the people around it.

| Feature | Description |
|---------|-------------|
| **Basic Movement** | Translates high-level motion intents (move forward, turn left, stop) into low-level velocity commands sent to the robot's motor controller. Movement is coordinated with the navigation stack when the ROS2 SDK is available, or executed directly when operating in open-loop mode. |
| **ASR** | Streams audio from the onboard microphone through a speech recognition pipeline that produces timestamped text transcriptions. These are fed into the LLM context as user input, enabling continuous voice-driven interaction. |
| **TTS** | Takes text output from the LLM and synthesizes it into audio, which is played through the robot's speaker. The synthesis pipeline is tuned for low latency so responses feel conversational rather than delayed. |
| **LLM Integration** | Manages the prompt lifecycle: assembles context (system prompt, conversation history, sensor summaries, task state), sends it to the configured language model, and routes the structured response back to the appropriate output channel (speech, movement, logging). Multiple LLM providers and models are supported through a plugin interface. |

### OM1 Avatar (`om1-avatar`)

The OM1 Avatar is the frontend interface layer displayed on the BrainPack screen. It gives the robot a visual identity and surfaces system state to anyone nearby, making the robot's inner workings legible without requiring a separate device or dashboard.

The avatar renders in real time and reacts to what the robot is doing — speaking, listening, navigating, or idle — so observers can read the robot's current state at a glance. It is built on React and communicates with the OM1 backend over a local websocket connection.

| Feature | Description |
|---------|-------------|
| **React-based UI** | Built as a responsive web application served locally on the BrainPack, accessible on the device's display. |
| **Real-time Avatar Rendering** | Animates a visual avatar that reflects the robot's current activity and emotional state, updated continuously as the robot operates. |
| **System Status Visualization** | Displays key system metrics and service health so operators can quickly spot issues without SSH-ing into the device. |
| **User Interaction** | Accepts touch or button input directly on the BrainPack screen, enabling local control and conversation without an external controller. |

---

## Premium Components

These components are accessible through the **Enterprise Plan**.

### OM1 ROS2 SDK (`om1-ros2-sdk`)

The OM1 ROS2 SDK is the robotics middleware layer that connects OM1's high-level decisions to the robot's physical hardware. It handles everything from raw sensor ingestion to autonomous navigation, remote control, and simulation — all built on the ROS2 (Robot Operating System 2) framework, which provides standardized topic-based messaging between hardware and software components.

Without the ROS2 SDK, OM1 can speak and reason but cannot navigate, map, or perceive its spatial environment. The SDK is what turns a conversational robot into an autonomous one.

#### Features

- **Sensor drivers** - Low-level drivers for onboard hardware sensors:
  - *Intel RealSense D435* — Provides an RGB color stream plus a calibrated depth map, giving the robot the ability to perceive the 3D structure of its surroundings.
  - *RPLidar* — Emits a 360° 2D laser scan of the environment, used as the primary input for mapping and localisation.
  Both sensors publish their data as typed ROS2 topics consumed by the orchestrator and navigation pipeline.

- **Remote robot control** — Exposes a network interface for sending motion commands and operational instructions to the robot from a remote system. This enables tele-operation, remote supervision, and integration with external control software without physical access to the robot.

- **Remote audio** — Enables bidirectional audio communication with the robot over the network. Operators can listen to what the robot hears and speak to it remotely, supporting use cases like remote supervision, guided operation, and off-site interaction.

- **SLAM (Simultaneous Localization and Mapping)** — Builds a live map of the environment as the robot moves, while simultaneously estimating the robot's position within that map. The system fuses LiDAR scans and depth data to construct and update a spatial model of the environment in real time, enabling the robot to navigate areas it has never explicitly been programmed for.

- **Full simulation support** — The complete ROS2 SDK stack — sensors, SLAM, navigation, and control — can run inside a simulator (such as Gazebo) without physical hardware. This enables developers to test navigation algorithms, tune parameters, and validate new features in a reproducible environment before deploying to a physical robot.

- **Multi-Robot Support** - Compatible with Unitree Go2, Unitree G1, and LimX Tron robots.

- **Auto Charging** - When the robot's battery falls below a threshold, the system initiates an autonomous return-to-dock sequence:

1. The Nav2 stack navigates the robot to the general vicinity of the charging station using the stored map
2. The robot switches to precision docking mode and activates its onboard cameras to detect AprilTag markers mounted on or near the dock
3. Visual servoing aligns the robot incrementally by tracking the AprilTag's pose in camera space
4. The robot approaches and physically docks, aligning its charging contacts with the pad's contact points

> **Note:** Currently supported on **Go2 only**.

- **Navigation & Localisation** - Integration with Nav2 for autonomous navigation. We have a custom localisation pipeline that the robot uses to determine its position and plan paths through its environment.Process incoming LaserScan message to determine feasible paths.Publish feasible paths and visualization markers.

| Component | Description |
|-----------|-------------|
| **Visual Place Recognition (VPR)** | Uses camera images to estimate which general area of the map the robot is in |
| **Correlative Scan Matching (CSM)** | Identifies distinctive geometric landmarks in sensor data and aligns them against a reference map |
| **Nav2 AMCL** | A probabilistic particle filter that converges on the most likely position as new sensor data arrives. Robust in dynamic environments where the map may have changed. |

> Refer to [Hybrid Localisation](./localization.md) for an in-depth explanation of the system.

- **Obstacle Avoidance** - The obstacle avoidance system generates candidate paths by projecting straight-line segments from the robot's origin across a configurable range of headings and distances. Each candidate path is evaluated by fusing data from multiple sensor sources — RPLidar 360° laser scans, Intel RealSense depth images, and hazard point clouds, to determine which paths are clear of obstacles. Path segment geometry is precomputed once to eliminate redundant calculations at runtime. Feasible paths are then published as ROS2 topics alongside RViz visualization markers, giving operators real-time visibility into the robot's path selection decisions.

#### Components

The `om1-ros2-sdk` is composed of four internal services:

| Service | Role |
|---------|------|
| `om1_sensor` | Manages low-level sensor drivers and continuously publishes raw sensor data (depth, RGB, LiDAR scans) to ROS2 topics |
| `orchestrator` | Consumes sensor topics to run SLAM and navigation; manages map storage and path planning; exposes a REST API for external control and status queries |
| `watchdog` | Monitors the health of sensor topics and the `om1_sensor` process; automatically restarts `om1_sensor` if data stops arriving or quality degrades |
| `zenoh_bridge` | Acts as a protocol bridge between the OM1 core runtime and the ROS2 ecosystem, translating between Zenoh pub/sub messages and ROS2 topics in both directions |

---

### OM1 Video Processor (`om1-video-processor`)

The OM1 Video Processor is a dedicated media processing pipeline that runs entirely on the robot's edge device. It handles real-time face anonymisation, audio cleanup, and AV streaming — all without sending raw video or audio off-device. CUDA acceleration via NVIDIA TensorRT ensures each stage meets real-time latency requirements even on embedded hardware.

| Feature | Description |
|---------|-------------|
| **Face Detection** | Each incoming video frame is scanned by the SCRFD model, a lightweight face detection architecture optimised with TensorRT for real-time inference at full frame rate. |
| **Face Blurring** | Expands detected bounding boxes, applies feathered masks, and uses strong Gaussian blur to make identity unrecoverable. Raw frames never leave the device. |
| **Audio Noise Cancellation** | Filters microphone input in real time to suppress ambient noise (fan hum, footsteps, crowd noise) before ASR processing or remote streaming. |
| **Audio Streaming** | Captures processed audio and streams it to an RTSP server as a continuous audio track for external consumers. |
| **Video Streaming** | Captures processed, blurred video output and streams it to the RTSP server as a synchronized video track. |

> **What is RTSP?**
> RTSP (Real Time Streaming Protocol) is a network control protocol for managing multimedia streaming sessions. Rather than transporting media itself, it manages the session — establishing the stream, synchronising audio and video tracks, and providing controls (play, pause, seek) to the consumer. The video processor uses RTSP so that any compatible media player or monitoring system can consume the robot's live feed without custom integration work.

---

### Person Following (`person-following`)

The Person Following service enables the robot to autonomously detect, track, and follow a designated person through its environment. It combines continuous visual detection with spatial reasoning and scene understanding, allowing the robot to stay close to a person while navigating around obstacles in its path.

This service is designed for use cases such as personal assistance, guided tours, and supervised autonomy — anywhere the robot needs to stay with a person rather than navigate to a fixed destination.

| Feature | Description |
|---------|-------------|
| **Body Detection** | Detects and tracks a person's full body using body pose and silhouette (not just face). Robust to turned faces, crouching, or partial occlusion. Maintains persistent identity for the target person even as others move through the scene. |
| **Distance Detection** | Uses depth data from the Intel RealSense D435 to estimate 3D distance between the robot and tracked person. Motion controller adjusts speed and heading to maintain comfortable following gap. |
| **Video Description** | Generates real-time natural language descriptions of what the robot sees using a vision-language model. Enables verbal narration of observations and provides foundation for complex reasoning behaviours. |

---

## OM1-OTA

All services in the full autonomy stack are delivered as OTA-managed containers. The `ota_agent` and `ota_updater` services handle the full lifecycle of every container on the device, ensuring the robot stays up to date without requiring manual intervention.

### `ota_agent`

The main OTA (over-the-air) lifecycle manager. It is responsible for the complete update cycle across all containerized services on the robot:

- Connects to the container registry and pulls new images when updates are available
- Starts, stops, and restarts service containers in the correct dependency order
- Applies version upgrades to application images without requiring a full system restart
- Reports service health and update status back to the management plane

### `ota_updater`

A self-update companion for `ota_agent`. Because `ota_agent` manages all other containers, it cannot update itself — `ota_updater` exists specifically to handle that case:

- Monitors for new versions of `ota_agent` and applies updates when available
- Ensures that the update agent itself never becomes outdated or incompatible with the services it manages
- Acts as the last line of the update chain, keeping the entire update infrastructure current

---

Your robot is now ready to accompany you, assist with tasks, explore new environments, and learn alongside you. To access the premium features through API endpoints refer the documentation [here](./api_endpoints.md).
