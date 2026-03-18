---
title: Video Processor Beta Release
description: "v1.0.1-beta.3"
icon: rectangle-beta
---

## What's included

First beta release for OM1 video processor. This release introduces major foundational features that enable developers and integrators to build advanced streaming and analytics solutions with ease.

## [v1.0.1-beta.3](https://github.com/OpenMind/OM1-video-processor/releases/tag/v1.0.1-beta.3)

- Introduced a new variable called ENABLE_CLOUD_STREAMING, which can be used to disable the online streaming feature.

## [v1.0.1-beta.2](https://github.com/OpenMind/OM1-video-processor/releases/tag/v1.0.1-beta.2)

- The raw video stream is published to the local media MTX server at the URL /top_camera_raw for local usage.

## [v1.0.1-beta.1](https://github.com/OpenMind/OM1-video-processor/releases/tag/v1.0.1-beta.1)

- Switched the base Docker image from JetPack to CUDA 13.0.0 with Ubuntu 24.04
- Updated the Python version from 3.10 to 3.12
- CUDA driver mismatch issue fixed
- Fixed TensorRT version

## [v1.0.0-beta.1](https://github.com/OpenMind/OM1-video-processor/releases/tag/v1.0.0-beta.1)

- Face Detection and Anonymization: Added advanced face detection capabilities with real-time anonymization. Faces can now be automatically blurred or masked to protect privacy in live or recorded streams. This process takes place on the edge device of the robot.
- RTSP for Audio and Video Streaming: Introduced full RTSP (Real-Time Streaming Protocol) support, enabling seamless transmission of both audio and video data. This allows integration with a wider range of cameras, streaming servers, and third-party applications. RTSP manages streaming sessions but does not typically transport the media data itself
- Support Multiple Video streams: Enhanced the system to support multiple concurrent video streams. Users can now view, process, and manage several input sources simultaneously without performance degradation.
- Support the Local and Remote Video Stream: Added the ability to handle both local camera feeds and remote video sources. This provides greater flexibility for hybrid setups that combine on-premise and cloud-based video inputs.
- Reduced Microphone Latency: Optimized the audio pipeline to significantly reduce microphone input latency. This ensures more natural and synchronized communication in real-time applications.
- Dynamic FPS Support: Implemented dynamic frame rate adjustment to optimize performance and bandwidth usage. The system now automatically adapts FPS based on network conditions and processing load.
- Noise Cancellation and Echo Reduction: Integrated advanced audio processing algorithms for noise suppression and echo reduction. This results in clearer, higher-quality sound for both streaming and recording scenarios.
