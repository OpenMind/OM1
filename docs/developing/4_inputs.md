---
title: Inputs
description: "Input Plugin Overview"
icon: pen
---

"Input Plugins" provide the sensory capabilities that allow robots to perceive their environment. These plugins capture, process, and format various types of input data, making them available to the robot's core runtime for decision-making.

## Basic Architecture

- `Sensor<T>` is the base abstract class that defines the core interface
- `FuserInput<T>` extends `Sensor<T>` and implements the polling mechanism
- `InputOrchestrator` manages multiple input sources
- `YourCustomCode` extends `FuserInput<T>` and implements the specific input functionality

## Examples

[Input plugin code examples](https://github.com/OpenMind/OM1/blob/main/src/inputs/plugins)

Here are a few examples for you to reuse and build on:

- [Google ASR](https://github.com/openmind/OM1/blob/main/src/inputs/plugins/google_asr.py)
- [Riva ASR](https://github.com/openmind/OM1/blob/main/src/inputs/plugins/riva_asr.py)
- [RPLidar](https://github.com/openmind/OM1/blob/main/src/inputs/plugins/rplidar.py)
- [VLM_COCO_Local](https://github.com/openmind/OM1/blob/main/src/inputs/plugins/vlm_coco_local.py)
- [VLM_Vila](https://github.com/openmind/OM1/blob/main/src/inputs/plugins/vlm_vila.py)
- [Arduino GPS](https://github.com/openmind/OM1/blob/main/src/inputs/plugins/gps.py)

Learn how to build a new input plugin [here](../developer_cookbook/input.md)
