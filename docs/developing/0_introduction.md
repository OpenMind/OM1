---
title: Introduction
description: "OpenMind builds open-source software that helps machines think, learn, and collaborate"
icon: brain
---
![Logo](../assets/openmind-intro-cover.png)

## What is OM1?

OM1 allows AI agents to be configured and deployed in both the digital and physical worlds. You can create *one* AI persona and run it in the cloud but also on physical robot hardware such as Quadrupeds, TurtleBot 4, and Humanoids.

With OM1, you can interact with OpenAI's `gpt-4o` (or Gemini, Claude, DeepSeek, or Ollama for local inference) and shake hands with it, mediated by physical robot hardware controlled by one or more LLMs. Agents/robots built on OM1 can ingest data from multiple sources (the web, X/Twitter, cameras, and LIDAR) and can then tweet, explore your house, and help your kids with their math homework.

Since it's open source, *you* have control and can optimize the system for your home or workplace.

This guide offers an overview of the OM1 agent runtime system, helping developers understand its core components and workflows. Inside, you'll find explanations of OM1’s CLI commands, recommended project structure, step-by-step instructions for adding new inputs and actions, and guidance on configuring your agents and robots for different environments. Additionally, the guide includes practical development tips to streamline your workflow.

Whether you're just getting started with OM1 or looking to optimize an existing project, this guide will equip you with the tools and best practices to develop, deploy, and maintain high-performance agents and robots.

## OM1 Capabilities

| **Title**                                | **Description**                                                                 |
|------------------------------------------|----------------------------------------------------------------------------------|
| Simple, modular architecture             | Human-intelligible architecture with natural language data buses.               |
| All Python                               | Independent modules that are easy to maintain, debug, and extend.              |
| Easy to add new data inputs              | Seamlessly integrate new data without major changes to the existing architecture. |
| Easy to support new hardware             | Via plugins for API endpoints and specific robot hardware.                      |
| Supports Standard Middleware             | `ROS2`, `Zenoh`, and `CycloneDDS`                                               |
| Includes a simple web-based debug display| Watch the system work (`WebSim` at [http://localhost:8000](http://localhost:8000)). |
| Preconfigured endpoints                  | Text-to-Speech, OpenAI's `gpt-4o`, DeepSeek, Gemini, Openrouter, Near AI, Ollama (local), and multiple VLMs. |
| Simulators                               | Gazebo, Issac Sim (Coming Soon)

To get started with OM1, head over to [Installation Guide](1_get-started).
