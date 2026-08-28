---
title: Examples
description: "Examples Overview"
icon: link-simple
---

## Examples

This section contains practical examples demonstrating how to use the OM1 project.

### Getting Started

Before running any examples, ensure you have the project dependencies installed. Refer the documentation [here](../developing/1_get-started.md).

### Running Examples

Examples can be executed using:

```bash
make run CONFIG=<example_name>
```

You can list every available config with `make list-configs`. Some examples to get started with:

- [Conversation](https://github.com/OpenMind/OM1/blob/main/config/conversation.json5) — `make run CONFIG=conversation`
- [Greeting Conversation](https://github.com/OpenMind/OM1/blob/main/config/greeting_conversation.json5) — `make run CONFIG=greeting_conversation`
- [Unitree G1 Humanoid](https://github.com/OpenMind/OM1/blob/main/config/unitree_g1_conversation.json5) — `make run CONFIG=unitree_g1_conversation`
- [Unitree Go2 Autonomy](https://github.com/OpenMind/OM1/blob/main/config/unitree_go2_autonomy.json5) — `make run CONFIG=unitree_go2_autonomy`
- [Unitree Go2 Modes](https://github.com/OpenMind/OM1/blob/main/config/unitree_go2_modes.json5) — `make run CONFIG=unitree_go2_modes`
- [MCP Integration](https://github.com/OpenMind/OM1/blob/main/config/conversation_mcp.json5) — `make run CONFIG=conversation_mcp`
