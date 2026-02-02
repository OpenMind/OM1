---
title: New Input Plugin
description: "Build a new input plugin"
---

## Overview
This guide walks you through creating a new input plugin for OM1. Input plugins allow you to integrate various data sources and sensors into your agent.

## Prerequisites

- Understanding of Python classes and inheritance
- Familiarity with async programming
- Knowledge of the data source you're integrating

## Implementation Steps

### Step 1: Create a Provider class (Optional)
If your plugin requires complex initialization or external service integration, create a provider class.

Location: /src/providers/your_provider.py

### Step 2: Create a new Plugin File

To proceed with a new input plugin integration, create a python file.
Location: `/src/inputs/plugins/your_plugin.py`

Required imports -
```bash
from inputs.base import Sensor, SensorConfig
from inputs.base.loop import FuserInput
from providers.plugin_provider import PluginProvider # import your provider class (if defined)
from providers.io_provider import IOProvider
```

### Step 3: Implement Your Plugin Class

```bash
class YourInput(FuserInput[YourRawType]):
    def __init__(self, config: SensorConfig = SensorConfig()):
        super().__init__(config)
        # Initialize your plugin-specific resources
        self.plugin = PluginProvider()
        self.io_provider = IOProvider()
        self.messages: list[Message] = []
```

### Step 4: Implement Required Methods

Implement any methods that will be required for the setup. Here are a few examples-

```bash
    async def start(self) -> None:
        """
        Initialize and start the input plugin.
        Called when the plugin begins operation.
        """
        self._is_running = True
        # Set up connections, open files, initialize hardware, etc.
        await self._connect()

    async def stop(self) -> None:
        """
        Cleanly shutdown the input plugin.
        Called when the plugin should stop operation.
        """
        self._is_running = False
        # Close connections, release resources, cleanup
        await self._disconnect()
```

## Plugin Registration

**Automatic Discovery** - Plugins are automatically discovered by the system. No manual registration required!

### How it works:

    - The system scans /src/inputs/plugins/ directory
    - Uses find_module_with_class() in /src/inputs/__init__.py
    - Finds all classes inheriting from FuserInput
    - Uses regex pattern: class {ClassName}(FuserInput)

### Requirements for auto-discovery:

    - Class must inherit from FuserInput
    - File must be in /src/inputs/plugins/ directory
    - Use standard class declaration syntax
