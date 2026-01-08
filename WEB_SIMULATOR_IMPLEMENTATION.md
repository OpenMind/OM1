# Browser-Based Three.js Simulator Implementation

This document describes the implementation of the browser-based three.js simulator for OM1, completed as part of the OM1 Bounty Program.

## Overview

The implementation provides a complete browser-based robot simulator that allows developers to run OM1 without installing ROS or having real hardware. The simulator uses three.js for 3D rendering and integrates seamlessly with OM1's existing action system.

## Components

### 1. Three.js Simulator (`src/simulators/plugins/ThreeJSSim.py`)

A FastAPI-based web server that:
- Serves an HTML page with embedded three.js application
- Provides WebSocket endpoint for real-time state updates
- Exposes HTTP API endpoint (`/api/command`) for receiving movement commands
- Renders a 3D playground environment with:
  - Green ground plane with grid
  - Boundary walls
  - Simple robot model (body, head, wheels)
  - Lighting and shadows
  - Camera controls (orbit controls)

**Key Features:**
- Runs on configurable port (default: 8001)
- Real-time WebSocket updates
- Responsive 3D rendering
- Info panel showing robot status

### 2. Move WebSim Action (`src/actions/move_web_sim/`)

A complete action implementation that:
- Defines movement interface (`interface.py`)
- Implements HTTP connector (`connector/websocket.py`)
- Translates OM1 commands to simulator commands
- Manages robot state and movement queue

**Supported Commands:**
- `turn left` - Rotate 90° counterclockwise
- `turn right` - Rotate 90° clockwise
- `move forwards` - Move forward 0.5m
- `move back` - Move backward 0.5m
- `stand still` - Stop movement

### 3. Configuration (`config/web_sim.json5`)

Pre-configured setup file that:
- Configures the ThreeJSSim simulator
- Sets up the move_web_sim action connector
- Includes conversation input for easy testing
- Configures TTS for speech output

### 4. Documentation

- **Setup Guide**: `docs/developing/web_simulator_setup.mdx`
- **Example Documentation**: `docs/examples/web_simulator.mdx`

## Architecture

```
OM1 Runtime
    ↓
Action Orchestrator
    ↓
MoveWebSimConnector (HTTP POST to /api/command)
    ↓
ThreeJSSim (FastAPI Server)
    ↓
WebSocket Broadcast
    ↓
Browser (Three.js Application)
```

## Usage

### Quick Start

```bash
python -m src.run web_sim
```

Then open `http://localhost:8001` in your browser.

### Custom Configuration

Add to your config file:

```json5
{
  "simulators": [
    {
      "type": "ThreeJSSim",
      "config": {
        "port": 8001
      }
    }
  ],
  "agent_actions": [
    {
      "name": "move_web_sim",
      "llm_label": "move",
      "connector": "websocket",
      "config": {
        "simulator_url": "http://localhost:8001"
      }
    }
  ]
}
```

## Features Implemented

✅ **Web Simulator**: Complete three.js-based 3D simulator  
✅ **OM1 Control**: Full integration with OM1 action system  
✅ **Environment**: Playground with floor, walls, and obstacles  
✅ **Modular Integration**: Follows OM1's plugin architecture  
✅ **Starter Templates**: Pre-configured setup file  
✅ **Documentation**: Comprehensive setup and usage guides  
✅ **Accessibility**: No ROS or hardware required  

## Technical Details

### Dependencies Added

- `requests>=2.31.0` - Added to `pyproject.toml` for HTTP communication

### Files Created

1. `src/simulators/plugins/ThreeJSSim.py` - Simulator implementation
2. `src/actions/move_web_sim/interface.py` - Action interface
3. `src/actions/move_web_sim/connector/websocket.py` - Action connector
4. `src/actions/move_web_sim/__init__.py` - Module init
5. `config/web_sim.json5` - Configuration file
6. `docs/examples/web_simulator.mdx` - Example documentation
7. `docs/developing/web_simulator_setup.mdx` - Setup guide

### Integration Points

- **Simulator System**: Extends `Simulator` base class
- **Action System**: Implements `ActionConnector` interface
- **Runtime**: Works with existing `ActionOrchestrator` and `SimulatorOrchestrator`

## Testing

To test the implementation:

1. Start OM1 with web simulator:
   ```bash
   python -m src.run web_sim
   ```

2. Open browser to `http://localhost:8001`

3. Send commands via conversation input or API

4. Observe robot movement in 3D simulator

## Limitations & Future Enhancements

### Current Limitations

- Simple box robot model (no custom 3D models yet)
- Basic physics (no collision detection)
- 2D movement only (no jumping/flying)
- Single robot per instance

### Potential Enhancements

- Import GLTF/GLB robot models
- Advanced physics with collision detection
- Multiple robot support
- Sensor visualization (LIDAR, cameras)
- Record and replay functionality
- Multi-player support

## Submission Details

**🤖 Robot(s) Simulated**: Basic wheeled robot (box robot with 4 wheels)

**🚀 PR Link**: [To be provided when PR is created]

**🎥 Demo Video**: [To be created]

**📑 Notes**:
- Setup: Use `python -m src.run web_sim` and open `http://localhost:8001`
- The simulator runs on port 8001 by default (configurable)
- No ROS or hardware required
- Works with any modern browser
- Fully integrated with OM1's action system

## Code Quality

- ✅ Follows OM1 code style and architecture patterns
- ✅ Type hints included
- ✅ Comprehensive error handling
- ✅ Logging throughout
- ✅ No linter errors
- ✅ Modular and extensible design

