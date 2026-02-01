# Multi-Agent Coordination Framework

## Overview

The Multi-Agent Coordination Framework enables multiple OM1 agents to discover, communicate, and coordinate with each other autonomously. This foundation provides agent registration, heartbeat monitoring, and discovery capabilities.

## Architecture

```
┌─────────────┐       ┌─────────────┐       ┌─────────────┐
│   Agent 1   │       │   Agent 2   │       │   Agent 3   │
│             │       │             │       │             │
│ Heartbeats  │       │ Heartbeats  │       │ Heartbeats  │
└──────┬──────┘       └──────┬──────┘       └──────┬──────┘
       │                     │                     │
       └─────────────────────┼─────────────────────┘
                             │
                    ┌────────▼────────┐
                    │ Agent Registry  │
                    │                 │
                    │ - Registration  │
                    │ - Heartbeat Mon │
                    │ - Discovery     │
                    └─────────────────┘
```

## Components

### 1. Protocol (`coordination/protocol.py`)

Defines message types and data structures:

- **MessageType**: Enum of message types (heartbeat, register, task messages)
- **AgentCapabilities**: Description of what an agent can do
- **AgentMessage**: Base class for all coordination messages
- **HeartbeatMessage**: Periodic "alive" signal from agents
- **RegisterMessage**: Initial registration message
- **TaskMessage classes**: For task coordination (coming in Phase 2)

### 2. Registry (`coordination/registry.py`)

Central coordinator managing all agents:

- **Agent Registration**: Track all active agents
- **Heartbeat Monitoring**: Detect failed agents automatically
- **Discovery**: Query active agents and their capabilities
- **Statistics**: Get real-time system stats

### 3. Agent Client (`coordination/agent_client.py`)

Interface for agents to participate:

- **Registration**: Join the coordination network
- **Automatic Heartbeats**: Stay alive in the registry
- **Capability Updates**: Announce capability changes
- **Graceful Shutdown**: Proper deregistration

## Quick Start

### Running the Demo

```bash
python3 examples/multi_agent_demo.py
```

This demonstrates:
- 3 simulated agents with different capabilities
- Automatic registration and heartbeats
- Real-time registry monitoring
- Graceful agent shutdown

### Creating a Coordinated Agent

```python
from coordination.agent_client import CoordinatedAgent
from coordination.registry import AgentRegistry
from coordination.protocol import AgentCapabilities

# Create registry
registry = AgentRegistry(heartbeat_timeout=3.0)
await registry.start()

# Define capabilities
capabilities = AgentCapabilities(
    can_navigate=True,
    can_manipulate=True,
    has_camera=True,
    battery_level=0.85
)

# Create and start agent
agent = CoordinatedAgent(
    agent_id="my_robot",
    capabilities=capabilities,
    registry=registry,
    heartbeat_interval=1.0
)

await agent.start()  # Registers and begins heartbeats

# Agent is now coordinating...

await agent.stop()  # Clean shutdown
await registry.stop()
```

## Configuration

### Registry Settings

```python
registry = AgentRegistry(
    heartbeat_timeout=5.0  # Seconds before agent considered dead
)
```

### Agent Settings

```python
agent = CoordinatedAgent(
    agent_id="robot_1",           # Unique identifier
    capabilities=AgentCapabilities(...),
    registry=registry,
    heartbeat_interval=1.0        # Heartbeat frequency (seconds)
)
```

### Agent Capabilities

```python
capabilities = AgentCapabilities(
    can_navigate=True,       # Can move autonomously
    can_manipulate=False,    # Has arm/gripper
    has_camera=True,         # Has vision sensor
    has_lidar=True,          # Has distance sensor
    battery_level=0.95,      # Battery percentage (0.0-1.0)
    custom={                 # Custom capabilities
        "max_speed": 2.0,
        "payload_kg": 10.0
    }
)
```

## API Reference

### AgentRegistry

#### Methods

**`register_agent(message: RegisterMessage) -> bool`**
- Register a new agent
- Returns True if successful

**`update_heartbeat(message: HeartbeatMessage) -> bool`**
- Update agent heartbeat
- Automatically called by CoordinatedAgent

**`deregister_agent(agent_id: str) -> bool`**
- Remove an agent from registry
- Returns True if agent was found

**`get_active_agents() -> List[AgentInfo]`**
- Get all currently active agents
- Filters out agents that missed heartbeats

**`get_agent(agent_id: str) -> Optional[AgentInfo]`**
- Get information about specific agent
- Returns None if agent not found

**`get_stats() -> Dict[str, Any]`**
- Get registry statistics
- Includes total registered, active count, agent IDs

**`async start()`**
- Start the registry and monitoring

**`async stop()`**
- Stop the registry

### CoordinatedAgent

#### Methods

**`async register() -> bool`**
- Register with the registry
- Automatically called by start()

**`async start() -> bool`**
- Register and begin heartbeats
- Returns True if successful

**`async stop()`**
- Stop heartbeats and deregister

**`update_capabilities(capabilities: AgentCapabilities)`**
- Update agent capabilities
- Next heartbeat will include new capabilities

## Examples

### Multi-Robot Warehouse

```python
# Create robots with different specializations
picker_caps = AgentCapabilities(
    can_navigate=True,
    can_manipulate=True,
    has_camera=True,
    custom={"role": "picker"}
)

transporter_caps = AgentCapabilities(
    can_navigate=True,
    can_manipulate=False,
    custom={"role": "transporter", "payload_kg": 50}
)

# Both register with central registry
picker = CoordinatedAgent("picker_1", picker_caps, registry)
transporter = CoordinatedAgent("transport_1", transporter_caps, registry)

await picker.start()
await transporter.start()

# Registry can now coordinate task allocation based on capabilities
active = registry.get_active_agents()
pickers = [a for a in active if a.capabilities.can_manipulate]
transporters = [a for a in active if "transporter" in a.capabilities.custom.get("role", "")]
```

### Fault Tolerance

```python
# Registry automatically detects failed agents
registry = AgentRegistry(heartbeat_timeout=3.0)

# If agent stops sending heartbeats for >3 seconds,
# it's automatically removed from active agents list
```

## Roadmap

### Phase 1: Foundation (Current)
- ✅ Agent registration and discovery
- ✅ Heartbeat monitoring
- ✅ Message protocol
- ✅ Demo with multiple agents

### Phase 2: Task Coordination (Planned)
- Task broker for work allocation
- Auction-based task assignment
- Task progress tracking
- Priority and deadline handling

### Phase 3: Advanced Features (Planned)
- Shared world model (map fusion)
- Formation control (line, wedge, circle patterns)
- Consensus decision-making
- Battery/resource management

### Phase 4: Production Ready (Planned)
- Zenoh pub/sub integration
- Distributed registry (fault-tolerant)
- Performance optimization
- Comprehensive testing

## Contributing

See `docs/developing/contributing.md` for contribution guidelines.

## License

MIT License - See LICENSE file for details.
