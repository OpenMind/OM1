# Multi-Agent Coordination Framework

Foundation for enabling multiple OM1 agents to discover, communicate, and coordinate autonomously.

## Quick Start

```bash
# Run the demo
python3 examples/multi_agent_demo.py
```

## Features

- **Agent Registry** - Central coordinator tracking all active agents
- **Heartbeat Monitoring** - Automatic detection of failed agents
- **Discovery** - Query active agents and their capabilities  
- **Message Protocol** - Structured communication between agents
- **Zero Dependencies** - Uses only Python standard library

## Architecture

```
Agent 1, 2, 3...  →  Heartbeats  →  Central Registry
                                         ↓
                                    Discovery
                                    Monitoring
                                    Coordination
```

## Components

### `protocol.py`
Message types and data structures:
- `AgentCapabilities` - What an agent can do
- `HeartbeatMessage` - Periodic alive signal
- `RegisterMessage` - Initial registration
- `TaskMessage` classes - Task coordination (Phase 2)

### `registry.py`
Central coordinator:
- Track all active agents
- Monitor heartbeats
- Provide discovery interface
- Auto-remove dead agents

### `agent_client.py`  
Agent participation interface:
- Easy registration
- Automatic heartbeats
- Capability announcements
- Graceful shutdown

## Example

```python
from coordination.agent_client import CoordinatedAgent
from coordination.registry import AgentRegistry
from coordination.protocol import AgentCapabilities

# Create registry
registry = AgentRegistry(heartbeat_timeout=3.0)
await registry.start()

# Create agent
agent = CoordinatedAgent(
    agent_id="robot_1",
    capabilities=AgentCapabilities(
        can_navigate=True,
        has_camera=True,
        battery_level=0.95
    ),
    registry=registry
)

await agent.start()  # Register and begin heartbeats
# ... agent coordinates with others ...
await agent.stop()   # Clean shutdown
```

## Documentation

See `docs/developer_cookbook/multi_agent_coordination.md` for full documentation.

## Roadmap

- ✅ **Phase 1** (Current): Agent registry, heartbeats, discovery
- 📋 **Phase 2**: Task allocation and coordination
- 📋 **Phase 3**: Shared world model, formation control
- 📋 **Phase 4**: Zenoh integration, distributed registry

## License

MIT License - See LICENSE file for details.
