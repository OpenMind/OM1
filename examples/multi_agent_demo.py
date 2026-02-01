#!/usr/bin/env python3
"""
Multi-Agent Coordination Demo

Demonstrates agent registration, heartbeats, and discovery.
Shows 3 simulated agents coordinating through a central registry.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coordination.agent_client import CoordinatedAgent
from coordination.registry import AgentRegistry
from coordination.protocol import AgentCapabilities


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def simulate_agent(
    agent_id: str,
    registry: AgentRegistry,
    capabilities: AgentCapabilities,
    duration: float = 10.0
):
    """
    Simulate a single agent's lifecycle.
    
    Parameters
    ----------
    agent_id : str
        Agent identifier
    registry : AgentRegistry
        Registry to connect to
    capabilities : AgentCapabilities
        Agent capabilities
    duration : float
        How long to run the agent (seconds)
    """
    agent = CoordinatedAgent(
        agent_id=agent_id,
        capabilities=capabilities,
        registry=registry,
        heartbeat_interval=1.0
    )
    
    await agent.start()
    print(f"✅ {agent_id} started")
    
    # Simulate agent working for specified duration
    await asyncio.sleep(duration)
    
    await agent.stop()
    print(f"🛑 {agent_id} stopped")


async def monitor_registry(registry: AgentRegistry, interval: float = 2.0):
    """
    Monitor and display registry statistics.
    
    Parameters
    ----------
    registry : AgentRegistry
        Registry to monitor
    interval : float
        Update interval in seconds
    """
    while True:
        stats = registry.get_stats()
        print(f"\n📊 Registry Stats:")
        print(f"   Total registered: {stats['total_registered']}")
        print(f"   Active agents: {stats['active_agents']}")
        print(f"   Agent IDs: {', '.join(stats['agent_ids'])}")
        
        await asyncio.sleep(interval)


async def main():
    """Run the multi-agent coordination demo."""
    print("🚀 Multi-Agent Coordination Demo")
    print("=" * 50)
    
    # Create central registry
    registry = AgentRegistry(heartbeat_timeout=3.0)
    await registry.start()
    print("✅ Registry started\n")
    
    # Define agent capabilities
    robot1_caps = AgentCapabilities(
        can_navigate=True,
        can_manipulate=False,
        has_camera=True,
        has_lidar=True,
        battery_level=0.95
    )
    
    robot2_caps = AgentCapabilities(
        can_navigate=True,
        can_manipulate=True,
        has_camera=True,
        has_lidar=False,
        battery_level=0.75
    )
    
    robot3_caps = AgentCapabilities(
        can_navigate=True,
        can_manipulate=False,
        has_camera=False,
        has_lidar=True,
        battery_level=0.50
    )
    
    # Create tasks for agents
    agent_tasks = [
        simulate_agent("robot_1", registry, robot1_caps, duration=8.0),
        simulate_agent("robot_2", registry, robot2_caps, duration=12.0),
        simulate_agent("robot_3", registry, robot3_caps, duration=6.0),
    ]
    
    # Monitor registry in background
    monitor_task = asyncio.create_task(monitor_registry(registry, interval=2.0))
    
    try:
        # Run all agents
        print("Starting agents...\n")
        await asyncio.gather(*agent_tasks)
        
        print("\n✅ All agents completed")
        
        # Show final stats
        await asyncio.sleep(2)
        final_stats = registry.get_stats()
        print(f"\n📊 Final Registry Stats:")
        print(f"   Total registered: {final_stats['total_registered']}")
        print(f"   Active agents: {final_stats['active_agents']}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    finally:
        monitor_task.cancel()
        await registry.stop()
        print("\n🛑 Registry stopped")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDemo terminated")
