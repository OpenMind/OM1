"""
Agent Registry and Discovery

Manages agent registration, heartbeat monitoring, and discovery.
"""

import asyncio
import logging
import time
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field

from coordination.protocol import (
    HeartbeatMessage,
    RegisterMessage,
    AgentCapabilities,
    MessageType
)


@dataclass
class AgentInfo:
    """Information about a registered agent."""
    agent_id: str
    capabilities: AgentCapabilities
    last_heartbeat: float
    status: str = "active"
    location: Optional[Dict[str, float]] = None
    sequence: int = 0
    registered_at: float = field(default_factory=time.time)


class AgentRegistry:
    """
    Centralized registry for managing multiple agents.
    
    Handles agent registration, heartbeat monitoring, and discovery.
    Automatically removes agents that miss heartbeats.
    """
    
    def __init__(self, heartbeat_timeout: float = 5.0):
        """
        Initialize the agent registry.
        
        Parameters
        ----------
        heartbeat_timeout : float
            Seconds to wait before considering an agent dead (default: 5.0)
        """
        self.agents: Dict[str, AgentInfo] = {}
        self.heartbeat_timeout = heartbeat_timeout
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        
    def register_agent(self, message: RegisterMessage) -> bool:
        """
        Register a new agent.
        
        Parameters
        ----------
        message : RegisterMessage
            Registration message from the agent
            
        Returns
        -------
        bool
            True if registration successful
        """
        agent_id = message.agent_id
        
        if agent_id in self.agents:
            logging.warning(f"Agent {agent_id} already registered, updating...")
        
        self.agents[agent_id] = AgentInfo(
            agent_id=agent_id,
            capabilities=message.capabilities,
            last_heartbeat=message.timestamp,
            location=message.location,
            sequence=message.sequence
        )
        
        logging.info(f"Agent {agent_id} registered with capabilities: {message.capabilities}")
        return True
    
    def update_heartbeat(self, message: HeartbeatMessage) -> bool:
        """
        Update agent heartbeat.
        
        Parameters
        ----------
        message : HeartbeatMessage
            Heartbeat message from the agent
            
        Returns
        -------
        bool
            True if update successful
        """
        agent_id = message.agent_id
        
        if agent_id not in self.agents:
            logging.warning(f"Heartbeat from unregistered agent {agent_id}, ignoring")
            return False
        
        agent = self.agents[agent_id]
        agent.last_heartbeat = message.timestamp
        agent.status = message.status
        agent.sequence = message.sequence
        
        if message.capabilities:
            agent.capabilities = message.capabilities
        
        logging.debug(f"Heartbeat from {agent_id}, status: {message.status}")
        return True
    
    def deregister_agent(self, agent_id: str) -> bool:
        """
        Deregister an agent.
        
        Parameters
        ----------
        agent_id : str
            ID of the agent to remove
            
        Returns
        -------
        bool
            True if agent was found and removed
        """
        if agent_id in self.agents:
            del self.agents[agent_id]
            logging.info(f"Agent {agent_id} deregistered")
            return True
        return False
    
    def get_active_agents(self) -> List[AgentInfo]:
        """
        Get list of active agents.
        
        Returns
        -------
        List[AgentInfo]
            List of currently active agents
        """
        current_time = time.time()
        active = []
        
        for agent in self.agents.values():
            time_since_heartbeat = current_time - agent.last_heartbeat
            if time_since_heartbeat < self.heartbeat_timeout:
                active.append(agent)
        
        return active
    
    def get_agent(self, agent_id: str) -> Optional[AgentInfo]:
        """
        Get information about a specific agent.
        
        Parameters
        ----------
        agent_id : str
            ID of the agent
            
        Returns
        -------
        Optional[AgentInfo]
            Agent information if found, None otherwise
        """
        return self.agents.get(agent_id)
    
    async def _monitor_heartbeats(self):
        """Background task to monitor agent heartbeats and remove dead agents."""
        while self._running:
            current_time = time.time()
            dead_agents = []
            
            for agent_id, agent in self.agents.items():
                time_since_heartbeat = current_time - agent.last_heartbeat
                if time_since_heartbeat > self.heartbeat_timeout:
                    dead_agents.append(agent_id)
                    logging.warning(
                        f"Agent {agent_id} missed heartbeat "
                        f"({time_since_heartbeat:.1f}s), removing"
                    )
            
            for agent_id in dead_agents:
                self.deregister_agent(agent_id)
            
            await asyncio.sleep(1.0)  # Check every second
    
    async def start(self):
        """Start the registry and heartbeat monitoring."""
        if self._running:
            logging.warning("Registry already running")
            return
        
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_heartbeats())
        logging.info("Agent registry started")
    
    async def stop(self):
        """Stop the registry and heartbeat monitoring."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logging.info("Agent registry stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics.
        
        Returns
        -------
        Dict[str, Any]
            Statistics about registered agents
        """
        active_agents = self.get_active_agents()
        return {
            "total_registered": len(self.agents),
            "active_agents": len(active_agents),
            "agent_ids": [a.agent_id for a in active_agents],
            "heartbeat_timeout": self.heartbeat_timeout
        }
