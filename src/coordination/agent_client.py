"""
Agent Client for Multi-Agent Coordination

Provides interface for agents to participate in coordination.
"""

import asyncio
import logging
from typing import Optional
import uuid

from coordination.protocol import (
    AgentCapabilities,
    HeartbeatMessage,
    RegisterMessage,
    MessageType
)
from coordination.registry import AgentRegistry


class CoordinatedAgent:
    """
    Client for agents participating in multi-agent coordination.
    
    Handles registration, heartbeats, and communication with the registry.
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        capabilities: Optional[AgentCapabilities] = None,
        registry: Optional[AgentRegistry] = None,
        heartbeat_interval: float = 1.0
    ):
        """
        Initialize a coordinated agent.
        
        Parameters
        ----------
        agent_id : Optional[str]
            Unique identifier for this agent (auto-generated if None)
        capabilities : Optional[AgentCapabilities]
            Agent capabilities (uses defaults if None)
        registry : Optional[AgentRegistry]
            Registry to connect to (required for coordination)
        heartbeat_interval : float
            Interval between heartbeats in seconds (default: 1.0)
        """
        self.agent_id = agent_id or f"agent_{uuid.uuid4().hex[:8]}"
        self.capabilities = capabilities or AgentCapabilities()
        self.registry = registry
        self.heartbeat_interval = heartbeat_interval
        self.sequence = 0
        self._running = False
        self._heartbeat_task: Optional[asyncio.Task] = None
        
    async def register(self) -> bool:
        """
        Register this agent with the registry.
        
        Returns
        -------
        bool
            True if registration successful
        """
        if not self.registry:
            logging.error(f"Agent {self.agent_id}: No registry configured")
            return False
        
        message = RegisterMessage(
            agent_id=self.agent_id,
            capabilities=self.capabilities,
            sequence=self.sequence
        )
        
        success = self.registry.register_agent(message)
        if success:
            logging.info(f"Agent {self.agent_id} registered successfully")
        else:
            logging.error(f"Agent {self.agent_id} registration failed")
        
        return success
    
    async def _send_heartbeats(self):
        """Background task to send periodic heartbeats."""
        while self._running:
            message = HeartbeatMessage(
                agent_id=self.agent_id,
                capabilities=self.capabilities,
                status="active",
                sequence=self.sequence
            )
            
            self.registry.update_heartbeat(message)
            self.sequence += 1
            
            await asyncio.sleep(self.heartbeat_interval)
    
    async def start(self):
        """
        Start the agent (register and begin heartbeats).
        
        Returns
        -------
        bool
            True if started successfully
        """
        if self._running:
            logging.warning(f"Agent {self.agent_id} already running")
            return False
        
        # Register with registry
        if not await self.register():
            return False
        
        # Start heartbeat task
        self._running = True
        self._heartbeat_task = asyncio.create_task(self._send_heartbeats())
        
        logging.info(f"Agent {self.agent_id} started")
        return True
    
    async def stop(self):
        """Stop the agent (deregister and stop heartbeats)."""
        self._running = False
        
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        
        if self.registry:
            self.registry.deregister_agent(self.agent_id)
        
        logging.info(f"Agent {self.agent_id} stopped")
    
    def update_capabilities(self, capabilities: AgentCapabilities):
        """
        Update agent capabilities.
        
        Parameters
        ----------
        capabilities : AgentCapabilities
            New capabilities
        """
        self.capabilities = capabilities
        logging.info(f"Agent {self.agent_id} capabilities updated")
