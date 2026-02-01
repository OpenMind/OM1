"""
Multi-Agent Communication Protocol

Defines message types and schemas for agent coordination.
"""

from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
import time


class MessageType(str, Enum):
    """Message types for agent coordination."""
    HEARTBEAT = "heartbeat"
    REGISTER = "register"
    DEREGISTER = "deregister"
    TASK_REQUEST = "task_request"
    TASK_ASSIGN = "task_assign"
    TASK_UPDATE = "task_update"
    OBSTACLE_ALERT = "obstacle_alert"


@dataclass
class AgentCapabilities:
    """Agent capabilities description."""
    can_navigate: bool = True
    can_manipulate: bool = False
    has_camera: bool = False
    has_lidar: bool = False
    battery_level: Optional[float] = None
    custom: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HeartbeatMessage:
    """Heartbeat message to indicate agent is alive."""
    agent_id: str
    message_type: MessageType = MessageType.HEARTBEAT
    capabilities: Optional[AgentCapabilities] = None
    status: str = "active"  # active, busy, degraded, error
    timestamp: float = field(default_factory=time.time)
    sequence: int = 0


@dataclass
class RegisterMessage:
    """Registration message for new agents."""
    agent_id: str
    capabilities: AgentCapabilities
    message_type: MessageType = MessageType.REGISTER
    location: Optional[Dict[str, float]] = None  # x, y, z coordinates
    timestamp: float = field(default_factory=time.time)
    sequence: int = 0


@dataclass
class DeregisterMessage:
    """Deregistration message."""
    agent_id: str
    message_type: MessageType = MessageType.DEREGISTER
    timestamp: float = field(default_factory=time.time)
    sequence: int = 0


@dataclass
class TaskRequestMessage:
    """Request to bid for a task."""
    agent_id: str
    task_id: str
    task_type: str
    message_type: MessageType = MessageType.TASK_REQUEST
    priority: int = 1
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    sequence: int = 0


@dataclass
class TaskUpdateMessage:
    """Update on task progress."""
    agent_id: str
    task_id: str
    status: str  # started, in_progress, completed, failed
    message_type: MessageType = MessageType.TASK_UPDATE
    progress: float = 0.0  # 0.0 to 1.0
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    sequence: int = 0
