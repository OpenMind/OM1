"""
Priority Task Queue Plugin for OM1
Issue #629 - Prioritized Task Queue API for Enhanced Agent Coordination

Provides a priority-based task queue for coordinating agent tasks,
ensuring critical tasks (emergency stop, low battery) are processed first.
"""

import heapq
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Any, Optional
from uuid import uuid4


class Priority(IntEnum):
    """Task priority levels (lower number = higher priority)."""
    URGENT = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass(order=True)
class Task:
    """Represents a task in the priority queue."""
    
    priority: Priority = field(compare=True)
    timestamp: float = field(compare=True, default_factory=time.time)
    task_id: str = field(compare=False, default_factory=lambda: str(uuid4()))
    description: str = field(compare=False, default="")
    metadata: dict[str, Any] = field(compare=False, default_factory=dict)
    created_at: str = field(compare=False, default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict[str, Any]:
        """Convert task to dictionary representation."""
        return {
            "task_id": self.task_id,
            "priority": Priority(self.priority).name,
            "description": self.description,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "timestamp": self.timestamp
        }


class TaskQueuePlugin:
    """
    Priority-based task queue for OM1 agent coordination.
    
    Features:
    - Thread-safe operations
    - Priority-based task ordering (URGENT > HIGH > NORMAL > LOW)
    - FIFO within same priority level (using timestamp)
    - Task history tracking
    - Queue statistics
    """
    
    def __init__(self, max_history: int = 100):
        """
        Initialize the task queue plugin.
        
        Args:
            max_history: Maximum number of completed tasks to keep in history
        """
        self._queue: list[Task] = []
        self._lock = threading.Lock()
        self._history: list[Task] = []
        self._max_history = max_history
        self._stats = {
            "total_added": 0,
            "total_completed": 0,
            "by_priority": {p.name: 0 for p in Priority}
        }
    
    def add_task(
        self,
        description: str,
        priority: Priority | str = Priority.NORMAL,
        metadata: Optional[dict[str, Any]] = None
    ) -> str:
        """
        Add a task to the priority queue.
        
        Args:
            description: Task description
            priority: Task priority (URGENT, HIGH, NORMAL, LOW)
            metadata: Optional metadata dictionary
            
        Returns:
            str: Task ID
            
        Example:
            >>> plugin = TaskQueuePlugin()
            >>> task_id = plugin.add_task(
            ...     "Emergency stop",
            ...     Priority.URGENT,
            ...     {"reason": "obstacle_detected"}
            ... )
        """
        # Convert string priority to enum if needed
        if isinstance(priority, str):
            priority = Priority[priority.upper()]
        
        task = Task(
            priority=priority,
            description=description,
            metadata=metadata or {}
        )
        
        with self._lock:
            heapq.heappush(self._queue, task)
            self._stats["total_added"] += 1
            self._stats["by_priority"][priority.name] += 1
        
        return task.task_id
    
    def get_next_task(self) -> Optional[dict[str, Any]]:
        """
        Get the highest priority task from the queue.
        
        Returns:
            Optional[dict]: Task dictionary or None if queue is empty
            
        Example:
            >>> task = plugin.get_next_task()
            >>> if task:
            ...     print(f"Processing: {task['description']}")
        """
        with self._lock:
            if not self._queue:
                return None
            
            task = heapq.heappop(self._queue)
            self._stats["total_completed"] += 1
            
            # Add to history (keep last N tasks)
            self._history.append(task)
            if len(self._history) > self._max_history:
                self._history.pop(0)
            
            return task.to_dict()
    
    def peek_next_task(self) -> Optional[dict[str, Any]]:
        """
        Peek at the highest priority task without removing it.
        
        Returns:
            Optional[dict]: Task dictionary or None if queue is empty
        """
        with self._lock:
            if not self._queue:
                return None
            return self._queue[0].to_dict()
    
    def get_queue_size(self) -> int:
        """Get current queue size."""
        with self._lock:
            return len(self._queue)
    
    def get_stats(self) -> dict[str, Any]:
        """
        Get queue statistics.
        
        Returns:
            dict: Statistics including queue size, total tasks, priority breakdown
        """
        with self._lock:
            return {
                "queue_size": len(self._queue),
                "total_added": self._stats["total_added"],
                "total_completed": self._stats["total_completed"],
                "by_priority": self._stats["by_priority"].copy(),
                "history_size": len(self._history)
            }
    
    def get_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Get recently completed tasks.
        
        Args:
            limit: Maximum number of tasks to return
            
        Returns:
            list: List of task dictionaries
        """
        with self._lock:
            return [task.to_dict() for task in self._history[-limit:]]
    
    def clear_queue(self) -> int:
        """
        Clear all tasks from the queue.
        
        Returns:
            int: Number of tasks cleared
        """
        with self._lock:
            count = len(self._queue)
            self._queue.clear()
            return count
    
    def get_all_tasks(self) -> list[dict[str, Any]]:
        """
        Get all tasks currently in the queue (sorted by priority).
        
        Returns:
            list: List of task dictionaries
        """
        with self._lock:
            # Return sorted copy without modifying the heap
            return [task.to_dict() for task in sorted(self._queue)]


# Singleton instance for easy access
_instance: Optional[TaskQueuePlugin] = None


def get_task_queue() -> TaskQueuePlugin:
    """Get or create the singleton TaskQueuePlugin instance."""
    global _instance
    if _instance is None:
        _instance = TaskQueuePlugin()
    return _instance
