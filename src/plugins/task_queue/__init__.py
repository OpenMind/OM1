"""
Priority Task Queue Plugin for OM1
Issue #629 - Prioritized Task Queue API
"""

from .task_queue_plugin import (
    Priority,
    Task,
    TaskQueuePlugin,
    get_task_queue,
)

__all__ = [
    "Priority",
    "Task",
    "TaskQueuePlugin",
    "get_task_queue",
]

__version__ = "1.0.0"
