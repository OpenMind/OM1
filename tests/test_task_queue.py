"""
Test suite for Priority Task Queue Plugin
Issue #629 - Prioritized Task Queue API
"""

import time
import pytest
from src.plugins.task_queue import Priority, TaskQueuePlugin


class TestTaskQueuePlugin:
    """Test suite for TaskQueuePlugin."""
    
    def setup_method(self):
        """Create a fresh plugin instance for each test."""
        self.plugin = TaskQueuePlugin()
    
    def test_add_task_basic(self):
        """Test adding a basic task."""
        task_id = self.plugin.add_task("Test task", Priority.NORMAL)
        
        assert task_id is not None
        assert isinstance(task_id, str)
        assert self.plugin.get_queue_size() == 1
    
    def test_priority_ordering(self):
        """Test that tasks are returned in priority order."""
        # Add tasks in mixed priority order
        self.plugin.add_task("Low priority", Priority.LOW)
        self.plugin.add_task("Urgent task", Priority.URGENT)
        self.plugin.add_task("Normal task", Priority.NORMAL)
        self.plugin.add_task("High priority", Priority.HIGH)
        
        # Verify they come out in priority order
        task1 = self.plugin.get_next_task()
        assert task1["priority"] == "URGENT"
        
        task2 = self.plugin.get_next_task()
        assert task2["priority"] == "HIGH"
        
        task3 = self.plugin.get_next_task()
        assert task3["priority"] == "NORMAL"
        
        task4 = self.plugin.get_next_task()
        assert task4["priority"] == "LOW"
    
    def test_fifo_within_same_priority(self):
        """Test FIFO ordering within same priority level."""
        # Add multiple tasks with same priority
        id1 = self.plugin.add_task("Task 1", Priority.NORMAL)
        time.sleep(0.01)  # Ensure different timestamps
        id2 = self.plugin.add_task("Task 2", Priority.NORMAL)
        time.sleep(0.01)
        id3 = self.plugin.add_task("Task 3", Priority.NORMAL)
        
        # Verify FIFO order
        task1 = self.plugin.get_next_task()
        assert task1["task_id"] == id1
        
        task2 = self.plugin.get_next_task()
        assert task2["task_id"] == id2
        
        task3 = self.plugin.get_next_task()
        assert task3["task_id"] == id3
    
    def test_empty_queue(self):
        """Test behavior with empty queue."""
        task = self.plugin.get_next_task()
        assert task is None
        
        peek = self.plugin.peek_next_task()
        assert peek is None
    
    def test_peek_task(self):
        """Test peeking at next task without removing it."""
        self.plugin.add_task("Test task", Priority.HIGH)
        
        # Peek should return task without removing
        peek1 = self.plugin.peek_next_task()
        assert peek1 is not None
        assert self.plugin.get_queue_size() == 1
        
        # Second peek should return same task
        peek2 = self.plugin.peek_next_task()
        assert peek1["task_id"] == peek2["task_id"]
        
        # get_next_task should now remove it
        task = self.plugin.get_next_task()
        assert task["task_id"] == peek1["task_id"]
        assert self.plugin.get_queue_size() == 0
    
    def test_task_metadata(self):
        """Test adding and retrieving task metadata."""
        metadata = {
            "reason": "low_battery",
            "battery_level": 15,
            "location": {"x": 10, "y": 20}
        }
        
        task_id = self.plugin.add_task(
            "Return to charging station",
            Priority.HIGH,
            metadata=metadata
        )
        
        task = self.plugin.get_next_task()
        assert task["metadata"] == metadata
    
    def test_statistics(self):
        """Test queue statistics tracking."""
        # Add tasks
        self.plugin.add_task("Task 1", Priority.URGENT)
        self.plugin.add_task("Task 2", Priority.HIGH)
        self.plugin.add_task("Task 3", Priority.HIGH)
        self.plugin.add_task("Task 4", Priority.NORMAL)
        
        stats = self.plugin.get_stats()
        
        assert stats["queue_size"] == 4
        assert stats["total_added"] == 4
        assert stats["total_completed"] == 0
        assert stats["by_priority"]["URGENT"] == 1
        assert stats["by_priority"]["HIGH"] == 2
        assert stats["by_priority"]["NORMAL"] == 1
        
        # Complete some tasks
        self.plugin.get_next_task()
        self.plugin.get_next_task()
        
        stats = self.plugin.get_stats()
        assert stats["queue_size"] == 2
        assert stats["total_completed"] == 2
    
    def test_history(self):
        """Test task history tracking."""
        # Add and complete tasks
        self.plugin.add_task("Task 1", Priority.NORMAL)
        self.plugin.add_task("Task 2", Priority.NORMAL)
        
        self.plugin.get_next_task()
        self.plugin.get_next_task()
        
        history = self.plugin.get_history()
        assert len(history) == 2
        assert history[0]["description"] == "Task 1"
        assert history[1]["description"] == "Task 2"
    
    def test_clear_queue(self):
        """Test clearing the queue."""
        self.plugin.add_task("Task 1", Priority.NORMAL)
        self.plugin.add_task("Task 2", Priority.HIGH)
        self.plugin.add_task("Task 3", Priority.LOW)
        
        assert self.plugin.get_queue_size() == 3
        
        cleared = self.plugin.clear_queue()
        assert cleared == 3
        assert self.plugin.get_queue_size() == 0
    
    def test_string_priority(self):
        """Test using string priority values."""
        task_id = self.plugin.add_task("Test", "URGENT")
        task = self.plugin.get_next_task()
        
        assert task["priority"] == "URGENT"
    
    def test_get_all_tasks(self):
        """Test retrieving all tasks in the queue."""
        self.plugin.add_task("Low", Priority.LOW)
        self.plugin.add_task("Urgent", Priority.URGENT)
        self.plugin.add_task("Normal", Priority.NORMAL)
        
        all_tasks = self.plugin.get_all_tasks()
        
        assert len(all_tasks) == 3
        # Verify sorted by priority
        assert all_tasks[0]["priority"] == "URGENT"
        assert all_tasks[1]["priority"] == "NORMAL"
        assert all_tasks[2]["priority"] == "LOW"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
