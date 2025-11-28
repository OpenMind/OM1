#!/usr/bin/env python3
"""
Demo script for Priority Task Queue Plugin
Issue #629 - Prioritized Task Queue API

Demonstrates real-world usage scenarios for agent coordination.
"""

import time
from src.plugins.task_queue import Priority, get_task_queue


def demo_basic_usage():
    """Demonstrate basic task queue operations."""
    print("=" * 60)
    print("DEMO 1: Basic Task Queue Operations")
    print("=" * 60)
    
    queue = get_task_queue()
    queue.clear_queue()  # Start fresh
    
    # Add tasks in mixed priority
    print("\n1. Adding tasks with different priorities:")
    
    tasks = [
        ("Navigate to waypoint A", Priority.NORMAL),
        ("Low battery - return to charging", Priority.HIGH),
        ("Background data sync", Priority.LOW),
        ("Emergency stop - obstacle detected", Priority.URGENT),
        ("Update system status", Priority.NORMAL),
    ]
    
    for desc, priority in tasks:
        task_id = queue.add_task(desc, priority)
        print(f"   Added [{priority.name:6}] {desc} (ID: {task_id[:8]}...)")
    
    print(f"\n2. Queue size: {queue.get_queue_size()} tasks")
    
    # Process tasks
    print("\n3. Processing tasks in priority order:")
    while queue.get_queue_size() > 0:
        task = queue.get_next_task()
        print(f"   Processing [{task['priority']:6}] {task['description']}")
        time.sleep(0.3)  # Simulate processing
    
    print("\n✓ All tasks completed")


def demo_emergency_scenario():
    """Demonstrate emergency task handling."""
    print("\n" + "=" * 60)
    print("DEMO 2: Emergency Scenario - Critical Task Interruption")
    print("=" * 60)
    
    queue = get_task_queue()
    queue.clear_queue()
    
    # Simulate normal operations
    print("\n1. Agent performing normal operations:")
    queue.add_task("Navigate to pickup point", Priority.NORMAL)
    queue.add_task("Execute pickup action", Priority.NORMAL)
    queue.add_task("Navigate to delivery point", Priority.NORMAL)
    
    print(f"   Queued: {queue.get_queue_size()} normal tasks")
    
    # Emergency occurs
    print("\n2. EMERGENCY: Obstacle detected!")
    queue.add_task(
        "Emergency stop - obstacle in path",
        Priority.URGENT,
        metadata={"obstacle_distance": 0.5, "type": "moving_object"}
    )
    
    print("\n3. Next task to execute:")
    next_task = queue.peek_next_task()
    print(f"   Priority: {next_task['priority']}")
    print(f"   Task: {next_task['description']}")
    print(f"   Metadata: {next_task['metadata']}")
    
    print("\n✓ Emergency task will be processed first")


def demo_battery_management():
    """Demonstrate battery management scenario."""
    print("\n" + "=" * 60)
    print("DEMO 3: Battery Management")
    print("=" * 60)
    
    queue = get_task_queue()
    queue.clear_queue()
    
    # Normal mission tasks
    print("\n1. Mission in progress:")
    queue.add_task("Patrol sector A", Priority.NORMAL)
    queue.add_task("Patrol sector B", Priority.NORMAL)
    queue.add_task("Return to base", Priority.LOW)
    
    # Battery drops below threshold
    print("\n2. Battery level critical (15%)!")
    queue.add_task(
        "Return to charging station immediately",
        Priority.HIGH,
        metadata={
            "battery_level": 15,
            "estimated_time_remaining": 10,
            "charging_station": "station_1"
        }
    )
    
    print("\n3. Current queue priority:")
    all_tasks = queue.get_all_tasks()
    for i, task in enumerate(all_tasks, 1):
        print(f"   {i}. [{task['priority']:6}] {task['description']}")
    
    print("\n✓ High-priority charging task inserted before normal patrol")


def demo_statistics():
    """Demonstrate statistics and history tracking."""
    print("\n" + "=" * 60)
    print("DEMO 4: Statistics and History")
    print("=" * 60)
    
    queue = get_task_queue()
    queue.clear_queue()
    
    # Add and process various tasks
    print("\n1. Processing multiple tasks...")
    
    for i in range(10):
        priority = [Priority.URGENT, Priority.HIGH, Priority.NORMAL, Priority.LOW][i % 4]
        queue.add_task(f"Task {i+1}", priority)
    
    # Process half of them
    for _ in range(5):
        queue.get_next_task()
    
    # Show statistics
    stats = queue.get_stats()
    
    print("\n2. Queue Statistics:")
    print(f"   Current queue size: {stats['queue_size']}")
    print(f"   Total tasks added: {stats['total_added']}")
    print(f"   Total tasks completed: {stats['total_completed']}")
    print("\n   Tasks by priority:")
    for priority, count in stats['by_priority'].items():
        print(f"   - {priority:6}: {count}")
    
    # Show history
    print("\n3. Recent task history:")
    history = queue.get_history(limit=5)
    for task in history:
        print(f"   [{task['priority']:6}] {task['description']}")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("Priority Task Queue Plugin - Demo")
    print("Issue #629 - Enhanced Agent Coordination")
    print("=" * 60)
    
    demo_basic_usage()
    time.sleep(1)
    
    demo_emergency_scenario()
    time.sleep(1)
    
    demo_battery_management()
    time.sleep(1)
    
    demo_statistics()
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
