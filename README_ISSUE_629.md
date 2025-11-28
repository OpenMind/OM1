# Priority Task Queue Plugin

**Issue #629** - Prioritized Task Queue API for Enhanced Agent Coordination

## Overview

This plugin provides a thread-safe, priority-based task queue for coordinating OM1 agent tasks. Critical tasks (emergency stop, low battery) are guaranteed to be processed before lower-priority tasks.

## Features

- **Priority-based ordering**: URGENT > HIGH > NORMAL > LOW
- **FIFO within same priority**: Tasks with equal priority are processed in order of arrival
- **Thread-safe operations**: Safe for concurrent access
- **Task metadata**: Attach custom data to tasks
- **History tracking**: Keep record of completed tasks
- **Statistics**: Monitor queue performance and usage

## Installation

The plugin is included in the OM1 codebase. No additional installation required.

\`\`\`bash
# Run tests
pytest tests/test_task_queue.py -v

# Run demo
python demo_task_queue.py
\`\`\`

## Usage

### Basic Example

\`\`\`python
from src.plugins.task_queue import Priority, get_task_queue

# Get queue instance
queue = get_task_queue()

# Add tasks
queue.add_task("Navigate to waypoint", Priority.NORMAL)
queue.add_task("Emergency stop", Priority.URGENT)
queue.add_task("Log status", Priority.LOW)

# Get next task (returns highest priority)
task = queue.get_next_task()
print(f"Processing: {task['description']}")
# Output: "Processing: Emergency stop"
\`\`\`

### Priority Levels

\`\`\`python
from src.plugins.task_queue import Priority

Priority.URGENT  # Critical safety tasks (emergency stop, collision avoidance)
Priority.HIGH    # Important tasks (low battery, sensor failures)
Priority.NORMAL  # Standard operations (navigation, routine tasks)
Priority.LOW     # Background tasks (logging, cleanup)
\`\`\`

### Advanced Usage

#### With Metadata

\`\`\`python
# Add task with custom metadata
task_id = queue.add_task(
    "Return to charging station",
    Priority.HIGH,
    metadata={
        "battery_level": 15,
        "station_id": "station_1",
        "estimated_time": 300
    }
)

# Retrieve task
task = queue.get_next_task()
print(task["metadata"]["battery_level"])  # 15
\`\`\`

#### Peek Without Removing

\`\`\`python
# Check next task without processing
next_task = queue.peek_next_task()
if next_task and next_task["priority"] == "URGENT":
    print("Emergency task pending!")
\`\`\`

#### Queue Statistics

\`\`\`python
stats = queue.get_stats()
print(f"Queue size: {stats['queue_size']}")
print(f"Total processed: {stats['total_completed']}")
print(f"Urgent tasks: {stats['by_priority']['URGENT']}")
\`\`\`

#### Task History

\`\`\`python
# Get last 10 completed tasks
history = queue.get_history(limit=10)
for task in history:
    print(f"{task['created_at']}: {task['description']}")
\`\`\`

## Real-World Scenarios

### Scenario 1: Emergency Stop

\`\`\`python
# Normal operations in progress
queue.add_task("Patrol sector A", Priority.NORMAL)
queue.add_task("Patrol sector B", Priority.NORMAL)

# Emergency detected
queue.add_task(
    "Emergency stop - obstacle detected",
    Priority.URGENT,
    metadata={"obstacle_distance": 0.5}
)

# Emergency task is processed first
next_task = queue.get_next_task()
assert next_task["priority"] == "URGENT"
\`\`\`

### Scenario 2: Battery Management

\`\`\`python
# Mission tasks queued
queue.add_task("Execute pickup", Priority.NORMAL)
queue.add_task("Navigate to delivery", Priority.NORMAL)

# Battery drops below threshold
if battery_level < 20:
    queue.add_task(
        "Return to charging station",
        Priority.HIGH,
        metadata={"battery_level": battery_level}
    )
\`\`\`

### Scenario 3: Sensor Failure

\`\`\`python
# High-priority warning
queue.add_task(
    "Sensor calibration required",
    Priority.HIGH,
    metadata={
        "sensor": "lidar",
        "error_code": "CALIB_001"
    }
)
\`\`\`

## API Reference

### TaskQueuePlugin

#### Methods

**add_task(description, priority, metadata=None)**
- Add a task to the queue
- Returns: task_id (str)

**get_next_task()**
- Get and remove highest priority task
- Returns: task dict or None

**peek_next_task()**
- View highest priority task without removing
- Returns: task dict or None

**get_queue_size()**
- Get current number of tasks
- Returns: int

**get_stats()**
- Get queue statistics
- Returns: dict with queue metrics

**get_history(limit=10)**
- Get recently completed tasks
- Returns: list of task dicts

**get_all_tasks()**
- Get all queued tasks (sorted by priority)
- Returns: list of task dicts

**clear_queue()**
- Remove all tasks from queue
- Returns: number of tasks cleared

## Testing

Run the test suite:

\`\`\`bash
# All tests
pytest tests/test_task_queue.py -v

# Specific test
pytest tests/test_task_queue.py::TestTaskQueuePlugin::test_priority_ordering -v

# With coverage
pytest tests/test_task_queue.py --cov=src.plugins.task_queue
\`\`\`

Test coverage includes:
- Priority ordering (URGENT > HIGH > NORMAL > LOW)
- FIFO within same priority
- Thread safety
- Metadata handling
- Statistics tracking
- History management
- Edge cases (empty queue, etc.)

## Demo

Run the interactive demo:

\`\`\`bash
python demo_task_queue.py
\`\`\`

The demo includes:
1. Basic operations
2. Emergency scenario
3. Battery management
4. Statistics tracking

## Performance

- **Add task**: O(log n)
- **Get next task**: O(log n)
- **Peek task**: O(1)
- **Thread-safe**: Uses threading.Lock
- **Memory**: O(n) where n = queue size + history size

## Integration with OM1

This plugin can be integrated into OM1 agent controllers:

\`\`\`python
from src.plugins.task_queue import get_task_queue, Priority

class AgentController:
    def __init__(self):
        self.task_queue = get_task_queue()
    
    def on_emergency(self, event):
        # Add urgent task
        self.task_queue.add_task(
            f"Emergency: {event.type}",
            Priority.URGENT,
            metadata=event.data
        )
    
    def run_loop(self):
        while True:
            task = self.task_queue.get_next_task()
            if task:
                self.execute_task(task)
            else:
                time.sleep(0.1)  # No tasks
\`\`\`

## Contributing

This implementation addresses Issue #629. For improvements or bug reports, please open an issue or submit a PR.

## License

Same as OM1 project license.
