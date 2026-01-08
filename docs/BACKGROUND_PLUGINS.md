# Background Plugins Guide

Background plugins in OM1 provide long-running services that operate alongside your robot's main behaviors. This guide explains available background plugins and how to use them.

## Available Background Plugins

### TeleopsConnection

Keeps your device showing as ONLINE in the OpenMind Teleops Portal.

**Use case**: Essential for simulation-only setups to appear online in the portal.

**Configuration**:
```json5
{
  "backgrounds": [
    {
      "type": "TeleopsConnection",
      "config": {
        "api_key": "om1_live_YOUR_URID_HERE"
      }
    }
  ]
}
```

**Features**:
- HTTP-based status reporting (no WebSocket required)
- Sends status updates every 5 seconds
- Minimal API overhead
- Works without real robot hardware

---

### HealthCheck

Monitors system resource usage and performance metrics.

**Use case**: Production monitoring and performance debugging.

**Configuration**:
```json5
{
  "backgrounds": [
    {
      "type": "HealthCheck",
      "config": {
        "check_interval": 60,
        "enable_logging": true,
        "alert_threshold_cpu": 90.0,
        "alert_threshold_memory": 90.0
      }
    }
  ]
}
```

**Features**:
- CPU, memory, disk I/O, network monitoring
- Automatic alerts for high resource usage
- Uptime tracking
- Programmatic API via `get_health_metrics()`

**Log Output**:
```
📊 HealthCheck [2h 15m] | CPU: 15.3% (8 cores) | Memory: 2.1% (450MB/16.0GB) | Threads: 12 | Disk R/W: 5.2/1.8 MB/s | Net TX/RX: 0.3/2.1 MB/s
```

---

### AutoRestart

Automatically restarts the system after crashes or extended unresponsiveness.

**Use case**: Production systems that need high availability.

**Configuration**:
```json5
{
  "backgrounds": [
    {
      "type": "AutoRestart",
      "config": {
        "check_interval": 30,
        "crash_threshold": 300,
        "max_restarts": 3,
        "restart_window": 3600
      }
    }
  ]
}
```

**Parameters**:
- `check_interval`: How often to check system health (seconds)
- `crash_threshold`: Time without heartbeat before considering crashed (seconds)
- `max_restarts`: Maximum restarts allowed within time window
- `restart_window`: Time window for restart limiting (seconds)

---

### ConfigWatcher

Watches configuration files for changes and triggers automatic reload.

**Use case**: Development and testing where config changes frequently.

**Configuration**:
```json5
{
  "backgrounds": [
    {
      "type": "ConfigWatcher",
      "config": {
        "config_paths": ["config/spot.json5"],
        "debounce_seconds": 2.0
      }
    }
  ]
}
```

**Note**: Requires `watchdog` package: `pip install watchdog`

---

### APILimiter

Controls API call rate to manage costs and prevent throttling.

**Use case**: Controlling OpenMind API costs in production.

**Configuration**:
```json5
{
  "backgrounds": [
    {
      "type": "APILimiter",
      "config": {
        "max_requests_per_minute": 60,
        "max_cost_per_hour": 1.0,
        "cost_per_request": 0.001
      }
    }
  ]
}
```

**Usage in Code**:
```python
from backgrounds.plugins.api_limiter import APILimiter

limiter = APILimiter()

if limiter.can_make_request():
    # Make API call
    response = make_api_call()
    limiter.record_request()
else:
    # Throttle request
    logging.warning("Rate limit reached")
```

---

## Creating Custom Background Plugins

To create your own background plugin:

1. **Create a new file** in `src/backgrounds/plugins/`:

```python
"""
My Custom Background Plugin
"""

import logging
import threading
import time

from backgrounds.base import Background, BackgroundConfig


class MyCustomPlugin(Background):
    """
    Description of what your plugin does.
    """

    def __init__(self, config: BackgroundConfig = BackgroundConfig()):
        super().__init__(config)

        # Get configuration parameters
        self.my_param = getattr(config, "my_param", "default_value")

        # Start background thread
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True
        )
        self._thread.start()

        logging.info("✅ MyCustomPlugin: Started")

    def _run_loop(self):
        """Main background loop."""
        while self._running:
            try:
                # Your plugin logic here
                time.sleep(1)
            except Exception as e:
                logging.error(f"MyCustomPlugin error: {e}")
```

2. **Register the plugin** in `src/backgrounds/plugins/__init__.py`:

```python
from .my_custom_plugin import MyCustomPlugin

__all__ = [..., "MyCustomPlugin"]
```

3. **Use in your config**:

```json5
{
  "backgrounds": [
    {
      "type": "MyCustomPlugin",
      "config": {
        "my_param": "value"
      }
    }
  ]
}
```

## Best Practices

1. **Keep it lightweight**: Background plugins run continuously. Avoid blocking operations.
2. **Handle errors**: Always wrap logic in try-except to prevent crashes.
3. **Use daemon threads**: Mark background threads as `daemon=True` so they exit cleanly.
4. **Log appropriately**: Use `logging.info()` for regular updates, `logging.warning()` for issues.
5. **Configuration**: Use `getattr(config, "param", default)` for optional parameters.
6. **Clean shutdown**: Implement cleanup in `stop()` if needed.

## Troubleshooting

### Plugin not starting

Check that:
- Plugin is registered in `__init__.py`
- Plugin name in config matches class name exactly
- No syntax errors in plugin code

### High CPU usage

- Increase check intervals
- Reduce work in background loops
- Use sleep/wait properly

### Memory leaks

- Clean up old data structures periodically
- Use weak references where appropriate
- Monitor with HealthCheck plugin

## See Also

- [Main Configuration Guide](../README.md)
- [Background Base Class Reference](../src/backgrounds/base.py)
- [Contributing Guidelines](../CONTRIBUTING.md)
