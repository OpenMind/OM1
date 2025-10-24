# Contribution

## Bug Fix

Fixed async handling bug in `CortexRuntime.run()` that was causing runtime hangs.

The method was mixing different awaitable types in `asyncio.gather()`. Fixed by wrapping all async operations in `asyncio.create_task()`.

```python
# Before
input_listener_task = await self._start_input_listeners()
simulator_start = self._start_simulator_task()
action_start = self._start_action_task()
await asyncio.gather(input_listener_task, cortex_loop_task, simulator_start, action_start)

# After  
input_listener_task = asyncio.create_task(self._start_input_listeners())
simulator_task = asyncio.create_task(self._start_simulator_task())
action_task = asyncio.create_task(self._start_action_task())
await asyncio.gather(input_listener_task, cortex_loop_task, simulator_task, action_task)
```

## Runtime Enhancements

Added `runtime_enhancements/` package with:
- Circuit breakers and retry mechanisms
- Metrics collection and tracing
- Configuration validation
- Performance caching
- Safety validation

```python
from runtime_enhancements import EnhancedCortexRuntime
from runtime.single_mode.config import RuntimeConfig

config = RuntimeConfig.load("spot")
runtime = EnhancedCortexRuntime(config)
await runtime.run()
```

## Testing

Added test suite for async task handling and runtime behavior.

```bash
python3 -m pytest tests/runtime/single_mode/ -v
```

## Files Changed

- `src/runtime/single_mode/cortex.py` - Fixed async bug
- `tests/runtime/single_mode/test_cortex_bug_fix.py` - Added tests
- `runtime_enhancements/` - New enhancement package

Backward compatible changes that improve runtime stability.
