# Hot Reload Configuration System for OM1

## Overview

This PR implements selective hot-reload for OM1 configuration files, allowing certain configuration fields to be updated without requiring a full system restart. This improves developer experience and reduces downtime during configuration tuning.

## Features

### 1. Selective Field-Based Reloading

Fields are categorized by their reload requirements:

| Category | Fields | Behavior |
|----------|--------|----------|
| **HOT_RELOAD** | `name`, `system_prompt_base`, `system_governance`, `system_prompt_examples`, `system_prompt_addons` | Applied immediately |
| **VALIDATE_FIRST** | `hertz` | Validated before applying |
| **RESTART_REQUIRED** | `cortex_llm`, `agent_inputs`, `agent_actions`, `simulators`, `backgrounds`, `api_key` | Triggers full reload |
| **IGNORE** | `$schema`, `_version` | Skipped |

### 2. Deep Change Detection

Unlike PR #1312 which had a bug with complex field comparison, this implementation uses proper deep equality checking:

```python
# This correctly detects that agent_inputs changed even though list length is same
old = {"agent_inputs": [{"type": "VLM", "model": "gpt-4"}]}
new = {"agent_inputs": [{"type": "VLM", "model": "gpt-4o"}]}  # Detected!
```

### 3. Validation Before Apply

Changes to `VALIDATE_FIRST` fields are validated before being applied:

```python
# hertz must be positive
hertz: -5  # Rejected with helpful error message
hertz: 500 # Rejected (too high, max recommended is 100)
hertz: 10  # Accepted
```

### 4. File Watching with Debouncing

- Uses `watchdog` library for efficient event-based file monitoring
- Falls back to polling if watchdog is unavailable
- Debouncing prevents rapid successive reloads (default: 1 second)

### 5. Event Callbacks & History

```python
manager = HotReloadManager(config_path)

# Register callback for reload events
def on_reload(event: ReloadEvent):
    print(f"Reload: success={event.success}, changes={event.diff.changed_fields}")

manager.on_reload(on_reload)

# Get reload statistics
stats = manager.get_reload_stats()
# {'total_reloads': 5, 'successful': 4, 'failed': 1, 'avg_duration_ms': 12.5}
```

### 6. Manual Reload Trigger (CLI Support)

```python
# Programmatically trigger a reload
manager.trigger_reload()

# Or in CortexRuntime
runtime.trigger_config_reload()
```

## Architecture

```
src/runtime/hot_reload/
├── __init__.py      # Public API exports
├── strategies.py    # ReloadStrategy enum and FieldCategory
├── diff.py          # ConfigDiff engine with deep comparison
├── watcher.py       # File watcher with debouncing
├── validator.py     # Config validation with custom validators
└── manager.py       # Main orchestrator (HotReloadManager)
```

## Usage

### Basic Usage

```python
from runtime.hot_reload import HotReloadManager

def apply_changes(changes: dict, requires_restart: bool):
    if requires_restart:
        restart_system()
    else:
        update_config(changes)

manager = HotReloadManager(
    config_path="config/agent.json5",
    apply_callback=apply_changes,
)
manager.start()
```

### Integration with CortexRuntime

See `cortex_integration.py` for the recommended integration pattern.

## Testing

The test suite includes **50+ tests** covering:

- Strategy and field categorization
- Deep equality comparison (including regression test for #1312 bug)
- Config diff detection
- Validation with edge cases
- File watcher with debouncing
- Manager lifecycle and events
- End-to-end integration

Run tests:
```bash
uv run pytest tests/runtime/test_hot_reload.py -v
```

## Comparison with Existing PRs

| Feature | PR #1090 | PR #1312 | This PR |
|---------|----------|----------|---------|
| Architecture | 4 separate files | Inline | 5 files (modular) |
| Deep comparison | Yes | **No (bug)** | Yes |
| Validation | Yes | No | Yes + custom |
| Event callbacks | Basic | No | Full + history |
| CLI support | No | No | **Yes** |
| Tests | 96 | 17 | 50+ |
| Documentation | Code comments | Code comments | **Full docs** |

## Changes

| File | Description |
|------|-------------|
| `src/runtime/hot_reload/__init__.py` | Module exports |
| `src/runtime/hot_reload/strategies.py` | Field categorization |
| `src/runtime/hot_reload/diff.py` | Deep config comparison |
| `src/runtime/hot_reload/watcher.py` | File watching |
| `src/runtime/hot_reload/validator.py` | Config validation |
| `src/runtime/hot_reload/manager.py` | Main orchestrator |
| `src/runtime/cortex_integration.py` | CortexRuntime integration |
| `tests/runtime/test_hot_reload.py` | Test suite |

## Checklist

- [x] Code follows project style guidelines
- [x] All tests pass
- [x] Lint checks pass (ruff, black, isort)
- [x] Thread-safe implementation
- [x] Documentation included
- [x] Deep comparison fixes #1312 bug
- [x] CLI support for manual reload
- [x] Event callbacks for monitoring
- [x] Validation with rollback

Closes #984
