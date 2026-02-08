"""Process management utilities for OM1 CLI.

Handles PID files, state tracking, and process lifecycle management.
"""

import json
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# OM1 state directory in user's home
OM1_STATE_DIR = Path.home() / ".om1"
PID_DIR = OM1_STATE_DIR / "pids"
STATE_DIR = OM1_STATE_DIR / "state"
LOGS_DIR = OM1_STATE_DIR / "logs"


def ensure_dirs() -> None:
    """Ensure OM1 state directories exist."""
    PID_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def get_pid_file(config_name: str) -> Path:
    """Get path to PID file for a config."""
    return PID_DIR / f"{config_name}.pid"


def get_state_file(config_name: str) -> Path:
    """Get path to state file for a config."""
    return STATE_DIR / f"{config_name}.json"


def write_pid_file(config_name: str) -> None:
    """Write current process PID to file."""
    ensure_dirs()
    pid_file = get_pid_file(config_name)
    pid_file.write_text(str(os.getpid()))


def remove_pid_file(config_name: str) -> None:
    """Remove PID file for a config."""
    pid_file = get_pid_file(config_name)
    pid_file.unlink(missing_ok=True)


def read_pid(config_name: str) -> Optional[int]:
    """Read PID from file, return None if not found or invalid."""
    pid_file = get_pid_file(config_name)
    if not pid_file.exists():
        return None
    try:
        return int(pid_file.read_text().strip())
    except (ValueError, OSError):
        return None


def is_process_running(pid: int) -> bool:
    """Check if a process with given PID is running."""
    if pid <= 0:
        return False
    try:
        # Send signal 0 to check if process exists
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we don't have permission to signal it
        return True


def write_state(
    config_name: str,
    mode: str = "default",
    hot_reload_enabled: bool = True,
    extra_data: Optional[Dict[str, Any]] = None,
) -> None:
    """Write runtime state to file."""
    ensure_dirs()
    state_file = get_state_file(config_name)
    state = {
        "pid": os.getpid(),
        "config": config_name,
        "started_at": datetime.now().isoformat(),
        "mode": mode,
        "hot_reload_enabled": hot_reload_enabled,
        "last_tick": datetime.now().isoformat(),
    }
    if extra_data:
        state.update(extra_data)
    state_file.write_text(json.dumps(state, indent=2))


def update_state(config_name: str, updates: Dict[str, Any]) -> None:
    """Update specific fields in state file."""
    state_file = get_state_file(config_name)
    if not state_file.exists():
        return
    try:
        state = json.loads(state_file.read_text())
        state.update(updates)
        state["last_tick"] = datetime.now().isoformat()
        state_file.write_text(json.dumps(state, indent=2))
    except (json.JSONDecodeError, OSError):
        pass


def read_state(config_name: str) -> Optional[Dict[str, Any]]:
    """Read state from file."""
    state_file = get_state_file(config_name)
    if not state_file.exists():
        return None
    try:
        return json.loads(state_file.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def remove_state(config_name: str) -> None:
    """Remove state file for a config."""
    state_file = get_state_file(config_name)
    state_file.unlink(missing_ok=True)


def cleanup_stale_pid(config_name: str) -> bool:
    """Clean up PID file if process is no longer running.

    Returns True if cleaned up, False if process is still running.
    """
    pid = read_pid(config_name)
    if pid is None:
        return True
    if not is_process_running(pid):
        remove_pid_file(config_name)
        remove_state(config_name)
        return True
    return False


def get_running_agents() -> List[Dict[str, Any]]:
    """Get list of all running OM1 agents."""
    ensure_dirs()
    agents = []

    for pid_file in PID_DIR.glob("*.pid"):
        config_name = pid_file.stem
        pid = read_pid(config_name)

        if pid is None:
            continue

        if not is_process_running(pid):
            # Clean up stale PID file
            cleanup_stale_pid(config_name)
            continue

        state = read_state(config_name) or {}
        started_at = state.get("started_at")

        # Calculate uptime
        uptime_str = "unknown"
        if started_at:
            try:
                start_time = datetime.fromisoformat(started_at)
                uptime = datetime.now() - start_time
                hours, remainder = divmod(int(uptime.total_seconds()), 3600)
                minutes, _ = divmod(remainder, 60)
                if hours > 0:
                    uptime_str = f"{hours}h {minutes}m"
                else:
                    uptime_str = f"{minutes}m"
            except ValueError:
                pass

        agents.append(
            {
                "pid": pid,
                "config": config_name,
                "mode": state.get("mode", "default"),
                "uptime": uptime_str,
                "started_at": started_at,
                "hot_reload": state.get("hot_reload_enabled", True),
                "status": "Running",
            }
        )

    return agents


def stop_agent(
    config_name: str, force: bool = False, timeout: int = 5
) -> tuple[bool, str]:
    """Stop a running agent.

    Returns (success, message) tuple.
    """
    pid = read_pid(config_name)
    if pid is None:
        return False, f"No agent running with config '{config_name}'"

    if not is_process_running(pid):
        cleanup_stale_pid(config_name)
        return True, f"Agent '{config_name}' was not running (cleaned up stale PID)"

    try:
        if force:
            os.kill(pid, signal.SIGKILL)
            remove_pid_file(config_name)
            remove_state(config_name)
            return True, f"Agent '{config_name}' (PID {pid}) force killed"
        else:
            os.kill(pid, signal.SIGTERM)
            # Wait for process to exit
            import time

            for _ in range(timeout * 10):
                if not is_process_running(pid):
                    remove_pid_file(config_name)
                    remove_state(config_name)
                    return True, f"Agent '{config_name}' (PID {pid}) stopped gracefully"
                time.sleep(0.1)

            # Process didn't exit in time
            return (
                False,
                f"Agent '{config_name}' (PID {pid}) did not stop within {timeout}s. Use --force to kill.",
            )
    except ProcessLookupError:
        cleanup_stale_pid(config_name)
        return True, f"Agent '{config_name}' already stopped"
    except PermissionError:
        return False, f"Permission denied to stop agent '{config_name}' (PID {pid})"


def stop_all_agents(
    force: bool = False, timeout: int = 5
) -> List[tuple[str, bool, str]]:
    """Stop all running agents.

    Returns list of (config_name, success, message) tuples.
    """
    results = []
    agents = get_running_agents()

    for agent in agents:
        config_name = agent["config"]
        success, message = stop_agent(config_name, force=force, timeout=timeout)
        results.append((config_name, success, message))

    return results


def is_agent_running(config_name: str) -> bool:
    """Check if an agent with given config is running."""
    pid = read_pid(config_name)
    if pid is None:
        return False
    return is_process_running(pid)


def register_cleanup_handler(config_name: str) -> None:
    """Register signal handlers to clean up on exit."""
    import atexit

    def cleanup():
        remove_pid_file(config_name)
        remove_state(config_name)

    atexit.register(cleanup)

    # Handle SIGTERM gracefully
    original_sigterm = signal.getsignal(signal.SIGTERM)

    def sigterm_handler(signum, frame):
        cleanup()
        if callable(original_sigterm):
            original_sigterm(signum, frame)
        sys.exit(0)

    signal.signal(signal.SIGTERM, sigterm_handler)
