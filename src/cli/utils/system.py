"""System check utilities for OM1 CLI.

Provides system health checks, dependency verification, and environment validation.
"""

import os
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional

# Config to optional dependency mapping
CONFIG_DEPENDENCIES = {
    # Configs that need no extras (minimal install)
    "ollama": [],
    "conversation": [],
    "deepseek": [],
    "openrouter": [],
    # Configs that need vision
    "open_ai": ["vision"],
    "gemini": ["vision"],
    "grok": ["vision"],
    # Configs that need specific extras
    "twitter": ["twitter"],
    "tesla": ["vehicles"],
    "yolo": ["yolo"],
    # Robot configs that need robotics
    "spot": ["robotics"],
    "spot_modes": ["robotics"],
    "turtlebot4": ["robotics"],
    "turtlebot4_lidar": ["robotics"],
    "turtlebot4_lidar_gps": ["robotics"],
    "unitree_go2_basic": ["robotics"],
    "unitree_go2_autonomy": ["robotics"],
    "unitree_go2_autonomy_advance": ["robotics"],
    "unitree_go2_autonomy_sim": ["robotics"],
    "unitree_go2_mapper": ["robotics"],
    "unitree_go2_mapper_bits": ["robotics"],
    "unitree_go2_mapper_bits_ai": ["robotics"],
    "unitree_go2_modes": ["robotics"],
    "unitree_go2_remote": ["robotics"],
    "unitree_g1_humanoid": ["robotics"],
    "ubtech_yanshee": ["robotics"],
    # Blockchain configs
    "robot_wallet_safe": ["blockchain"],
}

# API keys required by each config
CONFIG_API_KEYS = {
    "ollama": [],  # Local LLM, no API key needed
    "conversation": ["OPENAI_API_KEY"],
    "open_ai": ["OPENAI_API_KEY"],
    "deepseek": ["DEEPSEEK_API_KEY"],
    "gemini": ["GOOGLE_API_KEY"],
    "grok": ["XAI_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY"],
    "twitter": ["TWITTER_API_KEY", "TWITTER_API_SECRET"],
    "tesla": ["DIMO_CLIENT_ID", "DIMO_API_KEY"],
}

# Packages required for each optional group
OPTIONAL_PACKAGES = {
    "vision": ["opencv-python", "tensorflow", "torch", "deepface"],
    "yolo": ["ultralytics"],
    "twitter": ["tweepy"],
    "vehicles": ["dimo"],
    "blockchain": ["web3", "cdp-sdk"],
    "sensors": ["pynmeagps", "bleak", "hid", "pyserial", "pynput"],
    "robotics": [
        "opencv-python",
        "tensorflow",
        "torch",
        "deepface",
        "ultralytics",
        "pynmeagps",
        "bleak",
    ],
}


class CheckStatus(Enum):
    """Status of a health check."""

    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


@dataclass
class CheckResult:
    """Result of a single health check."""

    name: str
    status: CheckStatus
    message: str
    details: Optional[str] = None
    fix_hint: Optional[str] = None


def check_python_version() -> CheckResult:
    """Check Python version is 3.10+."""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version.major >= 3 and version.minor >= 10:
        return CheckResult(
            name="Python Version",
            status=CheckStatus.OK,
            message=f"Python {version_str}",
        )
    return CheckResult(
        name="Python Version",
        status=CheckStatus.FAIL,
        message=f"Python {version_str} (requires 3.10+)",
        fix_hint="Install Python 3.10 or higher",
    )


def check_uv_installed() -> CheckResult:
    """Check if uv package manager is installed."""
    uv_path = shutil.which("uv")
    if uv_path:
        try:
            result = subprocess.run(
                ["uv", "--version"], capture_output=True, text=True, timeout=5
            )
            version = (
                result.stdout.strip().split()[-1]
                if result.returncode == 0
                else "unknown"
            )
            return CheckResult(
                name="uv Package Manager",
                status=CheckStatus.OK,
                message=f"uv {version}",
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    return CheckResult(
        name="uv Package Manager",
        status=CheckStatus.FAIL,
        message="Not installed",
        fix_hint="Install: curl -LsSf https://astral.sh/uv/install.sh | sh",
    )


def check_venv_active() -> CheckResult:
    """Check if virtual environment is active."""
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        return CheckResult(
            name="Virtual Environment",
            status=CheckStatus.OK,
            message=f"Active: {Path(venv).name}",
        )
    return CheckResult(
        name="Virtual Environment",
        status=CheckStatus.WARN,
        message="Not active",
        fix_hint="Run: source .venv/bin/activate",
    )


def check_env_file() -> CheckResult:
    """Check if .env file exists."""
    env_path = Path(".env")
    if env_path.exists():
        return CheckResult(
            name=".env File",
            status=CheckStatus.OK,
            message="Found",
        )
    return CheckResult(
        name=".env File",
        status=CheckStatus.WARN,
        message="Not found",
        fix_hint="Run: cp .env.example .env",
    )


def check_api_key(key_name: str) -> CheckResult:
    """Check if specific API key is set."""
    value = os.environ.get(key_name, "")
    if value and value not in ("", "your_key_here", "sk-..."):
        # Mask the key for display
        masked = value[:4] + "..." + value[-4:] if len(value) > 8 else "***"
        return CheckResult(
            name=f"{key_name}",
            status=CheckStatus.OK,
            message=f"Set ({masked})",
        )
    return CheckResult(
        name=f"{key_name}",
        status=CheckStatus.FAIL,
        message="Not set",
        fix_hint=f"Add to .env: {key_name}=your_key",
    )


def check_ollama_running() -> CheckResult:
    """Check if Ollama is running on localhost."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(("localhost", 11434))
        sock.close()
        if result == 0:
            return CheckResult(
                name="Ollama Service",
                status=CheckStatus.OK,
                message="Running on localhost:11434",
            )
    except (socket.error, OSError):
        pass
    return CheckResult(
        name="Ollama Service",
        status=CheckStatus.WARN,
        message="Not running",
        fix_hint="Start with: ollama serve",
    )


def check_package_installed(package_name: str) -> bool:
    """Check if a Python package is installed."""
    try:
        # Handle package name variations
        import_name = package_name.replace("-", "_").split("[")[0]
        __import__(import_name)
        return True
    except ImportError:
        return False


def check_optional_deps(extras: List[str]) -> CheckResult:
    """Check if optional dependencies for given extras are installed."""
    missing = []
    for extra in extras:
        packages = OPTIONAL_PACKAGES.get(extra, [])
        for pkg in packages:
            if not check_package_installed(pkg):
                missing.append(pkg)

    if not missing:
        return CheckResult(
            name="Optional Dependencies",
            status=CheckStatus.OK,
            message=f"All packages for {', '.join(extras)} installed",
        )

    return CheckResult(
        name="Optional Dependencies",
        status=CheckStatus.WARN,
        message=f"Missing: {', '.join(missing[:3])}{'...' if len(missing) > 3 else ''}",
        fix_hint=f"Run: uv sync --extra {' --extra '.join(extras)}",
    )


def check_microphone() -> CheckResult:
    """Check if microphone is available."""
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        input_devices = [d for d in devices if d.get("max_input_channels", 0) > 0]
        if input_devices:
            default = sd.query_devices(kind="input")
            return CheckResult(
                name="Microphone",
                status=CheckStatus.OK,
                message=f"Found: {default.get('name', 'Unknown')[:30]}",
            )
    except Exception:
        pass
    return CheckResult(
        name="Microphone",
        status=CheckStatus.WARN,
        message="Not detected",
        fix_hint="Connect a microphone",
    )


def check_camera() -> CheckResult:
    """Check if camera is available."""
    try:
        import cv2

        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cap.release()
            return CheckResult(
                name="Camera",
                status=CheckStatus.OK,
                message="Found: Device 0",
            )
        cap.release()
    except Exception:
        pass
    return CheckResult(
        name="Camera",
        status=CheckStatus.WARN,
        message="Not detected",
        fix_hint="Connect a camera or webcam",
    )


def check_disk_space() -> CheckResult:
    """Check available disk space."""
    try:
        import shutil

        total, used, free = shutil.disk_usage(".")
        free_gb = free / (1024**3)
        if free_gb >= 5:
            return CheckResult(
                name="Disk Space",
                status=CheckStatus.OK,
                message=f"{free_gb:.1f} GB free",
            )
        elif free_gb >= 1:
            return CheckResult(
                name="Disk Space",
                status=CheckStatus.WARN,
                message=f"{free_gb:.1f} GB free (low)",
            )
        else:
            return CheckResult(
                name="Disk Space",
                status=CheckStatus.FAIL,
                message=f"{free_gb:.1f} GB free (critical)",
                fix_hint="Free up disk space",
            )
    except Exception:
        return CheckResult(
            name="Disk Space",
            status=CheckStatus.WARN,
            message="Could not check",
        )


def check_network() -> CheckResult:
    """Check network connectivity."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex(("api.openai.com", 443))
        sock.close()
        if result == 0:
            return CheckResult(
                name="Network",
                status=CheckStatus.OK,
                message="Connected (api.openai.com reachable)",
            )
    except (socket.error, OSError):
        pass
    return CheckResult(
        name="Network",
        status=CheckStatus.WARN,
        message="Limited connectivity",
        fix_hint="Check internet connection",
    )


def run_all_checks(config_name: Optional[str] = None) -> List[CheckResult]:
    """Run all system health checks.

    If config_name is provided, also checks config-specific requirements.
    """
    results = []

    # Core checks
    results.append(check_python_version())
    results.append(check_uv_installed())
    results.append(check_venv_active())
    results.append(check_env_file())
    results.append(check_disk_space())
    results.append(check_network())

    # Config-specific checks
    if config_name:
        # Check required extras
        extras = CONFIG_DEPENDENCIES.get(config_name, [])
        if extras:
            results.append(check_optional_deps(extras))

        # Check API keys
        api_keys = CONFIG_API_KEYS.get(config_name, [])
        for key in api_keys:
            results.append(check_api_key(key))

        # Check Ollama for local LLM configs
        if config_name == "ollama":
            results.append(check_ollama_running())

    # Hardware checks (optional, may fail)
    try:
        results.append(check_microphone())
    except Exception:
        pass

    return results


def get_required_extras_for_config(config_name: str) -> List[str]:
    """Get list of required optional extras for a config."""
    return CONFIG_DEPENDENCIES.get(config_name, [])


def get_required_api_keys_for_config(config_name: str) -> List[str]:
    """Get list of required API keys for a config."""
    return CONFIG_API_KEYS.get(config_name, [])
