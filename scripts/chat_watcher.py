#!/usr/bin/env python3
"""
Chat Watcher Utility
====================

A standalone script to filter, colorize, and rotate logs from the OM1 runtime.
Designed to be used with the ConsoleInput plugin for a clean "chat-like" experience
in headless environments.

Usage:
    PYTHONUNBUFFERED=1 uv run src/run.py conversation 2>&1 | python3 scripts/chat_watcher.py
"""

import sys
import re
import os
import shutil
from datetime import datetime

# Configuration
LOG_FILE = "chat_history.log"
MAX_SIZE = 10 * 1024 * 1024  # 10MB
BACKUP_COUNT = 3

# ANSI Colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
CYAN = "\033[96m"

def rotate_logs():
    """Rotate logs if size exceeds MAX_SIZE."""
    try:
        if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > MAX_SIZE:
            if os.path.exists(f"{LOG_FILE}.{BACKUP_COUNT}"):
                os.remove(f"{LOG_FILE}.{BACKUP_COUNT}")
            for i in range(BACKUP_COUNT - 1, 0, -1):
                src = f"{LOG_FILE}.{i}"
                dst = f"{LOG_FILE}.{i+1}"
                if os.path.exists(src):
                    os.rename(src, dst)
            os.rename(LOG_FILE, f"{LOG_FILE}.1")
    except Exception:
        # Silently fail on log rotation errors to avoid crashing the stream
        pass

def write_to_file(clean_line):
    """Write timestamped log to file."""
    rotate_logs()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{timestamp} - {clean_line}\n")
    except Exception:
        pass

def process_stream():
    """Process stdin stream line by line."""
    print(f"{CYAN}[SYSTEM] Chat Watcher Started. Waiting for OM1...{RESET}")
    print(f"{CYAN}[SYSTEM] Logging to: {os.path.abspath(LOG_FILE)}{RESET}\n")

    try:
        for line in sys.stdin:
            line = line.strip()
            
            # 1. Capture User Input (from ConsoleInput plugin)
            if "[USER SAID]:" in line:
                content = line.split("[USER SAID]:", 1)[1].strip()
                print(f"{GREEN}[USER]: {content}{RESET}")
                write_to_file(f"[USER]: {content}")
                continue

            # 2. Capture Robot Output (standard TTS logging)
            if "audio_stream:" in line:
                content = line.split("audio_stream:", 1)[1].strip()
                print(f"{YELLOW}[SPOT]: {content}{RESET}")
                write_to_file(f"[SPOT]: {content}")
                continue

            # 3. Capture Errors
            if "ERROR" in line or "Traceback" in line:
                # Filter out some common benign network errors if needed
                print(f"{RED}{line}{RESET}")
                write_to_file(f"[ERROR]: {line}")
                continue
            
            # (Optional) Uncomment to debug: print everything else faintly
            # print(f"\033[90m{line}{RESET}")

    except KeyboardInterrupt:
        print(f"\n{CYAN}[SYSTEM] Log watcher stopped.{RESET}")

if __name__ == "__main__":
    process_stream()
