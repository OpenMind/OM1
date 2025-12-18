#!/usr/bin/env bash
set -euo pipefail
if ! command -v docker >/dev/null 2>&1 && ! command -v gzserver >/dev/null 2>&1 && ! command -v gazebo >/dev/null 2>&1; then
  echo "[warn] Neither docker nor Gazebo found. Install one of them to run the demo."
fi
echo "[info] Quickstart:"
echo "  ./scripts/gazebo_run_deterministic.sh"
echo "[note] If Gazebo isn't installed locally, a container fallback (osrf/ros:noetic-desktop-full) will be used."
