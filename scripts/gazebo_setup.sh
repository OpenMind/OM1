#!/usr/bin/env bash
set -euo pipefail
if ! command -v docker >/dev/null 2>&1 && ! command -v gazebo >/dev/null 2>&1; then
  echo "[warn] Neither docker nor gazebo found. Install one of them to run demo."
fi
echo "[info] Quickstart:"
echo "  ./scripts/gazebo_run_deterministic.sh"
