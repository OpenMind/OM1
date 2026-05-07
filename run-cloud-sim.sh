#!/usr/bin/env bash
# Run OM1 against the OpenMind cloud simulator.
#
# Requires: OM_API_KEY exported, a "Ready" Cloud Simulator on the portal,
# and `uv sync` already run in this repo.

set -euo pipefail

OM1_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${OM_API_KEY:?OM_API_KEY not set — create one at portal.openmind.com}"

export OPENMIND_CLOUD_URL="wss://api.openmind.com/api/core/simulation/zenoh?api_key=${OM_API_KEY}"

cd "$OM1_DIR"
exec uv run --no-project src/run.py cloud_sim
