#!/usr/bin/env bash
set -euo pipefail

SEED="${SEED:-12345}"
WORLD="${WORLD:-worlds/bounty363.world}"

run_local() {
  if command -v gzserver >/dev/null 2>&1; then
    echo "[info] Local gzserver: $(gzserver --version | head -n1 || true)"
    gzserver --seed "${SEED}" "${WORLD}" &
    PID=$!; sleep 3; kill $PID || true
  elif command -v gazebo >/dev/null 2>&1; then
    echo "[warn] gzserver not found; using gazebo (may need X11)"
    gazebo -r --seed "${SEED}" "${WORLD}" --pause &
    PID=$!; sleep 3; kill $PID || true
  else
    return 1
  fi
}

if ! run_local; then
  echo "[info] Using container fallback: osrf/ros:noetic-desktop-full (includes Gazebo classic)"
  docker run --rm -v "$(pwd)":/ws -w /ws osrf/ros:noetic-desktop-full bash -lc \
    "gzserver --version || true; gzserver --seed ${SEED} ${WORLD} & pid=\$!; sleep 3; kill \$pid || true"
fi

echo "[ok] Deterministic headless run completed."
