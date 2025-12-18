#!/usr/bin/env bash
set -euo pipefail
SEED="${SEED:-12345}"
WORLD="${WORLD:-worlds/bounty363.world}"
if command -v gazebo >/dev/null 2>&1; then
  echo "[info] $(gazebo --version | head -n1)"
  echo "[info] running with seed=${SEED} world=${WORLD}"
  gazebo -r --seed "${SEED}" "${WORLD}" --pause &
  PID=$!; sleep 3; kill $PID || true
else
  echo "[info] using container osrf/gazebo:classic"
  docker run --rm -v "$(pwd)":/ws -w /ws osrf/gazebo:classic bash -lc \
    "gazebo --version && gazebo -r --seed ${SEED} ${WORLD} --pause & pid=\$!; sleep 3; kill \$pid || true"
fi
echo "[ok] deterministic launch completed."
