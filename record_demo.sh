#!/usr/bin/env bash
set -euo pipefail
PYINT="/srv/om1-dev/OM1/.venv/bin/python"
if [ ! -x "$PYINT" ]; then
  PYINT="$(command -v python3)"
fi

su - demo -s /bin/bash <<EOS
set -euo pipefail
cd /srv/om1-dev/OM1
unset HISTFILE || true
export PS1="\$ "
export PYTHONPATH="$(pwd)"

# Stop otomatis ≤5 detik, ambil 6 baris, senyapkan stderr
timeout 5s \
MOCK=1 SENSOR_TYPE=bme280 SENSOR_ID=bme280-1 POLL_MS=500 \
"$PYINT" -m om1_plugins.input_bme280.plugin 2>/dev/null | head -n 6

echo "[done] demo snippet recorded."
EOS
