#!/usr/bin/env bash
#
# Performance validation harness for the OM1 Python (main) vs Go benchmark.
#
# Collects the metrics that are NOT naturally captured as Prometheus time series
# (cold/warm start time, deployment footprint) by driving Docker directly, then
# queries Prometheus for the summary statistics of the metrics that ARE time
# series (CPU, memory, power, thermal, latency) and writes a Markdown report.
#
# Run this once on each branch (perf-metrics-python, perf-metrics-go) on the
# Jetson Thor, then diff the two reports.
#
# Usage:
#   perf/benchmark.sh <label> [window_seconds]
#     label           "python" or "go" (controls language-specific queries + report name)
#     window_seconds  steady-state observation window after warm start (default 120)
#
# Env:
#   PROM_URL   Prometheus base URL (default http://localhost:9091)
#   METRICS_URL  app /metrics URL  (default http://localhost:9090/metrics)
#   COMPOSE     docker compose command (auto-detected: "docker compose" or "docker-compose")
#   IMAGE       app image (default openmindagi/om1:latest)
#
set -euo pipefail

LABEL="${1:?usage: benchmark.sh <python|go> [window_seconds]}"
WINDOW="${2:-120}"
PROM_URL="${PROM_URL:-http://localhost:9091}"
METRICS_URL="${METRICS_URL:-http://localhost:9090/metrics}"
IMAGE="${IMAGE:-openmindagi/om1:latest}"
READY_TIMEOUT="${READY_TIMEOUT:-180}"

# Resolve the Compose command. Prefer an explicit $COMPOSE, then the v2 plugin
# ("docker compose"), then the standalone binary ("docker-compose"). We test that
# the candidate actually runs, since a missing plugin makes "docker compose" fail
# with a confusing "unknown shorthand flag: 'd'" error.
if [ -z "${COMPOSE:-}" ]; then
  if docker compose version >/dev/null 2>&1; then
    COMPOSE="docker compose"
  elif command -v docker-compose >/dev/null 2>&1 && docker-compose version >/dev/null 2>&1; then
    COMPOSE="docker-compose"
  else
    echo "error: neither 'docker compose' nor 'docker-compose' is available" >&2
    exit 1
  fi
fi
echo ">> using compose command: $COMPOSE"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$SCRIPT_DIR/results"
mkdir -p "$OUT_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
REPORT="$OUT_DIR/${LABEL}-${STAMP}.md"

# --- helpers ---------------------------------------------------------------

# promq <promql> -> prints the first scalar/instant value, or "NaN".
promq() {
  local q="$1"
  curl -sG "$PROM_URL/api/v1/query" --data-urlencode "query=$q" \
    | python3 -c 'import sys,json
try:
    d=json.load(sys.stdin)["data"]["result"]
    print(d[0]["value"][1] if d else "NaN")
except Exception:
    print("NaN")'
}

# app_gauge <metric_name> -> value of a bare gauge from the raw /metrics endpoint.
app_gauge() {
  curl -s "$METRICS_URL" | awk -v m="$1" '$1==m {print $2; found=1} END{if(!found)print "NaN"}'
}

# wait_ready -> seconds (wall clock) until the app exposes om1_startup_duration_seconds.
wait_ready() {
  local start now val
  start="$(date +%s.%N)"
  while :; do
    val="$(app_gauge om1_startup_duration_seconds)"
    if [ "$val" != "NaN" ]; then
      now="$(date +%s.%N)"
      echo "$(python3 -c "print(f'{$now-$start:.2f}')")|$val"
      return 0
    fi
    now="$(date +%s.%N)"
    if (( $(python3 -c "print(1 if $now-$start > $READY_TIMEOUT else 0)") )); then
      echo "TIMEOUT|NaN"; return 0
    fi
    sleep 0.25
  done
}

# --- footprint -------------------------------------------------------------

echo ">> measuring deployment footprint"
IMAGE_BYTES="$(docker image inspect "$IMAGE" --format '{{.Size}}' 2>/dev/null || echo 0)"
IMAGE_MB="$(python3 -c "print(f'{$IMAGE_BYTES/1e6:.1f}')")"

if [ "$LABEL" = "go" ]; then
  BIN_BYTES="$(docker run --rm --entrypoint stat "$IMAGE" -c '%s' /usr/local/bin/om1 2>/dev/null || echo 0)"
  FOOTPRINT_NOTE="Go binary: $(python3 -c "print(f'{$BIN_BYTES/1e6:.1f}')") MB (/usr/local/bin/om1)"
else
  BIN_BYTES="$(docker run --rm --entrypoint du "$IMAGE" -sb /app/OM1/.venv 2>/dev/null | awk '{print $1}' || echo 0)"
  FOOTPRINT_NOTE="Python venv: $(python3 -c "print(f'{${BIN_BYTES:-0}/1e6:.1f}')") MB (/app/OM1/.venv)"
fi

# --- cold start ------------------------------------------------------------

echo ">> cold start (force-recreate om1)"
$COMPOSE up -d --force-recreate om1 >/dev/null
COLD="$(wait_ready)"
COLD_WALL="${COLD%%|*}"; COLD_APP="${COLD##*|}"

# --- warm start ------------------------------------------------------------

echo ">> warm start (restart om1)"
$COMPOSE restart om1 >/dev/null
WARM="$(wait_ready)"
WARM_WALL="${WARM%%|*}"; WARM_APP="${WARM##*|}"

# --- steady-state observation ----------------------------------------------

echo ">> observing steady state for ${WINDOW}s"
sleep "$WINDOW"
R="${WINDOW}s"

CPU_CORES="$(promq "rate(container_cpu_usage_seconds_total{name=\"om1\"}[$R])")"
CPU_PROC="$(promq "rate(process_cpu_seconds_total{job=\"om1\"}[$R])")"
RSS_PEAK="$(promq "max_over_time(container_memory_rss{name=\"om1\"}[$R])")"
RSS_PROC_PEAK="$(promq "max_over_time(process_resident_memory_bytes{job=\"om1\"}[$R])")"
WS_PEAK="$(promq "max_over_time(container_memory_working_set_bytes{name=\"om1\"}[$R])")"

if [ "$LABEL" = "go" ]; then
  HEAP_RATE="$(promq "rate(go_memstats_alloc_bytes_total{job=\"om1\"}[$R])")"
  HEAP_UNIT="bytes/s"
else
  HEAP_RATE="$(promq "rate(om1_python_allocated_blocks{job=\"om1\"}[$R])")"
  HEAP_UNIT="blocks/s"
fi

PWR_AVG="$(promq "avg_over_time(om1_jetson_power_watts{rail=\"total\"}[$R])")"
PWR_PEAK="$(promq "max_over_time(om1_jetson_power_watts{rail=\"total\"}[$R])")"
TEMP_MAX="$(promq "max_over_time(max(om1_jetson_temp_celsius)[$R:])")"
THROTTLE_EVENTS="$(promq "increase(om1_jetson_throttle_events_total[$R])")"
THROTTLE_SECS="$(promq "sum_over_time(om1_jetson_cpu_throttled[$R]) * 1")"

# Throughput proxy: completed LLM responses per second over the window.
THROUGHPUT="$(promq "rate(om1_llm_latency_seconds_count[$R])")"
LLM_P50="$(promq "histogram_quantile(0.50, sum by (le) (rate(om1_llm_latency_seconds_bucket[$R])))")"
LLM_P95="$(promq "histogram_quantile(0.95, sum by (le) (rate(om1_llm_latency_seconds_bucket[$R])))")"

# Power efficiency: work done per watt (LLM responses per second per watt).
EFFICIENCY="$(python3 -c "
t='$THROUGHPUT'; p='$PWR_AVG'
try:
    t=float(t); p=float(p)
    print(f'{t/p:.4f}' if p>0 else 'NaN')
except Exception:
    print('NaN')")"

# --- report ----------------------------------------------------------------

human_mb() { python3 -c "v='$1'
try: print(f'{float(v)/1e6:.1f} MB')
except: print('n/a')"; }

cat > "$REPORT" <<EOF
# OM1 Performance Validation — \`$LABEL\` branch

- Generated: $STAMP
- Observation window: ${WINDOW}s
- Image: \`$IMAGE\`
- Prometheus: $PROM_URL

## Deployment footprint
| Metric | Value |
|---|---|
| Docker image size | ${IMAGE_MB} MB |
| Runtime artifact | ${FOOTPRINT_NOTE} |

## Startup time
| Metric | Wall clock | App-reported (om1_startup_duration_seconds) |
|---|---|---|
| Cold start (force-recreate) | ${COLD_WALL}s | ${COLD_APP}s |
| Warm start (restart) | ${WARM_WALL}s | ${WARM_APP}s |

## CPU
| Metric | Value |
|---|---|
| CPU usage (cAdvisor, cores) | ${CPU_CORES} |
| CPU usage (process, cores) | ${CPU_PROC} |

## Memory
| Metric | Value |
|---|---|
| Peak RSS (cAdvisor) | $(human_mb "$RSS_PEAK") |
| Peak RSS (process) | $(human_mb "$RSS_PROC_PEAK") |
| Peak working set (cAdvisor) | $(human_mb "$WS_PEAK") |
| Heap allocation rate | ${HEAP_RATE} ${HEAP_UNIT} |

## Power & thermal (Jetson)
| Metric | Value |
|---|---|
| Avg power (total rail) | ${PWR_AVG} W |
| Peak power (total rail) | ${PWR_PEAK} W |
| Max SoC temperature | ${TEMP_MAX} °C |
| Throttle onset events | ${THROTTLE_EVENTS} |
| Throttled sample-seconds | ${THROTTLE_SECS} |

## Throughput & efficiency
| Metric | Value |
|---|---|
| LLM responses/s | ${THROUGHPUT} |
| LLM latency p50 / p95 | ${LLM_P50}s / ${LLM_P95}s |
| Power efficiency (responses/s per watt) | ${EFFICIENCY} |
EOF

echo ">> report written: $REPORT"
cat "$REPORT"
