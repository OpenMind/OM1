# Performance Validation: Python (`main`) vs Go

This document describes the metrics, instrumentation and tooling used to compare
the Python (`main`) and Go runtimes of OM1 head-to-head on **NVIDIA Jetson Thor**.

Both runtimes already export latency metrics to Prometheus. This work adds CPU,
memory, power, thermal, startup-time and footprint measurements so the two can be
compared on identical infrastructure. The same monitoring stack and benchmark
harness exist on both branches (`perf-metrics-go`, `perf-metrics-python`); only
the language-specific instrumentation differs.

## What is collected, and how

| Metric | Source | Prometheus query (key series) |
|---|---|---|
| **CPU utilization** | cAdvisor + app process collector | `rate(container_cpu_usage_seconds_total{name="om1"}[1m])`, `rate(process_cpu_seconds_total{job="om1"}[1m])` |
| **Peak RSS** | cAdvisor + app process collector | `max_over_time(container_memory_rss{name="om1"}[$range])`, `process_resident_memory_bytes` |
| **Heap allocation rate** | app runtime (language-specific) | Go: `rate(go_memstats_alloc_bytes_total[1m])` · Python: `rate(om1_python_allocated_blocks[1m])` |
| **Power consumption (W)** | Jetson power exporter (sysfs INA rails) | `om1_jetson_power_watts{rail="total"}` |
| **Thermal throttling** | Jetson power exporter (cpufreq + thermal) | `om1_jetson_cpu_throttled`, `increase(om1_jetson_throttle_events_total[$range])` |
| **SoC temperature** | Jetson power exporter (thermal zones) | `om1_jetson_temp_celsius{zone="..."}` |
| **Power efficiency** | derived (throughput ÷ power) | `rate(om1_llm_latency_seconds_count[1m]) / on() om1_jetson_power_watts{rail="total"}` |
| **Cold/warm start time** | app gauge + benchmark wall clock | `om1_startup_duration_seconds` |
| **Binary / image size** | benchmark script (build-time) | n/a — see `perf/benchmark.sh` report |

### Prometheus-native vs. handled separately

- **Fully Prometheus-native** (scraped as time series, viewable in Grafana):
  CPU, memory/RSS, heap allocation rate, power, temperature, throttling, latency,
  and power efficiency (derived in PromQL). No manual sampling needed.
- **Hybrid** — *cold/warm start time*: the app records
  `om1_startup_duration_seconds` (process start → runtime ready) which Prometheus
  scrapes, and `perf/benchmark.sh` independently measures wall-clock readiness so
  the two cross-check each other.
- **Handled separately** — *binary/image size*: a static build-time number, not a
  time series. `perf/benchmark.sh` reads it via `docker image inspect` (image) and
  `stat`/`du` inside the container (Go binary vs Python venv) and writes it into
  the report.

## Why this approach on Jetson Thor

ARM SoCs have **no RAPL**, so Intel/AMD power tools (Scaphandre, Kepler-via-RAPL)
do not apply. Jetson instead exposes on-board **INA3221 power monitors** and SoC
thermal sensors through sysfs. `perf/jetson_power_exporter.py` reads those paths
directly (no dependency on the `tegrastats` binary) and serves them as Prometheus
metrics, so it runs inside a container with `/sys` bind-mounted read-only. It
auto-discovers rails/zones and skips anything unreadable, so it is safe to run on
a dev laptop too (it will simply report fewer series).

Throttling is derived: a core running below 95% of its max frequency
(`scaling_cur_freq` vs `cpuinfo_max_freq`) is counted as throttled, and each
not-throttled → throttled transition increments `om1_jetson_throttle_events_total`.

## Monitoring stack

`docker-compose.yml` defines, in addition to `om1`, `prometheus`, and `grafana`:

- **cadvisor** (`:8080`) — per-container CPU/memory.
- **node-exporter** (`:9100`, host PID/net) — host CPU/mem, thermal zones,
  cpufreq, hwmon sensors.
- **jetson-power-exporter** (`:9300`) — INA power rails, SoC temps, throttling.

`prometheus.yml` scrapes all of the above plus the app on `:9090` at a 1s interval.
Grafana auto-provisions the **OM1 Performance Validation (Python vs Go)** dashboard
(`grafana/dashboards/om1-perf-validation.json`), which contains both the Go and
Python heap-rate queries so the same dashboard works on either branch.

## Running a benchmark

On the Jetson Thor, for each branch:

```bash
git checkout perf-metrics-go     # or perf-metrics-python
docker compose up -d --build     # brings up om1 + full monitoring stack
# ...apply your standard workload (same on both branches)...
perf/benchmark.sh go 120         # label + steady-state window in seconds
#   -> writes perf/results/go-<timestamp>.md
```

Run the identical workload on both branches, then diff the two
`perf/results/*.md` reports. Grafana (`http://<host>:3000`, admin/admin) shows the
live time series during the run.

### Notes

- The `om1` container uses host networking, so Prometheus reaches every exporter
  via `host.docker.internal:<port>`; cAdvisor still labels the container `om1`.
- Set `OM1_TRACEMALLOC=1` on the Python branch to additionally export
  `om1_python_traced_heap_bytes` (byte-accurate heap, with per-allocation
  overhead — enable only during a benchmark, not in production).
- `om1_startup_duration_seconds` is set when the runtime is initialized and about
  to enter its main loop; it is **not** a measure of full workload readiness if
  your config lazily connects to external services.
