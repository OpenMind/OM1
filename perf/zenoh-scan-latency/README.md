# Zenoh scan-latency comparison (Go vs Python)

A minimal subscriber that measures the delay between when a lidar `scan` message
was **sent** and when this subscriber **received** it, so we can compare the Go
and Python zenoh clients on identical input.

- `zenoh_scan_latency.py` — Python subscriber (`eclipse-zenoh` 1.x)
- The matching Go subscriber (`main.go`) lives on the **`perf-metrics-go`**
  branch under the same path (`perf/zenoh-scan-latency/`); this branch is the
  Python build, so only the Python side ships here. Run each from its own
  checkout against the same router/key.

"Sent time" is the ROS2 `header.stamp` embedded in the CDR payload (`int32 sec`
at byte 4, `uint32 nanosec` at byte 8, little-endian) — the Go side reads the
exact same bytes, so the two are directly comparable. Delay =
`received_time − header.stamp`, reported as min / p50 / p95 / p99 / max / mean
plus message rate.

## ⚠️ Clock requirement
`received_time` is the local system clock; `header.stamp` is set on the robot.
The delay is only meaningful if the two clocks agree. **Run on the same host as
the lidar/bridge** (e.g. the Jetson), or ensure NTP/PTP sync. Even if the
absolute delay is off by a clock skew, the **Go-vs-Python difference** is valid
as long as both run on the same machine.

## Find the scan key
The ROS2 topic `/<robot_ns>/pi/scan` is bridged to a zenoh key. To discover the
exact key, subscribe against a wildcard and watch what arrives:

```bash
python3 perf/zenoh-scan-latency/zenoh_scan_latency.py --key '**/scan' -v --duration 5
# or list ROS topics if the bridge host has ROS:
ros2 topic list | grep scan
```

Typical value looks like `OM742d35Cc6634/pi/scan`.

## Run the comparison
Run the Python subscriber here and the Go subscriber (from the `perf-metrics-go`
checkout) **at the same time** so they see the same messages:

```bash
# Python (this branch)
pip install "eclipse-zenoh>=1.0,<2.0"
python3 perf/zenoh-scan-latency/zenoh_scan_latency.py --key 'OM742d35Cc6634/pi/scan' --duration 30

# Go (from a perf-metrics-go checkout)
make build && go build -o build/zenoh-scan-latency ./perf/zenoh-scan-latency
./build/zenoh-scan-latency -key 'OM742d35Cc6634/pi/scan' -duration 30s
```

Compare the two summaries (same window, same key). Add `-v` / `--verbose` for
per-message delay if you want the raw distribution.

## Measure message rate / jitter (e.g. `/odom`)
Pass `--rate` to measure the **arrival rate in Hz** and the inter-arrival
**jitter** instead of latency. Rate mode ignores the payload and times only the
gaps between consecutive receives, so **no clock sync is needed** and it works
on any topic — including `nav_msgs/Odometry` on `**/odom`:

```bash
# Python (this branch)
python3 perf/zenoh-scan-latency/zenoh_scan_latency.py --key '**/odom' --rate --duration 30

# Go (from a perf-metrics-go checkout)
go run ./perf/zenoh-scan-latency -key '**/odom' -rate -duration 30s
```

The summary reports `message rate: N Hz` plus the inter-arrival distribution
(min / p50 / p95 / p99 / max / mean / stddev in ms). Run both at the same time
to compare how steadily the Go vs Python zenoh client delivers the same stream.
This pairs with the in-process `tick-timing` logs (the cortex loop prints its
actual-vs-expected tick interval) to check the Python `asyncio.sleep` drift
against Go's timer.

## Notes
- Defaults to **client mode** connecting to `tcp/127.0.0.1:7447`; override with
  `--endpoint` if the router is elsewhere.
- "skipped(short)" counts messages whose payload is too small to contain a
  header.stamp (shouldn't happen for real LaserScan messages).
- If you'd rather measure zenoh's own publish timestamp instead of the sensor
  stamp, that requires the publisher to enable zenoh timestamping and a small
  change to read `sample.timestamp`; the header.stamp approach here needs no
  such setup and is identical across both languages.
