# Zenoh scan-latency comparison (Go vs Python)

Two minimal subscribers that measure the delay between when a lidar `scan`
message was **sent** and when each subscriber **received** it, so we can compare
Go and Python zenoh clients on identical input.

- `main.go` — Go subscriber (uses the repo's `internal/zenoh`, i.e. zenoh-go v1.9.0)
- `zenoh_scan_latency.py` — Python subscriber (`eclipse-zenoh` 1.x)

Both define "sent time" the same way: the ROS2 `header.stamp` embedded in the
CDR payload (`int32 sec` at byte 4, `uint32 nanosec` at byte 8, little-endian).
Delay = `received_time − header.stamp`, reported as min / p50 / p95 / p99 / max /
mean plus message rate.

## ⚠️ Clock requirement
`received_time` is the local system clock; `header.stamp` is set on the robot.
The delay is only meaningful if the two clocks agree. **Run both subscribers on
the same host as the lidar/bridge** (e.g. the Jetson), or ensure NTP/PTP sync.
Even if the absolute delay is off by a clock skew, the **Go-vs-Python
difference** is valid as long as both run on the same machine.

## Find the scan key
The ROS2 topic `/<robot_ns>/pi/scan` is bridged to a zenoh key. To discover the
exact key, run either subscriber against a wildcard and watch what arrives:

```bash
# Go: subscribe to everything ending in /scan
go run ./perf/zenoh-scan-latency -key '**/scan' -v -duration 5s
# or list ROS topics if the bridge host has ROS:
ros2 topic list | grep scan
```

Typical value looks like `OM742d35Cc6634/pi/scan`.

## Run the comparison
Run them **at the same time** (two terminals) so they see the same messages:

```bash
# Go
make build   # one-time: fetches zenoh-c + sets CGO flags
go build -o build/zenoh-scan-latency ./perf/zenoh-scan-latency
./build/zenoh-scan-latency -key 'OM742d35Cc6634/pi/scan' -duration 30s

# Python
pip install "eclipse-zenoh>=1.0,<2.0"
python3 perf/zenoh-scan-latency/zenoh_scan_latency.py --key 'OM742d35Cc6634/pi/scan' --duration 30
```

Compare the two summaries (same window, same key). Add `-v` / `--verbose` for
per-message delay if you want the raw distribution.

## Notes
- Both default to **client mode** connecting to `tcp/127.0.0.1:7447`; override
  with `-endpoint` / `--endpoint` if the router is elsewhere.
- "skipped(short)" counts messages whose payload is too small to contain a
  header.stamp (shouldn't happen for real LaserScan messages).
- If you'd rather measure zenoh's own publish timestamp instead of the sensor
  stamp, that requires the publisher to enable zenoh timestamping and a small
  change to read `sample.timestamp`; the header.stamp approach here needs no
  such setup and is identical across both languages.
