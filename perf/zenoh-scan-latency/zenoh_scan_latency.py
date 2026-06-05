#!/usr/bin/env python3
"""Subscribe to a zenoh scan topic and measure send->receive delay (Python).

Counterpart to main.go. Uses the SAME definition of "sent time" — the ROS2
header.stamp embedded in the CDR payload (bytes 4..11: int32 sec, uint32
nanosec, little-endian) — so the Python and Go numbers are directly comparable.

Install the zenoh client (match the Go side's zenoh 1.x protocol):

    pip install "eclipse-zenoh>=1.0,<2.0"

Run:

    python3 zenoh_scan_latency.py --key '<robot_ns>/pi/scan' --duration 30
"""

import argparse
import json
import struct
import time

import zenoh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="**/scan", help="zenoh key expression (e.g. <ns>/pi/scan)")
    ap.add_argument("--endpoint", default="tcp/127.0.0.1:7447", help="zenoh router endpoint (client mode)")
    ap.add_argument("--duration", type=float, default=30.0, help="seconds to sample before summary")
    ap.add_argument("-v", "--verbose", action="store_true", help="print per-message delay")
    args = ap.parse_args()

    delays_ms = []
    skipped = 0

    def cb(sample):
        nonlocal skipped
        recv = time.time()  # seconds since epoch (system clock, like ROS uses)
        payload = bytes(sample.payload)  # zenoh 1.x: ZBytes -> bytes
        if len(payload) < 12:
            skipped += 1
            return
        # CDR: [0:4] encapsulation header, then std_msgs/Header.stamp.
        sec = struct.unpack_from("<i", payload, 4)[0]
        nsec = struct.unpack_from("<I", payload, 8)[0]
        sent = sec + nsec * 1e-9
        ms = (recv - sent) * 1000.0
        delays_ms.append(ms)
        if args.verbose:
            print(f"delay={ms:.3f} ms (sent={sent:.9f})")

    conf = zenoh.Config()
    conf.insert_json5("mode", json.dumps("client"))
    conf.insert_json5("connect/endpoints", json.dumps([args.endpoint]))

    with zenoh.open(conf) as session:
        session.declare_subscriber(args.key, cb)
        print(f"[python] subscribed to {args.key!r} via {args.endpoint} — sampling for {args.duration}s ...")
        time.sleep(args.duration)

    print_summary("python", args.key, delays_ms, skipped, args.duration)


def pct(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    import math
    rank = max(0, math.ceil(p / 100 * len(sorted_vals)) - 1)
    return sorted_vals[min(rank, len(sorted_vals) - 1)]


def print_summary(lang, key, delays, skipped, dur):
    print(f"\n=== {lang} zenoh scan latency summary ===")
    print(f"key: {key} | window: {dur}s | messages: {len(delays)} | skipped(short): {skipped}")
    if not delays:
        print("no messages received — check the key expression and that the bridge is publishing")
        return
    s = sorted(delays)
    mean = sum(delays) / len(delays)
    print(
        f"delay ms — min {s[0]:.3f} | p50 {pct(s,50):.3f} | p95 {pct(s,95):.3f} | "
        f"p99 {pct(s,99):.3f} | max {s[-1]:.3f} | mean {mean:.3f}"
    )
    print(f"message rate: {len(delays)/dur:.1f} msg/s")


if __name__ == "__main__":
    main()
