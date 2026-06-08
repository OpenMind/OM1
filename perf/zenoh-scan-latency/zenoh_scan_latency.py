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
    ap.add_argument(
        "--rate",
        action="store_true",
        help="measure inter-arrival rate/jitter (Hz) instead of header.stamp latency; "
        "use for topics like **/odom (no clock sync needed)",
    )
    ap.add_argument(
        "-v", "--verbose", action="store_true", help="print per-message delay (latency) or inter-arrival gap (rate)"
    )
    args = ap.parse_args()

    delays_ms = []
    intervals_ms = []  # gaps between consecutive receives (rate mode)
    state = {"count": 0, "last_recv": None, "skipped": 0}

    def cb(sample):
        recv = time.time()  # seconds since epoch (system clock, like ROS uses)

        if args.rate:
            # Rate mode: ignore the payload and just measure how steady the
            # arrival cadence is. No clock sync required, so it works for any
            # topic (e.g. **/odom) regardless of whether it carries a header.stamp.
            if state["last_recv"] is not None:
                gap = (recv - state["last_recv"]) * 1000.0
                intervals_ms.append(gap)
                if args.verbose:
                    print(f"gap={gap:.3f} ms")
            state["last_recv"] = recv
            state["count"] += 1
            return

        payload = bytes(sample.payload)  # zenoh 1.x: ZBytes -> bytes
        if len(payload) < 12:
            state["skipped"] += 1
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

    mode = "rate" if args.rate else "latency"
    with zenoh.open(conf) as session:
        session.declare_subscriber(args.key, cb)
        print(f"[python] subscribed to {args.key!r} via {args.endpoint} ({mode} mode) — sampling for {args.duration}s ...")
        time.sleep(args.duration)

    if args.rate:
        print_rate_summary("python", args.key, intervals_ms, state["count"], args.duration)
    else:
        print_summary("python", args.key, delays_ms, state["skipped"], args.duration)


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


def print_rate_summary(lang, key, intervals, count, dur):
    """Report arrival rate (Hz) and inter-arrival jitter (clock-sync-free)."""
    print(f"\n=== {lang} zenoh rate summary ===")
    print(f"key: {key} | window: {dur}s | messages: {count}")
    if count == 0:
        print("no messages received — check the key expression and that the publisher is running")
        return
    rate = count / dur
    print(f"message rate: {rate:.2f} Hz ({rate:.2f} msg/s)")
    if not intervals:
        print("only one message — not enough to compute inter-arrival jitter")
        return
    s = sorted(intervals)
    mean = sum(intervals) / len(intervals)
    # Population standard deviation of the gaps = jitter.
    std = (sum((d - mean) ** 2 for d in intervals) / len(intervals)) ** 0.5
    print(
        f"inter-arrival ms — min {s[0]:.3f} | p50 {pct(s,50):.3f} | p95 {pct(s,95):.3f} | "
        f"p99 {pct(s,99):.3f} | max {s[-1]:.3f} | mean {mean:.3f} | stddev {std:.3f}"
    )


if __name__ == "__main__":
    main()
