#!/usr/bin/env python3
"""
Simple Zenoh subscriber to verify /odometer_state data transfer
"""

import zenoh

# Connect to zenoh on localhost:7447
print("Connecting to Zenoh...")
config = zenoh.Config()
config.insert_json5("mode", '"client"')
config.insert_json5("connect/endpoints", '["tcp/127.0.0.1:7447"]')
session = zenoh.open(config)
print("✓ Connected to Zenoh bridge on tcp/127.0.0.1:7447")

# Track received messages
received = {"count": 0}


def listener(sample):
    received["count"] += 1
    print(f"\n[Message #{received['count']}] {sample.key_expr}")
    print(f"  Timestamp: {sample.timestamp}")
    print(f"  Payload size: {len(sample.payload)} bytes")
    try:
        # Try to decode as UTF-8 string
        decoded = sample.payload.to_bytes().decode("utf-8", errors="ignore")
        if len(decoded) < 200:
            print(f"  Value: {decoded}")
        else:
            print(f"  Value: {decoded[:200]}...")
    except Exception:
        pass


# Subscribe to odometer_state
print("\nSubscribing to **/odometer_state...")
sub = session.declare_subscriber("**/odometer_state", listener)

print("\n✓ Listening for /odometer_state messages...")
print("(Press Ctrl+C to stop)\n")

try:
    import time

    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\n\n✓ Stopping subscriber...")
    print(f"Total messages received: {received['count']}")
    sub.undeclare()
    session.close()
    print("✓ Connection closed")
