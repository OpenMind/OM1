#!/usr/bin/env python3
"""
Simple Zenoh subscriber to verify /odometer_state data transfer
"""
import zenoh

# Connect to zenoh (defaults to localhost)
print("Connecting to Zenoh...")
session = zenoh.open()
print("✓ Connected to Zenoh bridge")

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
