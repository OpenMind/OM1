import argparse
import asyncio
import json
import os
import sys

# Add src directory to path to import local zenoh_msgs if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import zenoh

from zenoh_msgs import Odometer, RemoteControllerState


def deserialize_message(key, payload):
    """
    Attempt to deserialize payload based on key expression.
    Returns formatted string or None if unknown/failed.
    """
    try:
        if "remote_controller_state" in key:
            msg = RemoteControllerState.deserialize(payload)
            return f"RemoteControllerState(event={msg.event}, lx={msg.lx:.2f}, ly={msg.ly:.2f}, rx={msg.rx:.2f}, ry={msg.ry:.2f})"
        elif "odometer_state" in key:
            msg = Odometer.deserialize(payload)
            return f"Odometer(x={msg.x:.2f}, y={msg.y:.2f}, theta={msg.theta:.2f})"
        # Add more message types here as needed
        return None
    except Exception as e:
        return f"<Deserialization Error: {e}>"


def on_message(sample):
    """Callback for Zenoh message subscriber."""
    try:
        payload = sample.payload.to_bytes()
        key = str(sample.key_expr)

        # Try specific deserialization first
        deserialized = deserialize_message(key, payload)
        if deserialized:
            print(f"[{key}]: {deserialized}")
            return

        # Fallback to string/JSON/bytes
        try:
            decoded = payload.decode("utf-8")
            try:
                json_obj = json.loads(decoded)
                formatted = json.dumps(json_obj, indent=2)
                print(f"[{key}]:\n{formatted}")
            except json.JSONDecodeError:
                print(f"[{key}]: {decoded}")
        except UnicodeDecodeError:
            print(f"[{key}]: {payload}")
    except Exception as e:
        print(f"Error processing message on {sample.key_expr}: {e}")


async def main():
    parser = argparse.ArgumentParser(description="Echo Zenoh messages like 'ros2 topic echo'")
    parser.add_argument("topic", nargs="?", default="**", help="Topic to subscribe to (default: '**')")
    parser.add_argument("--endpoint", default="tcp/127.0.0.1:7447", help="Zenoh endpoint to connect to")
    parser.add_argument("--no-multicast", action="store_true", help="Disable multicast discovery")

    args = parser.parse_args()

    print("Connecting to Zenoh...")
    conf = zenoh.Config()

    if args.endpoint:
        conf.insert_json5("connect/endpoints", f'["{args.endpoint}"]')

    if args.no_multicast:
        conf.insert_json5("scouting/multicast/enabled", "false")

    session = zenoh.open(conf)
    print(f"Connected to Zenoh: {session}")
    print(f"Subscribing to: {args.topic}")
    print("Press Ctrl+C to exit...")

    sub = session.declare_subscriber(args.topic, on_message)

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        sub.undeclare()
        session.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
