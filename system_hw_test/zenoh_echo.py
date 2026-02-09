import argparse
import asyncio
import json
import os
import sys

# Add src directory to path to import local zenoh_msgs if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import zenoh


def on_message(sample):
    """Callback for Zenoh message subscriber."""
    try:
        payload = sample.payload.to_bytes()
        # Try to decode as string first
        try:
            decoded = payload.decode("utf-8")
            # Try to format as JSON if possible
            try:
                json_obj = json.loads(decoded)
                formatted = json.dumps(json_obj, indent=2)
                print(f"[{sample.key_expr}]:\n{formatted}")
            except json.JSONDecodeError:
                print(f"[{sample.key_expr}]: {decoded}")
        except UnicodeDecodeError:
            print(f"[{sample.key_expr}]: {payload}")
    except Exception as e:
        print(f"Error processing message on {sample.key_expr}: {e}")


async def main():
    parser = argparse.ArgumentParser(
        description="Echo Zenoh messages like 'ros2 topic echo'"
    )
    parser.add_argument(
        "topic", nargs="?", default="**", help="Topic to subscribe to (default: '**')"
    )
    parser.add_argument(
        "--endpoint", default="tcp/127.0.0.1:7447", help="Zenoh endpoint to connect to"
    )
    parser.add_argument(
        "--no-multicast", action="store_true", help="Disable multicast discovery"
    )

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
