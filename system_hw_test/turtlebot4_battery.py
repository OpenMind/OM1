import argparse
import time

import zenoh

parser = argparse.ArgumentParser()
parser.add_argument("--URID", help="your robot's URID, when using Zenoh", type=str)
print(parser.format_help())

args = parser.parse_args()


def listener(sample):
    bytesI = sample.payload.to_bytes()
    print(f"Received {bytesI}")


if __name__ == "__main__":

    URID = args.URID
    print(f"Using Zenoh to connect to robot using {URID}")
    print("[INFO] Opening zenoh session...")

    with zenoh.open(zenoh.Config()) as session:
        scans = session.declare_subscriber(f"{URID}/c3/battery_state", listener)
        print("Zenoh is open")
        while True:
            print("Waiting for battery messages")
            time.sleep(1)
