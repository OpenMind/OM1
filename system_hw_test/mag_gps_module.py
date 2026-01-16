"""
GPS Module for System Hardware Test
"""

import argparse
import time

import serial

parser = argparse.ArgumentParser()
parser.add_argument(
    "--serial", help="serial port to use, when using the low level driver", type=str
)
print(parser.format_help())

args = parser.parse_args()

if args.serial is None:
    print("No serial port provided. Exiting.")
    exit(1)

baudrate = 115200
timeout = 1

serial_connection = serial.Serial(args.serial, baudrate, timeout=timeout)

while True:
    data = serial_connection.readline().decode("utf-8").strip()
    try:
        if data.startswith("HDG (DEG):"):
            parts = data.split()
            if len(parts) >= 4:
                # that's a HDG packet
                yaw_mag_0_360 = float(parts[2])
                yaw_mag_cardinal = parts[3]
                print(f"MAG: {yaw_mag_0_360}")
            else:
                print(f"Unable to parse heading: {data}")
        elif data.startswith("YPR:"):
            yaw, pitch, roll = map(str.strip, data[4:].split(","))
            print(f"Orientation is Yaw: {yaw}°, Pitch: {pitch}°, Roll: {roll}°.")
        elif data.startswith("GPS:"):
            try:
                parts = data[4:].split(",")
                lat = parts[0]
                lon = parts[1]
                heading = parts[3].split(":")[1]
                alt = parts[4].split(":")[1]
                sat = parts[5].split(":")[1]
                alt = float(alt)
                sat = int(sat)
                if len(parts) >= 6:
                    time = parts[6][5:]
                    time_utc = time
                print(
                    (
                        f"Current location is {lat}, {lon} at {alt}m altitude. "
                        f"GPS Heading {heading}° with {sat} satellites locked. "
                        f"The time, if available, is {time_utc}."
                    )
                )
            except Exception as e:
                print(f"Failed to parse GPS: {data} ({e})")
    except Exception as e:
        print(f"Error processing serial MAG/GPS input: {data} ({e})")
