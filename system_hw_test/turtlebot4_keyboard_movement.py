import argparse
import logging
import sys
import threading
import time
from dataclasses import dataclass

from pycdr2 import IdlStruct
from pycdr2.types import float64
from pynput import keyboard

sys.path.insert(0, "../src")
from zenoh_msgs import open_zenoh_session

parser = argparse.ArgumentParser()
parser.add_argument("--URID", help="your robot's URID, when using Zenoh", type=str)
print(parser.format_help())

args = parser.parse_args()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Global variables
exit_program = False
current_linear = 0.0
current_angular = 0.0
key_pressed = set()


@dataclass
class Vector3(IdlStruct, typename="Vector3"):
    x: float64
    y: float64
    z: float64


@dataclass
class Twist(IdlStruct, typename="Twist"):
    linear: Vector3
    angular: Vector3


class MoveController:

    def __init__(self, URID: str = ""):
        self.session = None
        self.cmd_vel = f"{URID}/c3/cmd_vel"
        try:
            self.session = open_zenoh_session()
            logging.info("Zenoh client opened")
        except Exception as e:
            logging.error(f"Error opening Zenoh client: {e}")

    def pub_twist(self, linear, angular):
        if self.session is None:
            logging.info("No open Zenoh session, returning")
            return

        logging.debug(f"Publishing twist: linear={linear}, angular={angular}")
        try:
            t = Twist(
                linear=Vector3(x=float(linear), y=0.0, z=0.0),
                angular=Vector3(x=0.0, y=0.0, z=float(angular)),
            )
            serialized_data = t.serialize()
            self.session.put(self.cmd_vel, serialized_data)
        except Exception as e:
            logging.error(f"Error publishing twist: {e}")


def on_press(key, move_controller):
    global key_pressed

    try:
        k = key.char
    except Exception:
        k = key.name

    key_pressed.add(k)
    update_movement(move_controller)


def on_release(key, move_controller):
    global key_pressed

    try:
        k = key.char
    except Exception:
        k = key.name

    if k in key_pressed:
        key_pressed.remove(k)

    update_movement(move_controller)

    if k == "esc":
        global exit_program
        exit_program = True
        return False


def update_movement(move_controller):
    global key_pressed, current_linear, current_angular, last_print_time

    new_linear = 0.0
    new_angular = 0.0

    if "w" in key_pressed:
        new_linear = 0.5
    if "s" in key_pressed:
        new_linear = -0.5
    if "a" in key_pressed:
        new_angular = 0.5
    if "d" in key_pressed:
        new_angular = -0.5

    current_linear = new_linear
    current_angular = new_angular

    move_controller.pub_twist(current_linear, current_angular)

    if current_linear > 0:
        print("Moving forward")
    elif current_linear < 0:
        print("Moving backward")

    if current_angular > 0:
        print("Turning left")
    elif current_angular < 0:
        print("Turning right")

    if current_linear == 0 and current_angular == 0:
        print("Stopped")


def continuous_publish_thread(move_controller):
    """Thread function to continuously publish movement commands."""
    publish_rate = 10  # Hz (10 times per second)
    sleep_time = 1.0 / publish_rate

    while not exit_program:
        if key_pressed:
            move_controller.pub_twist(current_linear, current_angular)
        time.sleep(sleep_time)


def control_loop(move_controller):
    listener = keyboard.Listener(
        on_press=lambda key: on_press(key, move_controller),
        on_release=lambda key: on_release(key, move_controller),
    )
    listener.start()

    print("Keyboard Control Active:")
    print(" W - Move Forward")
    print(" S - Move Backward")
    print(" A - Turn Left")
    print(" D - Turn Right")
    print(" ESC - Quit")

    publisher_thread = threading.Thread(
        target=continuous_publish_thread, args=(move_controller,)
    )
    publisher_thread.daemon = True
    publisher_thread.start()

    while not exit_program:
        time.sleep(0.1)


if __name__ == "__main__":
    try:

        URID = args.URID
        print(f"Using Zenoh to connect to robot using {URID}")
        print("[INFO] Opening zenoh session...")

        move_controller = MoveController(URID)

        control_thread = threading.Thread(target=control_loop, args=(move_controller,))
        control_thread.daemon = True
        control_thread.start()

        while not exit_program:
            time.sleep(0.5)

    except KeyboardInterrupt:
        logging.info("Program interrupted")
    finally:
        logging.info("Exiting program")
        exit_program = True
        time.sleep(0.5)
        sys.exit(0)
