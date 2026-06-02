"""
Test script for Booster robot movement commands.
This script sends direct commands to the Booster robot via Zenoh.
"""

import asyncio
import os
import sys

# Add src directory to path to import local zenoh_msgs
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from zenoh_msgs import RemoteControllerState, open_zenoh_session


async def send_command(session, lx: float, ly: float, rx: float, duration: float = 1.0):
    """
    Send a movement command to the Booster robot.

    Parameters
    ----------
    session : zenoh.Session
        The Zenoh session
    lx : float
        Lateral movement (-1 to 1, positive is right)
    ly : float
        Forward/backward movement (-1 to 1, negative is forward)
    rx : float
        Rotation (-1 to 1)
    duration : float
        How long to send the command (seconds)
    """
    msg = RemoteControllerState(
        event=1536,
        lx=lx,
        ly=ly,
        rx=rx,
        ry=0.0,
    )
    session.put("remote_controller_state", msg.serialize())
    print(f"Sent command: lx={lx}, ly={ly}, rx={rx}")
    await asyncio.sleep(duration)


async def stop_robot(session):
    """Send stop command to the robot."""
    msg = RemoteControllerState(event=1536)
    session.put("remote_controller_state", msg.serialize())
    print("Sent STOP command")


async def test_forward(session):
    """Test moving forward."""
    print("\n=== Testing: Move Forward ===")
    await send_command(session, lx=0.0, ly=-0.2, rx=0.0, duration=2.0)
    await stop_robot(session)
    await asyncio.sleep(1)


async def test_backward(session):
    """Test moving backward."""
    print("\n=== Testing: Move Backward ===")
    await send_command(session, lx=0.0, ly=0.2, rx=0.0, duration=2.0)
    await stop_robot(session)
    await asyncio.sleep(1)


async def test_turn_left(session):
    """Test turning left."""
    print("\n=== Testing: Turn Left ===")
    await send_command(session, lx=0.0, ly=0.0, rx=0.3, duration=2.0)
    await stop_robot(session)
    await asyncio.sleep(1)


async def test_turn_right(session):
    """Test turning right."""
    print("\n=== Testing: Turn Right ===")
    await send_command(session, lx=0.0, ly=0.0, rx=-0.3, duration=2.0)
    await stop_robot(session)
    await asyncio.sleep(1)


async def test_strafe_left(session):
    """Test strafing left."""
    print("\n=== Testing: Strafe Left ===")
    await send_command(session, lx=-0.2, ly=0.0, rx=0.0, duration=2.0)
    await stop_robot(session)
    await asyncio.sleep(1)


async def test_strafe_right(session):
    """Test strafing right."""
    print("\n=== Testing: Strafe Right ===")
    await send_command(session, lx=0.2, ly=0.0, rx=0.0, duration=2.0)
    await stop_robot(session)
    await asyncio.sleep(1)


async def run_all_tests(session):
    """Run all movement tests."""
    print("\n" + "=" * 50)
    print("Starting Booster Robot Movement Tests")
    print("=" * 50)

    await test_forward(session)
    await test_backward(session)
    await test_turn_left(session)
    await test_turn_right(session)
    await test_strafe_left(session)
    await test_strafe_right(session)

    print("\n" + "=" * 50)
    print("All tests completed!")
    print("=" * 50)


async def interactive_mode(session):
    """Interactive mode for manual control."""
    print("\n" + "=" * 50)
    print("Interactive Control Mode")
    print("=" * 50)
    print("Commands:")
    print("  w - Move forward")
    print("  s - Move backward")
    print("  a - Turn left")
    print("  d - Turn right")
    print("  q - Strafe left")
    print("  e - Strafe right")
    print("  x - Stop")
    print("  exit - Quit")
    print("=" * 50 + "\n")

    while True:
        try:
            cmd = input("Enter command: ").strip().lower()

            if cmd == "exit":
                await stop_robot(session)
                break
            elif cmd == "w":
                await send_command(session, 0.0, -0.2, 0.0, 0.5)
            elif cmd == "s":
                await send_command(session, 0.0, 0.2, 0.0, 0.5)
            elif cmd == "a":
                await send_command(session, 0.0, 0.0, 0.3, 0.5)
            elif cmd == "d":
                await send_command(session, 0.0, 0.0, -0.3, 0.5)
            elif cmd == "q":
                await send_command(session, -0.2, 0.0, 0.0, 0.5)
            elif cmd == "e":
                await send_command(session, 0.2, 0.0, 0.0, 0.5)
            elif cmd == "x":
                await stop_robot(session)
            else:
                print("Unknown command!")

        except KeyboardInterrupt:
            print("\nStopping robot...")
            await stop_robot(session)
            break
        except Exception as e:
            print(f"Error: {e}")


async def main():
    """Main function."""
    print("Connecting to Zenoh...")
    session = open_zenoh_session()
    print(f"Connected to Zenoh: {session}\n")

    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        await interactive_mode(session)
    else:
        print("Running automated tests...")
        print("(Use 'python booster_move.py interactive' for manual control)")
        await run_all_tests(session)

    print("\nTest complete!")


if __name__ == "__main__":
    asyncio.run(main())
