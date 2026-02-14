"""
Test script for Booster robot movement commands via Zenoh to ROS2 service.
This script sends movement commands through Zenoh bridge to /booster_rpc_service.

The Zenoh bridge translates Zenoh messages to ROS2 service calls.
Service topic pattern: /booster_rpc_service (ROS2) -> rt/booster_rpc_service (Zenoh)
"""

import asyncio
import json
import os
import sys

# Add src directory to path to import local zenoh_msgs
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from zenoh_msgs import BoosterApiReqMsg, open_zenoh_session


class BoosterMoveZenohClient:
    """Zenoh client for testing Booster robot movement via RPC service."""

    # API ID for movement commands (kMove)
    API_MOVE = 2001

    def __init__(self):
        """Initialize the Zenoh client."""
        print("Connecting to Zenoh...")
        self.session = open_zenoh_session()
        print(f"Connected to Zenoh: {self.session}\n")

        # Zenoh service call topic (maps to ROS2 /booster_rpc_service)
        # The zenoh-bridge-ros2dds translates rt/service_name to /service_name
        self.service_topic = "rt/booster_rpc_service"

    def create_service_request(self, vx: float, vy: float, vyaw: float) -> bytes:
        """
        Create a service request payload for movement command.

        Parameters
        ----------
        vx : float
            Linear velocity in x direction (m/s), positive is forward
        vy : float
            Linear velocity in y direction (m/s), positive is left
        vyaw : float
            Angular velocity (rad/s), positive is counter-clockwise

        Returns
        -------
        bytes
            Serialized service request payload (CDR format)
        """
        # Create the request message using BoosterApiReqMsg
        request = BoosterApiReqMsg(
            api_id=self.API_MOVE, body=json.dumps({"vx": vx, "vy": vy, "vyaw": vyaw})
        )

        # Serialize to CDR format for Zenoh bridge
        return request.serialize()

    async def send_move_command(
        self, vx: float, vy: float, vyaw: float, duration: float = 1.0
    ):
        """
        Send a movement command to the robot via Zenoh service.

        Parameters
        ----------
        vx : float
            Linear velocity in x direction (m/s)
        vy : float
            Linear velocity in y direction (m/s)
        vyaw : float
            Angular velocity (rad/s)
        duration : float
            How long to execute the command (seconds)
        """
        print(f"Sending command: vx={vx}, vy={vy}, vyaw={vyaw}")

        # Create and send the service request
        request_payload = self.create_service_request(vx, vy, vyaw)

        # Use Zenoh query for service call (request/response pattern)
        # The bridge will translate this to a ROS2 service call
        try:
            replies = self.session.get(
                self.service_topic,
                payload=request_payload,
                timeout=5.0,  # 5 second timeout
            )

            # Process the reply
            for reply in replies:
                if reply.ok:
                    response = json.loads(reply.ok.payload.to_bytes().decode("utf-8"))
                    print(f"Service response: {response}")
                else:
                    print(f"Service error: {reply.err}")

        except Exception as e:
            print(f"Error calling service: {e}")

        await asyncio.sleep(duration)

    async def stop_robot(self):
        """Send stop command to the robot."""
        print("Sending STOP command")
        request_payload = self.create_service_request(0.0, 0.0, 0.0)

        try:
            replies = self.session.get(
                self.service_topic, payload=request_payload, timeout=5.0
            )

            for reply in replies:
                if reply.ok:
                    response = json.loads(reply.ok.payload.to_bytes().decode("utf-8"))
                    print(f"Stop response: {response}")

        except Exception as e:
            print(f"Error stopping robot: {e}")

    async def test_forward(self):
        """Test moving forward."""
        print("\n=== Testing: Move Forward ===")
        await self.send_move_command(vx=0.2, vy=0.0, vyaw=0.0, duration=2.0)
        await self.stop_robot()
        await asyncio.sleep(1)

    async def test_backward(self):
        """Test moving backward."""
        print("\n=== Testing: Move Backward ===")
        await self.send_move_command(vx=-0.2, vy=0.0, vyaw=0.0, duration=2.0)
        await self.stop_robot()
        await asyncio.sleep(1)

    async def test_turn_left(self):
        """Test turning left (counter-clockwise)."""
        print("\n=== Testing: Turn Left ===")
        await self.send_move_command(vx=0.0, vy=0.0, vyaw=0.3, duration=2.0)
        await self.stop_robot()
        await asyncio.sleep(1)

    async def test_turn_right(self):
        """Test turning right (clockwise)."""
        print("\n=== Testing: Turn Right ===")
        await self.send_move_command(vx=0.0, vy=0.0, vyaw=-0.3, duration=2.0)
        await self.stop_robot()
        await asyncio.sleep(1)

    async def test_strafe_left(self):
        """Test strafing left."""
        print("\n=== Testing: Strafe Left ===")
        await self.send_move_command(vx=0.0, vy=0.2, vyaw=0.0, duration=2.0)
        await self.stop_robot()
        await asyncio.sleep(1)

    async def test_strafe_right(self):
        """Test strafing right."""
        print("\n=== Testing: Strafe Right ===")
        await self.send_move_command(vx=0.0, vy=-0.2, vyaw=0.0, duration=2.0)
        await self.stop_robot()
        await asyncio.sleep(1)

    async def run_all_tests(self):
        """Run all movement tests."""
        print("\n" + "=" * 50)
        print("Starting Booster Robot Movement Tests (Zenoh Service)")
        print("=" * 50)

        await self.test_forward()
        await self.test_backward()
        await self.test_turn_left()
        await self.test_turn_right()
        await self.test_strafe_left()
        await self.test_strafe_right()

        print("\n" + "=" * 50)
        print("All tests completed!")
        print("=" * 50)

    async def interactive_mode(self):
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
                    await self.stop_robot()
                    break
                elif cmd == "w":
                    await self.send_move_command(0.2, 0.0, 0.0, 0.5)
                elif cmd == "s":
                    await self.send_move_command(-0.2, 0.0, 0.0, 0.5)
                elif cmd == "a":
                    await self.send_move_command(0.0, 0.0, 0.3, 0.5)
                elif cmd == "d":
                    await self.send_move_command(0.0, 0.0, -0.3, 0.5)
                elif cmd == "q":
                    await self.send_move_command(0.0, 0.2, 0.0, 0.5)
                elif cmd == "e":
                    await self.send_move_command(0.0, -0.2, 0.0, 0.5)
                elif cmd == "x":
                    await self.stop_robot()
                else:
                    print("Unknown command!")

            except KeyboardInterrupt:
                print("\nStopping robot...")
                await self.stop_robot()
                break
            except Exception as e:
                print(f"Error: {e}")


async def main():
    """Main function."""
    client = BoosterMoveZenohClient()

    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        await client.interactive_mode()
    else:
        print("Running automated tests...")
        print(
            "(Use 'python test_booster_move_zenoh_service.py interactive' for manual control)"
        )
        await client.run_all_tests()

    print("\nTest complete!")


if __name__ == "__main__":
    asyncio.run(main())
