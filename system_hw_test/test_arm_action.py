"""
Test script for G1 arm actions via ros2 topic pub.
Verifies that arm action commands reach g1_arm_action_node.

Usage:
    python3 test_arm_action.py shake_hand
    python3 test_arm_action.py face_wave
    python3 test_arm_action.py hands_up
    python3 test_arm_action.py stand_still
    python3 test_arm_action.py show_hand
"""

import asyncio
import sys

ROS2_TOPIC = "/api/sport/request"
ROS2_MSG_TYPE = "unitree_api/msg/Request"

CUSTOM_API_ID = 9001
BUILTIN_API_ID = 7106

CUSTOM_ACTIONS = {
    "shake_hand",
    "face_wave",
    "hands_up",
    "stand_still",
    "show_hand",
    "wave",
    "move",
    "show_hand1",
    "show_hand2",
    "my_gesture",
}

BUILTIN_ACTIONS = {
    "left_kiss": 12,
    "right_kiss": 13,
    "clap": 17,
    "high_five": 18,
    "heart": 20,
    "high_wave": 26,
}


async def publish_custom_action(action_name: str) -> None:
    # Match the exact format that works manually:
    # ros2 topic pub --once /api/sport/request unitree_api/msg/Request \
    #   "{header: {identity: {api_id: 9001}}, parameter: '{\"action\": \"shake_hand\"}'}"
    msg = (
        "{header: {identity: {api_id: " + str(CUSTOM_API_ID) + "}}, "
        'parameter: \'{"action": "' + action_name + "\"}'}"
    )

    cmd = ["ros2", "topic", "pub", "--once", ROS2_TOPIC, ROS2_MSG_TYPE, msg]
    print(f"Running: {' '.join(cmd)}")

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    print(f"stdout: {stdout.decode()}")
    if stderr:
        print(f"stderr: {stderr.decode()}")
    print(f"Return code: {proc.returncode}")


async def publish_builtin_action(action_name: str, action_id: int) -> None:
    msg = (
        "{header: {identity: {api_id: " + str(BUILTIN_API_ID) + "}}, " 'parameter: \'{"data": ' + str(action_id) + "}'}"
    )

    cmd = ["ros2", "topic", "pub", "--once", ROS2_TOPIC, ROS2_MSG_TYPE, msg]
    print(f"Running: {' '.join(cmd)}")

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    print(f"stdout: {stdout.decode()}")
    if stderr:
        print(f"stderr: {stderr.decode()}")
    print(f"Return code: {proc.returncode}")


async def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python3 test_arm_action.py <action_name>")
        print(f"\nCustom actions: {sorted(CUSTOM_ACTIONS)}")
        print(f"Built-in actions: {sorted(BUILTIN_ACTIONS.keys())}")
        sys.exit(1)

    action = sys.argv[1]

    if action in CUSTOM_ACTIONS:
        print(f"Sending CUSTOM action: {action} (api_id={CUSTOM_API_ID})")
        await publish_custom_action(action)
    elif action in BUILTIN_ACTIONS:
        action_id = BUILTIN_ACTIONS[action]
        print(f"Sending BUILT-IN action: {action} (api_id={BUILTIN_API_ID}, data={action_id})")
        await publish_builtin_action(action, action_id)
    else:
        print(f"Unknown action: {action}")
        print(f"\nCustom actions: {sorted(CUSTOM_ACTIONS)}")
        print(f"Built-in actions: {sorted(BUILTIN_ACTIONS.keys())}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
