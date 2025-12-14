#!/usr/bin/env python3
"""
Example OM1 agent controlling the Three.js simulator.

Usage:
    1. Start bridge: cd bridge && npm start
    2. Start client: cd client && npm run dev
    3. Run this script: python example_agent.py
"""
import sys
import time

import requests

BRIDGE_URL = "http://localhost:8081"


def test_connection():
    """Test if bridge is running."""
    try:
        resp = requests.post(f"{BRIDGE_URL}/reset", timeout=2)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False


def simple_forward_agent():
    """Simple agent that moves forward until collision or done."""
    print("🤖 Starting simple forward agent...")

    # Reset environment
    requests.post(f"{BRIDGE_URL}/reset")
    print("✅ Environment reset")

    step = 0
    total_reward = 0.0

    try:
        while True:
            # Send forward action
            resp = requests.post(
                f"{BRIDGE_URL}/action", json={"v": 0.05, "w": 0.0}
            ).json()

            step += 1
            total_reward += resp["reward"]
            min_dist = resp["info"]["minDist"]
            collisions = resp["info"]["collisions"]

            print(
                f"Step {step:3d}: reward={resp['reward']:+.3f}, "
                f"minDist={min_dist:.2f}m, collisions={collisions}"
            )

            if resp["done"]:
                print("\n🏁 Episode done!")
                print(f"   Total steps: {step}")
                print(f"   Total reward: {total_reward:.3f}")
                print(f"   Collisions: {collisions}")
                break

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\n⏹️  Interrupted by user")


def obstacle_avoidance_agent():
    """Agent that tries to avoid obstacles using sensor data."""
    print("🤖 Starting obstacle avoidance agent...")

    # Reset environment
    requests.post(f"{BRIDGE_URL}/reset")
    print("✅ Environment reset")

    step = 0
    total_reward = 0.0

    try:
        while True:
            # Send forward action
            resp = requests.post(
                f"{BRIDGE_URL}/action", json={"v": 0.05, "w": 0.0}
            ).json()

            step += 1
            total_reward += resp["reward"]
            min_dist = resp["info"]["minDist"]
            collisions = resp["info"]["collisions"]
            distances = resp["sensors"]["distances"]

            # Simple avoidance logic: turn if too close
            if min_dist < 0.5:
                # Turn away from obstacle
                # Find direction with most space
                left_avg = sum(distances[:6]) / 6
                right_avg = sum(distances[7:]) / 6

                turn_direction = 0.15 if left_avg > right_avg else -0.15

                print(f"Step {step:3d}: ⚠️  Obstacle close! Turning...")
                resp = requests.post(
                    f"{BRIDGE_URL}/action", json={"v": 0.02, "w": turn_direction}
                ).json()
            else:
                print(
                    f"Step {step:3d}: reward={resp['reward']:+.3f}, "
                    f"minDist={min_dist:.2f}m"
                )

            if resp["done"]:
                print("\n🏁 Episode done!")
                print(f"   Total steps: {step}")
                print(f"   Total reward: {total_reward:.3f}")
                print(f"   Collisions: {collisions}")
                break

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\n⏹️  Interrupted by user")


def main():
    print("=" * 50)
    print("OM1 × Three.js Simulator - Example Agent")
    print("=" * 50)
    print()

    # Check connection
    print("🔍 Checking bridge connection...")
    if not test_connection():
        print("❌ Error: Cannot connect to bridge at", BRIDGE_URL)
        print("   Make sure bridge is running: cd bridge && npm start")
        sys.exit(1)

    print("✅ Bridge connected!")
    print()

    # Choose agent
    print("Select agent:")
    print("  1. Simple forward agent")
    print("  2. Obstacle avoidance agent")
    print()

    choice = input("Enter choice (1 or 2): ").strip()
    print()

    if choice == "1":
        simple_forward_agent()
    elif choice == "2":
        obstacle_avoidance_agent()
    else:
        print("Invalid choice!")
        sys.exit(1)


if __name__ == "__main__":
    main()
