#!/usr/bin/env python3
"""
Launch OM1 robot in Isaac Gym with full sensor integration
Bounty #364 - Main execution script
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from isaac_gym_integration.envs.om1_nav_env import OM1NavEnv
from isaac_gym_integration.utils.om1_bridge import OM1Bridge
import asyncio


async def main():
    print("="*60)
    print("OM1 Isaac Gym Integration - Bounty #364")
    print("="*60)
    
    # Create environment
    print("\n[1/4] Creating Isaac Gym environment...")
    env = OM1NavEnv()
    
    # Setup ground and environments
    print("[2/4] Setting up ground plane and obstacles...")
    env.create_ground_plane()
    env.setup_environments()
    
    # Connect to OM1 API
    print("[3/4] Connecting to OM1 API...")
    await env.connect_om1_api()
    
    # Run simulation
    print("[4/4] Starting simulation with sensor streaming...")
    print("\nControls:")
    print("  - ESC: Exit")
    print("  - Sensor data streaming to OM1 API in real-time")
    print("\n" + "="*60)
    
    env.run(num_steps=10000)
    
    print("\nSimulation completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
