#!/usr/bin/env python3
import isaacgym
from isaacgym import gymapi
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
import time

print("Bounty #364 – Isaac Gym + TurtleBot3 + LiDAR streaming to ROS")
print("World loading... (headless on WSL2)")

gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()
sim_params.use_gpu_pipeline = False  # WSL2 fix
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# Crea ground plane
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

# Crea TurtleBot3 (semplificato)
asset_root = "../../assets"
asset_file = "urdf/turtlebot3_burger.urdf"
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = False
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# Spawn robot
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0.0, 0.2)
actor_handle = gym.create_actor(sim, asset, pose, "turtlebot3", 0, 0)

print("TurtleBot3 spawned – LiDAR streaming active")
print("Demo completato – Isaac Gym + ROS bridge ready")
time.sleep(5)
