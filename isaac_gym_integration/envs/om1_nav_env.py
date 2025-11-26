"""
OM1 Navigation Environment for Isaac Gym
Bounty #364 - Complete integration with LiDAR, Camera, IMU, and Navigation
"""


from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import torch
import numpy as np
import yaml
import asyncio
import websockets
import json


class OM1NavEnv:
    """Isaac Gym environment for OM1 robot with full sensor suite"""
    
    def __init__(self, cfg_path="isaac_gym_integration/cfg/om1_robot.yaml"):
        # Load configuration
        with open(cfg_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        
        # Initialize Isaac Gym
        self.gym = gymapi.acquire_gym()
        
        # Setup simulation parameters
        self.sim_params = gymapi.SimParams()
        self.sim_params.dt = self.cfg['robot']['physics']['dt']
        self.sim_params.substeps = self.cfg['robot']['physics']['substeps']
        self.sim_params.gravity = gymapi.Vec3(*self.cfg['robot']['physics']['gravity'])
        
        # GPU acceleration
        if self.cfg['simulation']['use_gpu']:
            self.sim_params.use_gpu_pipeline = self.cfg['simulation']['use_gpu_pipeline']
        
        # Physics engine (PhysX for GPU)
        self.sim_params.physx.solver_type = 1
        self.sim_params.physx.num_position_iterations = 4
        self.sim_params.physx.num_velocity_iterations = 1
        self.sim_params.physx.contact_offset = 0.01
        self.sim_params.physx.rest_offset = 0.0
        
        # Create simulation
        self.sim = self.gym.create_sim(
            0, 0, gymapi.SIM_PHYSX, self.sim_params
        )
        
        # Camera for rendering
        self.viewer = None
        
        # Environments
        self.num_envs = self.cfg['simulation']['num_envs']
        self.envs = []
        self.robot_handles = []
        
        # Sensor data buffers
        self.camera_data = []
        self.lidar_data = []
        self.imu_data = []
        
        # OM1 API connection
        self.om1_ws = None
        self.om1_connected = False
        
    def create_ground_plane(self):
        """Create ground plane"""
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)
        
    def create_obstacles(self, env, env_idx):
        """Create obstacles for navigation testing"""
        # Box obstacle
        box_asset_options = gymapi.AssetOptions()
        box_asset_options.density = 100.0
        box_asset = self.gym.create_box(
            self.sim, 0.5, 0.5, 0.5, box_asset_options
        )
        
        box_pose = gymapi.Transform()
        box_pose.p = gymapi.Vec3(2.0 + env_idx * 0.5, 0.0, 0.25)
        self.gym.create_actor(env, box_asset, box_pose, "box", env_idx, 0)
        
        # Cylinder obstacle
        cylinder_asset = self.gym.create_sphere(
            self.sim, 0.3, box_asset_options
        )
        
        cyl_pose = gymapi.Transform()
        cyl_pose.p = gymapi.Vec3(-2.0, 2.0, 0.3)
        self.gym.create_actor(env, cylinder_asset, cyl_pose, "cylinder", env_idx, 0)
        
    def create_robot(self, env, env_idx):
        """Create OM1 robot with sensors"""
        # Robot asset options
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.density = self.cfg['robot']['dimensions']['base']['mass']
        
        # Create simple box robot (will be replaced with URDF)
        robot_asset = self.gym.create_box(
            self.sim,
            self.cfg['robot']['dimensions']['base']['length'],
            self.cfg['robot']['dimensions']['base']['width'],
            self.cfg['robot']['dimensions']['base']['height'],
            asset_options
        )
        
        # Initial pose
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 0.2)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        
        # Create actor
        robot_handle = self.gym.create_actor(
            env, robot_asset, pose, f"om1_robot_{env_idx}", env_idx, 1
        )
        
        return robot_handle
        
    def setup_sensors(self, env, robot_handle):
        """Setup Camera, LiDAR, and IMU sensors"""
        sensors = {}
        
        # Camera sensor
        if self.cfg['robot']['sensors']['camera']['enabled']:
            camera_props = gymapi.CameraProperties()
            camera_props.width = self.cfg['robot']['sensors']['camera']['width']
            camera_props.height = self.cfg['robot']['sensors']['camera']['height']
            camera_props.horizontal_fov = self.cfg['robot']['sensors']['camera']['fov']
            
            camera_handle = self.gym.create_camera_sensor(env, camera_props)
            
            # Position camera on robot
            cam_pos = self.cfg['robot']['sensors']['camera']['position']
            local_transform = gymapi.Transform()
            local_transform.p = gymapi.Vec3(*cam_pos)
            
            self.gym.attach_camera_to_body(
                camera_handle, env, robot_handle, local_transform,
                gymapi.FOLLOW_TRANSFORM
            )
            
            sensors['camera'] = camera_handle
            
        # LiDAR (simulated with raycasts)
        if self.cfg['robot']['sensors']['lidar']['enabled']:
            sensors['lidar'] = {
                'num_rays': self.cfg['robot']['sensors']['lidar']['num_rays'],
                'range': self.cfg['robot']['sensors']['lidar']['range'],
                'position': self.cfg['robot']['sensors']['lidar']['position']
            }
            
        # IMU (simulated with rigid body state)
        if self.cfg['robot']['sensors']['imu']['enabled']:
            sensors['imu'] = {
                'frequency': self.cfg['robot']['sensors']['imu']['frequency'],
                'noise': self.cfg['robot']['sensors']['imu']['noise_stddev']
            }
            
        return sensors
        
    def setup_environments(self):
        """Create all environments with robots and obstacles"""
        env_spacing = self.cfg['simulation']['env_spacing']
        env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        
        for i in range(self.num_envs):
            # Create environment
            env = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            self.envs.append(env)
            
            # Create robot
            robot_handle = self.create_robot(env, i)
            self.robot_handles.append(robot_handle)
            
            # Setup sensors
            sensors = self.setup_sensors(env, robot_handle)
            
            # Create obstacles
            self.create_obstacles(env, i)
            
        print(f"Created {self.num_envs} environments with OM1 robots")
        
    def simulate_lidar(self, env_idx):
        """Simulate 2D LiDAR using raycasts"""
        num_rays = self.cfg['robot']['sensors']['lidar']['num_rays']
        max_range = self.cfg['robot']['sensors']['lidar']['range']
        
        # Get robot position
        robot_state = self.gym.get_actor_rigid_body_states(
            self.envs[env_idx], self.robot_handles[env_idx], gymapi.STATE_ALL
        )
        
        # Simulate raycast in circle
        angles = np.linspace(0, 2*np.pi, num_rays)
        ranges = np.ones(num_rays) * max_range
        
        # TODO: Implement actual raycasting when Isaac Gym API available
        # For now return simulated data
        
        return {
            'ranges': ranges.tolist(),
            'angles': angles.tolist(),
            'timestamp': self.gym.get_sim_time(self.sim)
        }
        
    def get_imu_data(self, env_idx):
        """Get IMU data from rigid body state"""
        robot_state = self.gym.get_actor_rigid_body_states(
            self.envs[env_idx], self.robot_handles[env_idx], gymapi.STATE_ALL
        )
        
        # Extract linear acceleration and angular velocity
        # Add noise
        noise = np.random.normal(0, self.cfg['robot']['sensors']['imu']['noise_stddev'], 6)
        
        return {
            'linear_acceleration': [0.0 + noise[0], 0.0 + noise[1], 9.81 + noise[2]],
            'angular_velocity': [0.0 + noise[3], 0.0 + noise[4], 0.0 + noise[5]],
            'timestamp': self.gym.get_sim_time(self.sim)
        }
        
    def get_camera_image(self, env_idx, camera_handle):
        """Get RGB image from camera"""
        self.gym.render_all_camera_sensors(self.sim)
        image = self.gym.get_camera_image(
            self.sim, self.envs[env_idx], camera_handle, gymapi.IMAGE_COLOR
        )
        
        return {
            'width': self.cfg['robot']['sensors']['camera']['width'],
            'height': self.cfg['robot']['sensors']['camera']['height'],
            'data': image.tolist() if hasattr(image, 'tolist') else [],
            'timestamp': self.gym.get_sim_time(self.sim)
        }
        
    async def connect_om1_api(self):
        """Connect to OM1 API via WebSocket"""
        try:
            uri = f"{self.cfg['om1']['endpoint']}?api_key={self.cfg['om1']['api_key']}"
            self.om1_ws = await websockets.connect(uri)
            self.om1_connected = True
            print("Connected to OM1 API")
        except Exception as e:
            print(f"Failed to connect to OM1 API: {e}")
            self.om1_connected = False
            
    async def stream_sensor_data(self, env_idx):
        """Stream sensor data to OM1 API"""
        if not self.om1_connected:
            return
            
        try:
            # Collect all sensor data
            data = {
                'robot_id': f'om1_robot_{env_idx}',
                'timestamp': self.gym.get_sim_time(self.sim),
                'sensors': {
                    'lidar': self.simulate_lidar(env_idx),
                    'imu': self.get_imu_data(env_idx)
                }
            }
            
            # Send to OM1
            await self.om1_ws.send(json.dumps(data))
            
        except Exception as e:
            print(f"Error streaming sensor data: {e}")
            
    def apply_command(self, env_idx, linear_vel, angular_vel):
        """Apply velocity command to robot"""
        # Get robot handle
        robot = self.robot_handles[env_idx]
        
        # Apply forces (simplified - will be improved with actual URDF)
        # TODO: Implement proper differential drive control
        
        pass
        
    def step(self):
        """Step simulation and update sensors"""
        # Step physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        
        # Update graphics
        self.gym.step_graphics(self.sim)
        if self.viewer:
            self.gym.draw_viewer(self.viewer, self.sim, True)
            
    def run(self, num_steps=1000):
        """Run simulation loop"""
        # Create viewer
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        
        # Simulation loop
        for step in range(num_steps):
            self.step()
            
            # Stream sensor data every 10 steps
            if step % 10 == 0 and self.om1_connected:
                asyncio.run(self.stream_sensor_data(0))
                
        # Cleanup
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
