"""
OM1 API Bridge for Isaac Gym
Handles bidirectional communication between Isaac Gym and OM1 API
Bounty #364
"""

import asyncio
import websockets
import json
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OM1Bridge:
    """WebSocket bridge for OM1 API integration"""
    
    def __init__(self, api_key, endpoint="wss://api.openmind.org"):
        self.api_key = api_key
        self.endpoint = endpoint
        self.ws = None
        self.connected = False
        self.command_queue = asyncio.Queue()
        
    async def connect(self):
        """Establish WebSocket connection to OM1 API"""
        try:
            uri = f"{self.endpoint}?api_key={self.api_key}"
            self.ws = await websockets.connect(
                uri,
                ping_interval=20,
                ping_timeout=10
            )
            self.connected = True
            logger.info(f"✓ Connected to OM1 API at {self.endpoint}")
            
            # Start command receiver
            asyncio.create_task(self._receive_commands())
            
        except Exception as e:
            logger.error(f"✗ Failed to connect to OM1 API: {e}")
            self.connected = False
            
    async def disconnect(self):
        """Close WebSocket connection"""
        if self.ws:
            await self.ws.close()
            self.connected = False
            logger.info("Disconnected from OM1 API")
            
    async def send_sensor_data(self, robot_id, sensor_data):
        """
        Send sensor data to OM1 API
        
        Args:
            robot_id: Unique robot identifier
            sensor_data: Dict containing camera, lidar, imu data
        """
        if not self.connected:
            logger.warning("Not connected to OM1 API")
            return False
            
        try:
            message = {
                "type": "sensor_data",
                "robot_id": robot_id,
                "timestamp": datetime.utcnow().isoformat(),
                "data": sensor_data
            }
            
            await self.ws.send(json.dumps(message))
            return True
            
        except Exception as e:
            logger.error(f"Error sending sensor data: {e}")
            return False
            
    async def send_camera_frame(self, robot_id, image_data, width, height):
        """
        Send camera frame to OM1 API
        
        Args:
            robot_id: Unique robot identifier
            image_data: RGB image array (flattened)
            width: Image width
            height: Image height
        """
        if not self.connected:
            return False
            
        try:
            # Convert numpy array to base64 or compressed format
            # For now send metadata only (full images are large)
            message = {
                "type": "camera_frame",
                "robot_id": robot_id,
                "timestamp": datetime.utcnow().isoformat(),
                "width": width,
                "height": height,
                "format": "rgb",
                # In production: compress and encode image_data
                "data": "base64_encoded_image_here"
            }
            
            await self.ws.send(json.dumps(message))
            return True
            
        except Exception as e:
            logger.error(f"Error sending camera frame: {e}")
            return False
            
    async def send_lidar_scan(self, robot_id, ranges, angles):
        """
        Send LiDAR scan to OM1 API
        
        Args:
            robot_id: Unique robot identifier
            ranges: Array of distance measurements
            angles: Array of angle measurements
        """
        if not self.connected:
            return False
            
        try:
            message = {
                "type": "lidar_scan",
                "robot_id": robot_id,
                "timestamp": datetime.utcnow().isoformat(),
                "ranges": ranges,
                "angles": angles,
                "num_rays": len(ranges),
                "max_range": max(ranges) if ranges else 0
            }
            
            await self.ws.send(json.dumps(message))
            return True
            
        except Exception as e:
            logger.error(f"Error sending LiDAR scan: {e}")
            return False
            
    async def send_imu_data(self, robot_id, linear_acc, angular_vel):
        """
        Send IMU data to OM1 API
        
        Args:
            robot_id: Unique robot identifier
            linear_acc: [x, y, z] linear acceleration
            angular_vel: [x, y, z] angular velocity
        """
        if not self.connected:
            return False
            
        try:
            message = {
                "type": "imu_data",
                "robot_id": robot_id,
                "timestamp": datetime.utcnow().isoformat(),
                "linear_acceleration": {
                    "x": linear_acc[0],
                    "y": linear_acc[1],
                    "z": linear_acc[2]
                },
                "angular_velocity": {
                    "x": angular_vel[0],
                    "y": angular_vel[1],
                    "z": angular_vel[2]
                }
            }
            
            await self.ws.send(json.dumps(message))
            return True
            
        except Exception as e:
            logger.error(f"Error sending IMU data: {e}")
            return False
            
    async def _receive_commands(self):
        """Receive commands from OM1 API (async background task)"""
        try:
            async for message in self.ws:
                try:
                    data = json.loads(message)
                    
                    if data.get('type') == 'velocity_command':
                        # Extract velocity commands
                        cmd = {
                            'robot_id': data.get('robot_id'),
                            'linear': data.get('linear_velocity', 0.0),
                            'angular': data.get('angular_velocity', 0.0)
                        }
                        await self.command_queue.put(cmd)
                        logger.info(f"Received command: linear={cmd['linear']:.2f}, angular={cmd['angular']:.2f}")
                        
                    elif data.get('type') == 'navigation_goal':
                        # Extract navigation goal
                        cmd = {
                            'robot_id': data.get('robot_id'),
                            'goal_x': data.get('x', 0.0),
                            'goal_y': data.get('y', 0.0),
                            'goal_theta': data.get('theta', 0.0)
                        }
                        await self.command_queue.put(cmd)
                        logger.info(f"Received nav goal: ({cmd['goal_x']:.2f}, {cmd['goal_y']:.2f})")
                        
                except json.JSONDecodeError:
                    logger.error("Invalid JSON received from OM1 API")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Connection to OM1 API closed")
            self.connected = False
            
    async def get_command(self):
        """Get next command from queue (non-blocking)"""
        try:
            return await asyncio.wait_for(self.command_queue.get(), timeout=0.01)
        except asyncio.TimeoutError:
            return None
            
    def has_commands(self):
        """Check if commands are available"""
        return not self.command_queue.empty()


# Example usage
async def main():
    """Test bridge connection"""
    bridge = OM1Bridge(
        api_key="om1_live_482c8015602d9ef9e58c9cdcc9e8d6d6e6b75f7a9c8d635d75e482fa594e4f84be6ea052e2aa6a92"
    )
    
    await bridge.connect()
    
    if bridge.connected:
        # Send test sensor data
        await bridge.send_lidar_scan(
            "test_robot",
            ranges=[1.0, 2.0, 3.0] * 120,  # 360 rays
            angles=list(np.linspace(0, 2*np.pi, 360))
        )
        
        await bridge.send_imu_data(
            "test_robot",
            linear_acc=[0.0, 0.0, 9.81],
            angular_vel=[0.0, 0.0, 0.0]
        )
        
        # Wait for commands
        await asyncio.sleep(5)
        
    await bridge.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
