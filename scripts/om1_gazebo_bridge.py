#!/usr/bin/env python3
"""
OM1 Gazebo Bridge - WebSocket connection to OM1 API
Streams sensor data from Gazebo to OM1 and receives control commands
"""

import asyncio
import websockets
import json
import yaml
from pathlib import Path

class OM1GazeboBridge:
    def __init__(self, config_path="config/om1_config.yaml"):
        self.config = self.load_config(config_path)
        self.ws = None
        
    def load_config(self, path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    async def connect(self):
        """Connect to OM1 API via WebSocket"""
        api_key = self.config['om1']['api_key']
        endpoint = self.config['om1']['endpoint']
        
        headers = {
            'Authorization': f'Bearer {api_key}'
        }
        
        try:
            self.ws = await websockets.connect(endpoint, extra_headers=headers)
            print(f"[OM1 Bridge] Connected to {endpoint}")
            return True
        except Exception as e:
            print(f"[OM1 Bridge] Connection error: {e}")
            return False
    
    async def stream_sensor_data(self, sensor_type, data):
        """Stream sensor data to OM1 API"""
        if not self.ws:
            return
        
        message = {
            'type': 'sensor_data',
            'sensor': sensor_type,
            'timestamp': data.get('timestamp'),
            'data': data
        }
        
        try:
            await self.ws.send(json.dumps(message))
        except Exception as e:
            print(f"[OM1 Bridge] Send error: {e}")
    
    async def receive_commands(self):
        """Receive control commands from OM1 API"""
        if not self.ws:
            return None
        
        try:
            message = await self.ws.recv()
            command = json.loads(message)
            return command
        except Exception as e:
            print(f"[OM1 Bridge] Receive error: {e}")
            return None
    
    async def run(self):
        """Main bridge loop"""
        if await self.connect():
            print("[OM1 Bridge] Bridge running...")
            while True:
                # Receive commands from OM1
                command = await self.receive_commands()
                if command:
                    print(f"[OM1 Bridge] Received: {command}")
                
                await asyncio.sleep(0.1)

if __name__ == "__main__":
    bridge = OM1GazeboBridge()
    asyncio.run(bridge.run())
