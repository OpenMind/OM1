import logging
import os
from typing import Optional
import aiohttp
from dotenv import load_dotenv
from actions.base import ActionConfig, ActionConnector
from actions.iot_control.interface import IoTInput

load_dotenv()


class HomeAssistantConnector(ActionConnector[IoTInput]):
    """
    Home Assistant REST API connector for IoT control.
    
    Works with Home Assistant demo entities - no physical devices required!
    
    Demo Entities:
    - light.bed_light
    - light.ceiling_lights
    - light.kitchen_lights
    - fan.living_room_fan
    
    These are automatically available in Home Assistant demo mode.
    """
    
    def __init__(self, config: ActionConfig):
        super().__init__(config)
        
        # Home Assistant configuration
        self.ha_url = os.getenv("HOME_ASSISTANT_URL", "http://localhost:8123")
        self.ha_token = os.getenv("HOME_ASSISTANT_TOKEN", "")
        
        # Demo mode if no token (just prints, doesn't actually connect)
        self.demo_mode = not self.ha_token
        
        if self.demo_mode:
            logging.warning("HOME_ASSISTANT_TOKEN not found - running in DEMO mode (simulated)")
            logging.info("Demo entities: light.bed_light, light.ceiling_lights, fan.living_room_fan")
        else:
            logging.info(f"Home Assistant configured: {self.ha_url}")
    
    def get_entity_id(self, device: str, action: str) -> str:
        """
        Map device name to Home Assistant entity ID.
        
        Args:
            device: Device name (e.g., "all", "bedroom", "thermostat", "fan")
            action: Action type (e.g., "lights on", "fan on", "play music")
        
        Returns:
            Home Assistant entity ID or shell command identifier
        """
        action_lower = action.lower()
        
        # Music control - uses shell command
        if "music" in action_lower or "spotify" in action_lower or "apple" in action_lower:
            return "shell.apple_music"
        
        # Special device types
        if device == "thermostat" or "heat" in action_lower:
            return "climate.thermostat"
        if device == "fan" or "fan" in action_lower:
            return "fan.living_room_fan"
        
        # Determine domain (light or fan)
        domain = "light" if "light" in action_lower else "fan"
        
        # Map device to entity - using virtual light for demo
        device_mapping = {
            "all": "input_boolean.living_room_light",
            "bedroom": "input_boolean.living_room_light",
            "kitchen": "input_boolean.living_room_light",
            "living_room": "input_boolean.living_room_light",
            "home_assistant": "script.order_service",  # For order workflow
        }
        
        return device_mapping.get(device, "input_boolean.living_room_light")
    
    def get_service(self, action: str, entity_id: str = "") -> str:
        """
        Map action to Home Assistant service.
        
        Args:
            action: IoT action (e.g., "lights on", "fan off")
            entity_id: Entity ID to check if it's a script
        
        Returns:
            Service name (e.g., "turn_on", "turn_off", "toggle")
        """
        # Scripts always use turn_on to execute
        if entity_id.startswith("script."):
            return "turn_on"
        
        action_lower = action.lower()
        
        # Use word boundaries to avoid false matches like "coffee" containing "off"
        words = action_lower.split()
        
        if "on" in words:
            return "turn_on"
        elif "off" in words:
            return "turn_off"
        elif "toggle" in words:
            return "toggle"
        else:
            return "turn_on"
    
    async def call_ha_service_demo(self, entity_id: str, service: str) -> bool:
        """Simulated Home Assistant call for demo mode."""
        action_emoji = "[LIGHT]" if "light" in entity_id else "[FAN]"
        state = "ON" if service == "turn_on" else "OFF" if service == "turn_off" else "TOGGLED"
        
        print(f"\n{action_emoji} [HOME ASSISTANT DEMO]")
        print(f"   Entity: {entity_id}")
        print(f"   Service: {service}")
        print(f"   Result: {state}")
        print(f"   (Simulated - no real device)\n")
        
        logging.info(f"Demo: {entity_id} → {service} → {state}")
        return True
    
    async def call_ha_service_real(self, entity_id: str, service: str) -> bool:
        """Real Home Assistant API call with fallback to demo."""
        domain = entity_id.split(".")[0]
        url = f"{self.ha_url}/api/services/{domain}/{service}"
        
        headers = {
            "Authorization": f"Bearer {self.ha_token}",
            "Content-Type": "application/json"
        }
        
        data = {"entity_id": entity_id}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        print(f"\n[HOME ASSISTANT] {entity_id} -> {service}")
                        print(f"   Status: Success\n")
                        logging.info(f"HA Success: {entity_id} → {service}")
                        return True
                    elif response.status == 400:
                        # Entity doesn't exist - fall back to demo mode
                        print(f"\nEntity {entity_id} not found - using demo mode")
                        return await self.call_ha_service_demo(entity_id, service)
                    else:
                        error_text = await response.text()
                        print(f"\n[HOME ASSISTANT ERROR] {response.status}")
                        print(f"   {error_text}\n")
                        logging.error(f"HA Error {response.status}: {error_text}")
                        return False
        
        except Exception as e:
            print(f"\nConnection failed - using demo mode")
            logging.warning(f"HA Connection error, using demo: {e}")
            return await self.call_ha_service_demo(entity_id, service)
    
    async def control_apple_music(self, action: str) -> bool:
        """Control Apple Music via osascript shell command."""
        import subprocess
        
        action_lower = action.lower()
        
        if "play" in action_lower or "start" in action_lower:
            # First activate Music app, then play
            cmd = '''osascript -e 'tell application "Music" to activate' -e 'delay 0.5' -e 'tell application "Music" to play' '''
            state = "PLAYING"
        elif "pause" in action_lower or "stop" in action_lower:
            cmd = 'osascript -e \'tell application "Music" to pause\''
            state = "PAUSED"
        elif "next" in action_lower or "skip" in action_lower:
            cmd = 'osascript -e \'tell application "Music" to next track\''
            state = "NEXT TRACK"
        else:
            cmd = '''osascript -e 'tell application "Music" to activate' -e 'delay 0.5' -e 'tell application "Music" to play' '''
            state = "PLAYING"
        
        print(f"\n[APPLE MUSIC]")
        print(f"   Command: {state}")
        
        try:
            subprocess.run(cmd, shell=True, check=True, capture_output=True)
            print(f"   Status: Success")
            return True
        except subprocess.CalledProcessError as e:
            print(f"   Status: Failed - {e}")
            return False

    async def connect(self, output_interface: IoTInput) -> None:
        """
        Execute IoT control action.
        
        Args:
            output_interface: IoT action to perform
        """
        try:
            # Parse action
            entity_id = self.get_entity_id(output_interface.device, output_interface.action)
            
            # Handle Apple Music separately
            if entity_id == "shell.apple_music":
                await self.control_apple_music(output_interface.action)
                return
            
            service = self.get_service(output_interface.action, entity_id)
            
            print(f"\n[IoT CONTROL] Processing request...")
            print(f"   Action: {output_interface.action}")
            print(f"   Device: {output_interface.device}")
            print(f"   Entity: {entity_id}")
            
            # Execute (demo or real)
            if self.demo_mode:
                success = await self.call_ha_service_demo(entity_id, service)
            else:
                success = await self.call_ha_service_real(entity_id, service)
            
            if not success:
                logging.warning(f"IoT action may have failed: {output_interface.action}")
        
        except Exception as e:
            print(f"[IoT ERROR] Unexpected error: {e}")
            logging.error(f"IoT control error: {e}", exc_info=True)

