import yaml
import time
import random
from typing import Dict, Any

class SmartAssistantPlugin:
    """
    Integrates OM1 with smart assistants (Home Assistant, Alexa, Google)
    Processes voice commands and triggers wallet payments
    """
    
    def __init__(self, config_path="src/plugins/assistant/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        
        self.mock_mode = cfg.get("mock_mode", True)
        self.assistant_type = cfg.get("assistant_type", "alexa")
        self.webhook_url = cfg.get("webhook_url", "")
        
        self.last_command = None
        self.order_history = []
    
    def process_voice_command(self, command: str) -> Dict[str, Any]:
        """
        Process natural language command and extract payment intent
        Returns: {action, recipient, amount, currency, order_details}
        """
        command_lower = command.lower()
        
        # Parse payment commands
        if "pay" in command_lower or "send" in command_lower:
            parts = command.split()
            
            result = {
                "action": "payment",
                "raw_command": command,
                "parsed": True,
                "status": "pending"
            }
            
            # Simple parsing logic
            try:
                for i, word in enumerate(parts):
                    if word.replace('.', '').replace('-', '').isdigit() or ('.' in word and word.replace('.', '').isdigit()):
                        result["amount"] = float(word)
                    if word.upper() in ["ETH", "BTC", "USDT"]:
                        result["currency"] = word.upper()
                    if "to" in parts and i == parts.index("to") + 1:
                        result["recipient"] = word
                
                self.last_command = result
                return result
            except Exception as e:
                return {
                    "action": "error",
                    "error": f"Failed to parse command: {str(e)}",
                    "raw_command": command
                }
        
        # Parse order commands
        elif "order" in command_lower or "buy" in command_lower:
            return {
                "action": "order",
                "raw_command": command,
                "status": "pending",
                "requires_payment": True
            }
        
        return {
            "action": "unknown",
            "raw_command": command,
            "error": "Command not recognized"
        }
    
    def create_order(self, item: str, quantity: int = 1) -> Dict[str, Any]:
        """Create an order via smart assistant"""
        order = {
            "order_id": f"ORD-{random.randint(1000, 9999)}",
            "item": item,
            "quantity": quantity,
            "timestamp": time.time(),
            "status": "created",
            "payment_required": True
        }
        
        self.order_history.append(order)
        return order
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get order status"""
        for order in self.order_history:
            if order["order_id"] == order_id:
                return order
        return {"error": "Order not found"}
