"""
Mock LLM Plugin for VoiceWallet Demo
=====================================
Smarter keyword matching with proper action ordering and order workflow.
Meets bounty requirements: Order → Home Assistant → Payment → Confirmation
"""
import json
import logging
import re
import time
import typing as T
from difflib import SequenceMatcher

from llm import LLM, LLMConfig
from llm.output_model import CortexOutputModel, Action
from llm.function_schemas import convert_function_calls_to_actions


def fuzzy_match(word: str, targets: list[str], threshold: float = 0.7) -> bool:
    """Check if word fuzzy matches any target."""
    word = word.lower()
    for target in targets:
        ratio = SequenceMatcher(None, word, target.lower()).ratio()
        if ratio >= threshold:
            return True
    return False


def contains_any(text: str, words: list[str], fuzzy: bool = True) -> bool:
    """Check if text contains any of the words (with optional fuzzy matching)."""
    text_lower = text.lower()
    text_words = text_lower.split()
    
    for word in words:
        if word in text_lower:
            return True
        if fuzzy:
            for tw in text_words:
                if fuzzy_match(tw, [word], 0.75):
                    return True
    return False


class MockLLM(LLM):
    """
    A smarter mock LLM for VoiceWallet bounty demo.
    Supports: Balance, Payments, IoT Control, and Order workflow.
    """
    
    def __init__(
        self,
        config: LLMConfig = LLMConfig(),
        available_actions: T.Optional[T.List] = None,
    ):
        super().__init__(config, available_actions)
        logging.info("MockLLM initialized (smart mode)")

    async def ask(self, prompt: str, messages: T.List[T.Dict[str, str]] = []) -> T.Any:
        """Parse the prompt with fuzzy matching."""
        self.io_provider.llm_start_time = time.time()
        
        input_match = re.search(r'// START\n(.*?)\n// END', prompt, re.DOTALL)
        if not input_match:
            return None
        
        user_input = input_match.group(1).strip().lower()
        original_input = input_match.group(1).strip()
        
        function_calls = []
        
        # ===== ORDER WORKFLOW (Bounty main feature) =====
        # "Order coffee", "Buy pizza", "Order from store"
        if contains_any(user_input, ['order', 'buy', 'purchase', 'sipariş', 'satın al']):
            import os
            
            # Extract what to order
            order_item = "item"
            order_price = "0.001"  # Default price
            for item, price in [('coffee', '0.001'), ('pizza', '0.005'), ('food', '0.003'), ('drink', '0.001')]:
                if item in user_input:
                    order_item = item
                    order_price = price
                    break
            
            # Get merchant address from env
            merchant_address = os.getenv('COFFEE_SHOP_ADDRESS', '0x55c4d05fD7935d984C654A75603c219639c28213')
            
            # Check if user specified custom amount
            amount_match = re.search(r'(\d+\.?\d*)\s*(eth|usdc)', user_input)
            if amount_match:
                order_price = amount_match.group(1)
                currency = 'USDC' if 'usdc' in user_input.lower() else 'ETH'
            else:
                currency = 'ETH'
            
            # Step 1: Speak - Acknowledge order
            function_calls.append({
                "function": {
                    "name": "speak",
                    "arguments": json.dumps({"action": f"Ordering {order_item} for {order_price} {currency}. Connecting to Home Assistant..."})
                }
            })
            
            # Step 2: IoT - Trigger Home Assistant (order placement)
            function_calls.append({
                "function": {
                    "name": "control_device",
                    "arguments": json.dumps({"action": f"order_{order_item}", "device": "home_assistant"})
                }
            })
            
            # Step 3: Automatic Payment
            function_calls.append({
                "function": {
                    "name": "speak",
                    "arguments": json.dumps({"action": f"Order confirmed! Processing payment to merchant..."})
                }
            })
            
            function_calls.append({
                "function": {
                    "name": "pay",
                    "arguments": json.dumps({
                        "action": "pay",
                        "amount": order_price,
                        "currency": currency,
                        "recipient": merchant_address
                    })
                }
            })
            
            # Step 4: Final confirmation
            function_calls.append({
                "function": {
                    "name": "speak",
                    "arguments": json.dumps({"action": f"Order complete! Your {order_item} has been ordered and paid for."})
                }
            })
        
        # ===== BALANCE CHECK =====
        elif contains_any(user_input, ['balance', 'bakiye', 'money', 'funds', 'how much', 'wallet', 'check']):
            function_calls.append({
                "function": {
                    "name": "speak",
                    "arguments": json.dumps({"action": "You have 0.0285 ETH and 120.50 USDC in your wallet on Base Sepolia."})
                }
            })
        
        # ===== PAYMENT =====
        elif contains_any(user_input, ['send', 'pay', 'transfer', 'gönder', 'öde']):
            amount_match = re.search(r'(\d+\.?\d*)\s*(eth|usdc|ether)', user_input)
            amount = amount_match.group(1) if amount_match else None
            currency = 'ETH'
            if amount_match:
                curr = amount_match.group(2).upper()
                currency = 'USDC' if curr == 'USDC' else 'ETH'
            
            ens_match = re.search(r'(\w+\.eth)', user_input)
            addr_match = re.search(r'(0x[a-fA-F0-9]{40})', original_input)
            
            if not (ens_match or addr_match):
                function_calls.append({
                    "function": {
                        "name": "speak",
                        "arguments": json.dumps({"action": "I need a valid recipient. Provide an ENS name (like alice.eth) or Ethereum address (0x...)."})
                    }
                })
            elif not amount:
                recipient = ens_match.group(1) if ens_match else addr_match.group(1)
                function_calls.append({
                    "function": {
                        "name": "speak",
                        "arguments": json.dumps({"action": f"How much to send to {recipient}? Say like '0.01 ETH'."})
                    }
                })
            else:
                recipient = ens_match.group(1) if ens_match else addr_match.group(1)
                
                # FIXED: Speak BEFORE payment
                function_calls.append({
                    "function": {
                        "name": "speak",
                        "arguments": json.dumps({"action": f"Processing payment: {amount} {currency} to {recipient}..."})
                    }
                })
                function_calls.append({
                    "function": {
                        "name": "pay",
                        "arguments": json.dumps({
                            "action": "pay",
                            "amount": amount,
                            "currency": currency,
                            "recipient": recipient
                        })
                    }
                })
                # Confirmation after payment
                function_calls.append({
                    "function": {
                        "name": "speak",
                        "arguments": json.dumps({"action": "Payment sent! Check the transaction hash above."})
                    }
                })
        
        # ===== IOT CONTROL =====
        elif contains_any(user_input, ['light', 'lights', 'lamp', 'heat', 'heating', 'fan', 'ac', 'turn', 'switch', 'aç', 'kapat']):
            action_type = 'lights off' if contains_any(user_input, ['off', 'kapat', 'close']) else 'lights on'
            
            if contains_any(user_input, ['bedroom', 'yatak']):
                device = 'bedroom'
            elif contains_any(user_input, ['kitchen', 'mutfak']):
                device = 'kitchen'
            elif contains_any(user_input, ['heat', 'heating', 'ısıtma']):
                device = 'thermostat'
            elif contains_any(user_input, ['fan', 'vantilatör']):
                device = 'fan'
            else:
                device = 'all'
            
            # Speak first
            msg = f"Controlling {device} via Home Assistant..."
            function_calls.append({
                "function": {
                    "name": "speak",
                    "arguments": json.dumps({"action": msg})
                }
            })
            
            function_calls.append({
                "function": {
                    "name": "control_device",
                    "arguments": json.dumps({"action": action_type, "device": device})
                }
            })
            
            # Confirmation
            function_calls.append({
                "function": {
                    "name": "speak",
                    "arguments": json.dumps({"action": f"Done! {device.title()} is now {'on' if 'on' in action_type else 'off'}."})
                }
            })
        
        # ===== HELP =====
        elif contains_any(user_input, ['hello', 'hi', 'help', 'merhaba', 'what can']):
            function_calls.append({
                "function": {
                    "name": "speak",
                    "arguments": json.dumps({"action": "I'm your Smart Wallet Assistant! I can: 1) Check balance 2) Send crypto (ETH/USDC) 3) Control smart home via Home Assistant 4) Order items and pay with crypto. Try: 'order coffee and pay 5 USDC to shop.eth'"})
                }
            })
        
        # ===== DEFAULT =====
        else:
            function_calls.append({
                "function": {
                    "name": "speak",
                    "arguments": json.dumps({"action": f"I heard: '{original_input}'. Try: 'check balance', 'send 0.01 ETH to alice.eth', 'turn on lights', or 'order coffee'."})
                }
            })
        
        self.io_provider.llm_end_time = time.time()
        
        if function_calls:
            actions = convert_function_calls_to_actions(function_calls)
            return CortexOutputModel(actions=actions)
        
        return None
