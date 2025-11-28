from src.plugins.wallet.wallet_plugin import WalletPlugin
from src.plugins.assistant.assistant_plugin import SmartAssistantPlugin

def main():
    print("=== Demo: Error Handling ===\n")
    
    wallet = WalletPlugin()
    assistant = SmartAssistantPlugin()
    
    # Test 1: Insufficient funds
    print("1. Test insufficient funds:")
    balance = wallet.get_balance()
    print(f"   Current balance: {balance['eth']} ETH")
    
    result = wallet.send_payment(to="TestUser", amount=999)
    print(f"   Trying to send 999 ETH...")
    print(f"   Result: {result['status']}")
    print(f"   Error: {result['error']}")
    print()
    
    # Test 2: Unknown command
    print("2. Test unknown command:")
    parsed = assistant.process_voice_command("play music")
    print(f"   Command: 'play music'")
    print(f"   Action: {parsed['action']}")
    print(f"   Error: {parsed.get('error', 'N/A')}")
    print()
    
    # Test 3: Malformed command
    print("3. Test malformed payment command:")
    parsed = assistant.process_voice_command("pay something to someone")
    print(f"   Command: 'pay something to someone'")
    print(f"   Action: {parsed['action']}")
    print(f"   Parsed amount: {parsed.get('amount', 'N/A')}")
    print()
    
    print("All error handling tests completed!")

if __name__ == "__main__":
    main()
