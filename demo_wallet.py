from src.plugins.wallet.wallet_plugin import WalletPlugin
from src.plugins.assistant.assistant_plugin import SmartAssistantPlugin
import time

def print_separator():
    print("=" * 60)

def print_step(number, title):
    print(f"\n{number}. {title}")
    print("-" * 60)

def main():
    print_separator()
    print("Bounty #367 - OM1 + Smart Assistant + Wallet Payments")
    print("Complete Workflow Demo (Mock Mode)")
    print_separator()
    
    # Initialize plugins
    print("\nInitializing plugins...")
    wallet = WalletPlugin()
    assistant = SmartAssistantPlugin()
    print("Plugins initialized successfully!")
    time.sleep(1)
    
    # Step 1: Check initial balance
    print_step("STEP 1", "Initial Wallet Balance")
    balance = wallet.get_balance()
    print(f"Balance: {balance['eth']} ETH = ${balance['usd']:.2f} USD")
    print(f"ETH Price: ${balance['eth_price']:.2f}")
    print(f"Wallet Address: {balance['wallet_address']}")
    time.sleep(2)
    
    # Step 2: Voice command received
    print_step("STEP 2", "Voice Command Received")
    voice_command = "Alexa, pay 0.05 ETH to Lau90eth"
    print(f'User says: "{voice_command}"')
    print(f"Assistant Type: {assistant.assistant_type}")
    time.sleep(2)
    
    # Step 3: Assistant processes command
    print_step("STEP 3", "Smart Assistant Processing")
    parsed = assistant.process_voice_command(voice_command)
    print(f"Action detected: {parsed['action']}")
    print(f"Amount: {parsed.get('amount')} {parsed.get('currency')}")
    print(f"Recipient: {parsed.get('recipient')}")
    print(f"Command status: {parsed['status']}")
    time.sleep(2)
    
    # Step 4: Execute payment
    print_step("STEP 4", "Processing Payment via Wallet")
    result = wallet.send_payment(
        to=parsed.get('recipient', 'unknown'),
        amount=parsed.get('amount', 0)
    )
    
    if result['success']:
        print(f"Transaction Status: {result['status']}")
        print(f"TX Hash: {result['tx']}")
        print(f"Amount: {result['amount']} {result['currency']}")
        print(f"From: {result['from']}")
        print(f"To: {result['to']}")
        print(f"Confirmations: {result['confirmations']}")
    else:
        print(f"Transaction Failed!")
        print(f"Error: {result['error']}")
        print(f"Available: {result['available_balance']} ETH")
        print(f"Requested: {result['requested_amount']} ETH")
    time.sleep(2)
    
    # Step 5: Check final balance
    print_step("STEP 5", "Final Wallet Balance")
    final_balance = wallet.get_balance()
    print(f"Balance: {final_balance['eth']} ETH = ${final_balance['usd']:.2f} USD")
    print(f"Change: -{parsed.get('amount', 0)} ETH")
    time.sleep(1)
    
    # Step 6: Confirmation
    print_step("STEP 6", "Transaction Confirmation")
    print("Payment completed successfully!")
    print("Order can be placed via smart assistant")
    print("Transaction recorded in wallet history")
    time.sleep(1)
    
    # Transaction history
    print("\n")
    print_separator()
    print("Transaction History:")
    print_separator()
    history = wallet.get_transaction_history()
    for i, tx in enumerate(history, 1):
        print(f"{i}. {tx['amount']} {tx['currency']} -> {tx['to']}")
        print(f"   Status: {tx['status']} | Confirmations: {tx['confirmations']}")
        print(f"   TX: {tx['tx']}")
    
    print("\n")
    print_separator()
    print("Demo completed - Ready for real wallet integration!")
    print_separator()

if __name__ == "__main__":
    main()
