import pytest
from src.plugins.wallet.wallet_plugin import WalletPlugin
from src.plugins.assistant.assistant_plugin import SmartAssistantPlugin
import time

class TestWalletPlugin:
    """Test suite for WalletPlugin"""
    
    def test_wallet_initialization(self):
        """Test wallet initializes correctly"""
        wallet = WalletPlugin()
        assert wallet.mock_mode == True
        assert wallet.balance_eth > 0
        assert wallet.eth_price_usd > 0
    
    def test_get_balance(self):
        """Test wallet balance retrieval"""
        wallet = WalletPlugin()
        balance = wallet.get_balance()
        
        assert 'eth' in balance
        assert 'usd' in balance
        assert 'eth_price' in balance
        assert balance['eth'] > 0
        assert balance['usd'] > 0
    
    def test_get_wallet_info(self):
        """Test wallet info retrieval"""
        wallet = WalletPlugin()
        info = wallet.get_wallet_info()
        
        assert 'address' in info
        assert 'balance' in info
        assert 'transaction_count' in info
        assert 'mode' in info
        assert info['mode'] == 'mock'
    
    def test_payment_success(self):
        """Test successful payment"""
        wallet = WalletPlugin()
        initial_balance = wallet.get_balance()['eth']
        
        result = wallet.send_payment(to="TestUser", amount=0.01)
        
        assert result['success'] == True
        assert result['status'] in ['pending', 'confirmed']
        assert 'tx' in result
        assert result['amount'] == 0.01
        assert result['to'] == "TestUser"
        assert wallet.get_balance()['eth'] < initial_balance
    
    def test_payment_insufficient_funds(self):
        """Test payment with insufficient funds"""
        wallet = WalletPlugin()
        result = wallet.send_payment(to="TestUser", amount=999)
        
        assert result['success'] == False
        assert result['status'] == 'failed'
        assert 'error' in result
        assert result['error'] == 'insufficient funds'
        assert 'available_balance' in result
        assert 'requested_amount' in result
    
    def test_transaction_history(self):
        """Test transaction history tracking"""
        wallet = WalletPlugin()
        
        # Make a payment
        wallet.send_payment(to="User1", amount=0.01)
        
        history = wallet.get_transaction_history()
        assert len(history) > 0
        assert history[0]['to'] == "User1"
        assert history[0]['amount'] == 0.01
    
    def test_get_transaction_status(self):
        """Test transaction status retrieval"""
        wallet = WalletPlugin()
        
        # Make a payment
        result = wallet.send_payment(to="User1", amount=0.01)
        tx_hash = result['tx']
        
        # Get status
        status = wallet.get_transaction_status(tx_hash)
        assert 'status' in status
        assert 'confirmations' in status
        assert status['tx'] == tx_hash
    
    def test_transaction_not_found(self):
        """Test getting status of non-existent transaction"""
        wallet = WalletPlugin()
        status = wallet.get_transaction_status("0xinvalid")
        
        assert 'error' in status
        assert status['error'] == 'Transaction not found'


class TestSmartAssistantPlugin:
    """Test suite for SmartAssistantPlugin"""
    
    def test_assistant_initialization(self):
        """Test assistant initializes correctly"""
        assistant = SmartAssistantPlugin()
        assert assistant.mock_mode == True
        assert assistant.assistant_type in ['alexa', 'google', 'homeassistant']
    
    def test_parse_payment_command_full(self):
        """Test parsing complete payment command"""
        assistant = SmartAssistantPlugin()
        result = assistant.process_voice_command("pay 0.05 ETH to Lau90eth")
        
        assert result['action'] == 'payment'
        assert result['amount'] == 0.05
        assert result['currency'] == 'ETH'
        assert result['recipient'] == 'Lau90eth'
        assert result['parsed'] == True
    
    def test_parse_payment_command_alexa(self):
        """Test parsing Alexa-style command"""
        assistant = SmartAssistantPlugin()
        result = assistant.process_voice_command("Alexa, send 0.1 ETH to Bob")
        
        assert result['action'] == 'payment'
        assert result['amount'] == 0.1
        assert result['currency'] == 'ETH'
    
    def test_parse_order_command(self):
        """Test parsing order command"""
        assistant = SmartAssistantPlugin()
        result = assistant.process_voice_command("order pizza")
        
        assert result['action'] == 'order'
        assert result['requires_payment'] == True
    
    def test_parse_unknown_command(self):
        """Test parsing unknown command"""
        assistant = SmartAssistantPlugin()
        result = assistant.process_voice_command("play music")
        
        assert result['action'] == 'unknown'
        assert 'error' in result
    
    def test_create_order(self):
        """Test order creation"""
        assistant = SmartAssistantPlugin()
        order = assistant.create_order(item="pizza", quantity=2)
        
        assert 'order_id' in order
        assert order['item'] == "pizza"
        assert order['quantity'] == 2
        assert order['status'] == "created"
        assert order['payment_required'] == True
    
    def test_get_order_status(self):
        """Test getting order status"""
        assistant = SmartAssistantPlugin()
        order = assistant.create_order(item="pizza")
        
        status = assistant.get_order_status(order['order_id'])
        assert status['order_id'] == order['order_id']
    
    def test_order_not_found(self):
        """Test getting non-existent order"""
        assistant = SmartAssistantPlugin()
        status = assistant.get_order_status("INVALID")
        
        assert 'error' in status


class TestIntegration:
    """Integration tests for complete workflow"""
    
    def test_full_workflow_success(self):
        """Test complete voice-to-payment workflow"""
        wallet = WalletPlugin()
        assistant = SmartAssistantPlugin()
        
        initial_balance = wallet.get_balance()['eth']
        
        # Step 1: Parse voice command
        parsed = assistant.process_voice_command("Alexa, pay 0.02 ETH to Bob")
        assert parsed['action'] == 'payment'
        
        # Step 2: Execute payment
        result = wallet.send_payment(
            to=parsed['recipient'],
            amount=parsed['amount']
        )
        assert result['success'] == True
        
        # Step 3: Verify balance changed
        final_balance = wallet.get_balance()['eth']
        assert final_balance < initial_balance
        
        # Step 4: Check transaction history
        history = wallet.get_transaction_history()
        assert len(history) > 0
        assert history[-1]['to'] == 'Bob'
    
    def test_workflow_insufficient_funds(self):
        """Test workflow with insufficient funds"""
        wallet = WalletPlugin()
        assistant = SmartAssistantPlugin()
        
        # Try to send more than available
        parsed = assistant.process_voice_command("pay 999 ETH to Bob")
        result = wallet.send_payment(
            to=parsed.get('recipient', 'unknown'),
            amount=parsed.get('amount', 0)
        )
        
        assert result['success'] == False
        assert result['status'] == 'failed'
    
    def test_multiple_transactions(self):
        """Test multiple sequential transactions"""
        wallet = WalletPlugin()
        assistant = SmartAssistantPlugin()
        
        # Execute 3 transactions
        for i in range(3):
            parsed = assistant.process_voice_command(f"pay 0.01 ETH to User{i}")
            result = wallet.send_payment(
                to=parsed['recipient'],
                amount=parsed['amount']
            )
            assert result['success'] == True
        
        # Verify all in history
        history = wallet.get_transaction_history()
        assert len(history) >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
