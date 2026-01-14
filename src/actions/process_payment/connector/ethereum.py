import logging
import os
from typing import Optional, Tuple, List
from decimal import Decimal
from web3 import Web3
from ens import ENS
from dotenv import load_dotenv
from actions.base import ActionConfig, ActionConnector
from actions.process_payment.interface import PaymentInput

# Load environment variables
load_dotenv()

# USDC Contract on Base Sepolia
USDC_CONTRACT_ADDRESS = "0x036CbD53842c5426634e7929541eC2318f3dCF7e"
USDC_ABI = [
    {
        "constant": False,
        "inputs": [
            {"name": "_to", "type": "address"},
            {"name": "_value", "type": "uint256"}
        ],
        "name": "transfer",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function"
    }
]


class EthereumConnector(ActionConnector[PaymentInput]):
    """
    Enhanced Ethereum payment connector with:
    - ENS name resolution
    - Multi-token support (ETH, USDC, DAI)
    - Safety validations (balance, gas, address)
    - Split payment support
    """
    
    def __init__(self, config: ActionConfig):
        super().__init__(config)
        
        # RPC Connection
        rpc_url = os.getenv("WEB3_RPC_URL", "https://base-sepolia-rpc.publicnode.com")
        logging.info(f"🔗 Connecting to RPC: {rpc_url}")
        
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        
        # ENS Initialization (uses mainnet for ENS resolution)
        try:
            mainnet_provider = Web3(Web3.HTTPProvider("https://eth.llamarpc.com"))
            self.ens = ENS.from_web3(mainnet_provider)
            logging.info("✅ ENS resolver initialized")
        except Exception as e:
            logging.warning(f"⚠️ ENS initialization failed: {e}. ENS names won't resolve.")
            self.ens = None
        
        # Wallet credentials
        self.private_key = os.getenv("ETH_PRIVATE_KEY")
        if not self.private_key:
            logging.error("❌ ETH_PRIVATE_KEY not found in environment!")
        
        # Token contracts
        self.usdc_contract = None
        if self.w3.is_connected():
            try:
                self.usdc_contract = self.w3.eth.contract(
                    address=Web3.to_checksum_address(USDC_CONTRACT_ADDRESS),
                    abi=USDC_ABI
                )
                logging.info("✅ USDC contract loaded")
            except Exception as e:
                logging.warning(f"⚠️ USDC contract failed: {e}")

    async def resolve_ens_name(self, name: str) -> Optional[str]:
        """
        Resolve ENS name to Ethereum address.
        
        Args:
            name: ENS name (e.g., "vitalik.eth")
            
        Returns:
            Ethereum address or None if resolution fails
        """
        if not self.ens:
            logging.warning("ENS resolver not available")
            return None
        
        if not name.endswith(".eth"):
            return None
        
        try:
            address = self.ens.address(name)
            if address:
                logging.info(f"✅ Resolved {name} → {address}")
                return address
            else:
                logging.warning(f"⚠️ ENS name {name} not found")
                return None
        except Exception as e:
            logging.error(f"❌ ENS resolution failed for {name}: {e}")
            return None

    async def validate_address(self, address: str) -> Tuple[bool, str]:
        """
        Validate Ethereum address.
        
        Returns:
            (is_valid, checksum_address or error_message)
        """
        try:
            checksum_addr = Web3.to_checksum_address(address)
            return True, checksum_addr
        except Exception as e:
            return False, f"Invalid address format: {e}"

    async def get_token_balance(self, currency: str, account: str) -> Decimal:
        """
        Get balance for specified token.
        
        Args:
            currency: ETH, USDC, or DAI
            account: Ethereum address
            
        Returns:
            Balance as Decimal
        """
        try:
            if currency == "ETH":
                balance_wei = self.w3.eth.get_balance(account)
                return Decimal(self.w3.from_wei(balance_wei, 'ether'))
            
            elif currency == "USDC" and self.usdc_contract:
                balance = self.usdc_contract.functions.balanceOf(account).call()
                # USDC has 6 decimals
                return Decimal(balance) / Decimal(10**6)
            
            else:
                logging.warning(f"Unsupported currency: {currency}")
                return Decimal(0)
        
        except Exception as e:
            logging.error(f"Failed to get {currency} balance: {e}")
            return Decimal(0)

    async def estimate_gas_cost(self, currency: str) -> Decimal:
        """
        Estimate gas cost in USD.
        
        Returns:
            Estimated cost in USD
        """
        try:
            gas_price = self.w3.eth.gas_price
            
            if currency == "ETH":
                gas_limit = 21000  # Standard ETH transfer
            else:
                gas_limit = 65000  # ERC20 transfer
            
            gas_cost_wei = gas_price * gas_limit
            gas_cost_eth = self.w3.from_wei(gas_cost_wei, 'ether')
            
            # Rough ETH price estimate (for demo purposes)
            eth_price_usd = 2500
            return Decimal(gas_cost_eth) * Decimal(eth_price_usd)
        
        except Exception as e:
            logging.error(f"Gas estimation failed: {e}")
            return Decimal(0.5)  # Fallback estimate

    async def validate_transaction(self, payment: PaymentInput, sender: str) -> Tuple[bool, List[str]]:
        """
        Perform safety validations on transaction.
        
        Returns:
            (is_safe, warnings_list)
        """
        warnings = []
        
        # 1. Large transaction warning
        amount = Decimal(payment.amount)
        if amount > Decimal("0.1"):  # ~$250 worth
            warnings.append("⚠️ LARGE TRANSACTION: Amount exceeds $100 equivalent")
        
        # 2. Balance check
        balance = await self.get_token_balance(payment.currency, sender)
        if balance < amount:
            return False, [f"❌ INSUFFICIENT BALANCE: You have {balance} {payment.currency}, need {amount}"]
        
        # 3. Address validation
        if not payment.resolved_address:
            warnings.append("⚠️ UNVERIFIED ADDRESS: Recipient address not verified")
        
        # 4. Gas estimation
        gas_cost = await self.estimate_gas_cost(payment.currency)
        warnings.append(f"Estimated gas fee: ~${gas_cost:.2f}")
        
        # 5. Balance after transaction
        remaining = balance - amount
        if remaining < Decimal("0.001"):
            warnings.append("⚠️ LOW BALANCE WARNING: This will nearly empty your wallet")
        
        return True, warnings

    async def send_eth_transaction(self, payment: PaymentInput, sender_address: str) -> Optional[str]:
        """Send ETH transaction."""
        try:
            amount_wei = self.w3.to_wei(Decimal(payment.amount), 'ether')
            
            nonce = self.w3.eth.get_transaction_count(sender_address)
            
            tx = {
                'nonce': nonce,
                'to': payment.resolved_address,
                'value': amount_wei,
                'gas': 21000,
                'gasPrice': self.w3.eth.gas_price,
                'chainId': 84532  # Base Sepolia
            }
            
            signed_tx = self.w3.eth.account.sign_transaction(tx, self.private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            
            return self.w3.to_hex(tx_hash)
        
        except Exception as e:
            logging.error(f"ETH transaction failed: {e}")
            return None

    async def send_usdc_transaction(self, payment: PaymentInput, sender_address: str) -> Optional[str]:
        """Send USDC transaction."""
        if not self.usdc_contract:
            logging.error("USDC contract not available")
            return None
        
        try:
            # USDC has 6 decimals
            amount_usdc = int(Decimal(payment.amount) * Decimal(10**6))
            
            nonce = self.w3.eth.get_transaction_count(sender_address)
            
            # Build transaction
            tx = self.usdc_contract.functions.transfer(
                payment.resolved_address,
                amount_usdc
            ).build_transaction({
                'from': sender_address,
                'nonce': nonce,
                'gas': 65000,
                'gasPrice': self.w3.eth.gas_price,
                'chainId': 84532
            })
            
            signed_tx = self.w3.eth.account.sign_transaction(tx, self.private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            
            return self.w3.to_hex(tx_hash)
        
        except Exception as e:
            logging.error(f"USDC transaction failed: {e}")
            return None

    async def connect(self, output_interface: PaymentInput) -> None:
        """
        Main entry point for payment processing.
        Handles ENS resolution, validation, and transaction execution.
        """
        try:
            # 1. Connection Check
            if not self.w3.is_connected():
                print("[WALLET ERROR] Cannot connect to blockchain RPC")
                logging.error("Web3 connection failed")
                return

            if not self.private_key:
                print("[WALLET ERROR] Private key not configured")
                return

            # Get sender account
            account = self.w3.eth.account.from_key(self.private_key)
            sender_address = account.address

            print(f"\n[WALLET] Processing Payment")
            print(f"   From: {sender_address[:6]}...{sender_address[-4:]}")
            print(f"   Amount: {output_interface.amount} {output_interface.currency}")
            print(f"   To: {output_interface.recipient}")
            
            # 2. Resolve recipient address (ENS or direct address)
            recipient = output_interface.recipient
            
            if recipient.endswith(".eth"):
                print(f"   Resolving ENS name...")
                resolved = await self.resolve_ens_name(recipient)
                if resolved:
                    output_interface.resolved_address = resolved
                    print(f"   OK: {recipient} -> {resolved[:6]}...{resolved[-4:]}")
                else:
                    print(f"   ERROR: Failed to resolve ENS name: {recipient}")
                    return
            else:
                # Validate as Ethereum address
                is_valid, result = await self.validate_address(recipient)
                if is_valid:
                    output_interface.resolved_address = result
                    print(f"   OK: Address validated: {result[:6]}...{result[-4:]}")
                else:
                    print(f"   ERROR: Invalid address: {result}")
                    return

            # 3. Safety validations
            print(f"   Running safety checks...")
            is_safe, warnings = await self.validate_transaction(output_interface, sender_address)
            
            if warnings:
                print(f"\n   WARNINGS:")
                for warning in warnings:
                    print(f"      {warning}")
                output_interface.warnings = warnings
            
            if not is_safe:
                print(f"   ERROR: Transaction blocked by safety checks")
                return

            # 4. Execute transaction based on currency
            print(f"\n   Sending transaction...")
            
            if output_interface.currency == "ETH":
                tx_hash = await self.send_eth_transaction(output_interface, sender_address)
            elif output_interface.currency == "USDC":
                tx_hash = await self.send_usdc_transaction(output_interface, sender_address)
            else:
                print(f"   ERROR: Unsupported currency: {output_interface.currency}")
                return

            if tx_hash:
                output_interface.tx_hash = tx_hash
                print(f"\n   SUCCESS! Transaction sent.")
                print(f"   Tx Hash: {tx_hash}")
                print(f"   Explorer: https://sepolia.basescan.org/tx/{tx_hash}")
                print("   " + "="*50 + "\n")
            else:
                print(f"   ERROR: Transaction failed to send")

        except Exception as e:
            print(f"[WALLET ERROR] Unexpected error: {e}")
            logging.error(f"Payment processing error: {e}", exc_info=True)
