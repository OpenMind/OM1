import logging
import os
from decimal import Decimal
from web3 import Web3
from dotenv import load_dotenv
from actions.base import ActionConfig, ActionConnector
from actions.check_balance.interface import BalanceInput

load_dotenv()

# USDC Contract on Base Sepolia
USDC_CONTRACT_ADDRESS = "0x036CbD53842c5426634e7929541eC2318f3dCF7e"
USDC_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function"
    }
]


class EthereumBalanceConnector(ActionConnector[BalanceInput]):
    """
    Connector to check wallet balance on Ethereum/Base.
    """
    
    def __init__(self, config: ActionConfig):
        super().__init__(config)
        
        rpc_url = os.getenv("WEB3_RPC_URL", "https://base-sepolia-rpc.publicnode.com")
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        
        self.private_key = os.getenv("ETH_PRIVATE_KEY")
        
        # USDC contract
        self.usdc_contract = None
        if self.w3.is_connected():
            try:
                self.usdc_contract = self.w3.eth.contract(
                    address=Web3.to_checksum_address(USDC_CONTRACT_ADDRESS),
                    abi=USDC_ABI
                )
            except Exception as e:
                logging.warning(f"USDC contract failed: {e}")

    async def connect(self, input_data: BalanceInput) -> None:
        """
        Check and display wallet balance.
        """
        try:
            if not self.w3.is_connected():
                print("[BALANCE ERROR] Cannot connect to blockchain")
                return
            
            if not self.private_key:
                print("[BALANCE ERROR] Private key not configured")
                return
            
            # Get wallet address
            account = self.w3.eth.account.from_key(self.private_key)
            address = account.address
            
            # Get ETH balance
            eth_balance_wei = self.w3.eth.get_balance(address)
            eth_balance = Decimal(self.w3.from_wei(eth_balance_wei, 'ether'))
            
            # Get USDC balance
            usdc_balance = Decimal(0)
            if self.usdc_contract:
                try:
                    usdc_raw = self.usdc_contract.functions.balanceOf(address).call()
                    usdc_balance = Decimal(usdc_raw) / Decimal(10**6)
                except Exception as e:
                    logging.warning(f"USDC balance check failed: {e}")
            
            # Display balance
            print(f"\n[WALLET BALANCE]")
            print(f"   Address: {address[:6]}...{address[-4:]}")
            print(f"   ETH: {eth_balance:.6f}")
            print(f"   USDC: {usdc_balance:.2f}")
            print(f"   Network: Base Sepolia")
            print(f"   " + "="*40 + "\n")
            
        except Exception as e:
            print(f"[BALANCE ERROR] {e}")
            logging.error(f"Balance check error: {e}", exc_info=True)
