import logging
import aiohttp
from typing import Optional

# PR #2069: Use strict decoding library instead of manual slicing
from eth_abi import decode
from eth_utils import to_bytes
from eth_abi.exceptions import DecodingError

from inputs.base import SensorConfig
from inputs.base.loop import FuserInput

logger = logging.getLogger(__name__)

class GovernanceEthereum(FuserInput[SensorConfig, Optional[str]]):
    """
    Ethereum governance reader implementing strict ABI validation.
    Aligned with PR #2069 standards for data integrity.
    """

    async def load_rules_from_blockchain(self) -> Optional[str]:
        """
        Fetches and decodes governance rules from the Ethereum blockchain.
        """
        logger.info("Governance: Fetching rules from Ethereum...")

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "eth_call",
            "params": [
                {
                    "to": self.contract_address,
                    "data": f"{self.function_selector}{self.function_argument}",
                },
                "latest",
            ],
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.rpc_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    if response.status != 200:
                        logger.error(f"Governance: RPC request failed. Status: {response.status}")
                        return None

                    result = await response.json()
                    
                    # Validate RPC structure
                    if "result" not in result:
                        logger.error("Governance: Malformed RPC response (missing 'result').")
                        return None
                        
                    hex_response = result["result"]
                    return self.decode_eth_response(hex_response)

        except Exception as e:
            logger.error(f"Governance: Network/RPC error: {e}")
            return None

    def decode_eth_response(self, hex_response: Optional[str]) -> str:
        """
        Decodes Ethereum RPC response using strict validation (eth-abi).
        
        Ref: Fixes Issue #1824 & Aligns with PR #2069 standards.
        
        Args:
            hex_response: The raw hex string from the RPC node.
            
        Returns:
            str: Decoded string if valid, empty string otherwise.
        """
        # SCENARIO B: Network Error / Null Response
        if hex_response is None:
            logger.warning("Governance: Received None response from RPC.")
            return ""

        # Validation: Ensure it is a string
        if not isinstance(hex_response, str):
            logger.warning(f"Governance: Invalid type received: {type(hex_response)}")
            return ""

        # SCENARIO A: Valid Empty Response (0x)
        # This explicitly means the contract returned empty data.
        if hex_response == "0x":
            logger.debug("Governance: Received valid empty '0x' response.")
            return ""

        try:
            # Normalize hex string
            if hex_response.startswith("0x"):
                hex_response = hex_response[2:]
                
            byte_data = to_bytes(hexstr=hex_response)

            # SCENARIO C: Strict Decoding
            # eth-abi raises DecodingError if data is truncated or malformed.
            # decode() returns a tuple, so we extract the first element.
            decoded_tuple = decode(['string'], byte_data)
            
            result_string = decoded_tuple[0]
            logger.debug(f"Governance: Successfully decoded rules (Length: {len(result_string)})")
            return result_string

        except DecodingError as e:
            # Critical: Log explicitly to prevent silent failures on corruption.
            logger.error(f"Governance: ABI Decoding Failed (Corrupted Data): {e}")
            return ""
            
        except Exception as e:
            logger.error(f"Governance: Unexpected error during decoding: {e}")
            return ""