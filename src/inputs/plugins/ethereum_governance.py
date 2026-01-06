import asyncio
import logging
import time
from typing import Optional

import requests

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider

# RULES are stored on the ETHEREUM HOLESKY testnet
# "GovernanceContract": "0xe706b7e30e378b89c7b2ee7bfd8ce2b91959d695"
# https://holesky.etherscan.io/address/0xe706b7e30e378b89c7b2ee7bfd8ce2b91959d695
# Note: Etherscan.io does not handle bytes[]/json well. See below for ways to
# interact with HOLESKY and decode the data, generating an ASCII string.


class GovernanceEthereum(FuserInput[SensorConfig, Optional[str]]):
    """
    Ethereum ERC-7777 reader that tracks governance rules.

    Queries the Ethereum blockchain for relevant governance rules.

    Raises
    ------
    Exception
        If connection to Ethereum network fails
    """

    def load_rules_from_blockchain(self):
        """
        Load governance rules from the Ethereum blockchain.
        """
        logging.info("Loading rules from Ethereum blockchain")

        # Construct JSON-RPC request
        payload = {
            "jsonrpc": "2.0",
            "id": 636815446436324,
            "method": "eth_call",
            "params": [
                {
                    "from": "0x0000000000000000000000000000000000000000",
                    "to": self.contract_address,
                    "data": f"{self.function_selector}{self.function_argument}",
                },
                "latest",
            ],
        }

        try:
            response = requests.post(
                self.rpc_url, json=payload, headers={"Content-Type": "application/json"}
            )
            logging.debug(f"Blockchain response status: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                if "result" in result and result["result"]:
                    hex_response = result["result"]
                    logging.debug(f"Raw blockchain response: {hex_response}")

                    # Decode the response using Web3.py
                    decoded_data = self.decode_eth_response(hex_response)
                    logging.debug(f"Decoded blockchain data: {decoded_data}")
                    return decoded_data
                else:
                    logging.error("Error: No valid result in blockchain response")
            else:
                logging.error(
                    f"Error: Blockchain request failed with status {response.status_code}"
                )

        except Exception as e:
            logging.error(f"Error loading rules from blockchain: {e}")

        return None

    def decode_eth_response(self, hex_response):
        """
        Decodes an Ethereum eth_call response.
        Extracts and decodes a UTF-8 string from ABI-encoded data.
        Cleans any unwanted control characters.
        """
        if hex_response.startswith("0x"):
            hex_response = hex_response[2:]

        try:
            response_bytes = bytes.fromhex(hex_response)

            # Read offsets and string length
            # offset = int.from_bytes(response_bytes[:32], "big")
            string_length = int.from_bytes(response_bytes[96:128], "big")

            # Extract and decode string
            string_bytes = response_bytes[128 : 128 + string_length]
            decoded_string = string_bytes.decode("utf-8")

            # Remove unexpected control characters (like \x19)
            cleaned_string = "".join(ch for ch in decoded_string if ch.isprintable())

            return cleaned_string

        except Exception as e:
            logging.error(f"Decoding error: {e}")
            return None

    def __init__(self, config: SensorConfig):
        """
        Initialize GovernanceEthereum instance.
        """
        super().__init__(config)

        self.descriptor_for_LLM = "Universal Laws"

        self.io_provider = IOProvider()
        self.POLL_INTERVAL = 5  # seconds
        self.rpc_url = "https://holesky.gateway.tenderly.co"  # Ethereum RPC URL

        # The smart contract address of the ERC-7777 Governance Smart Contract
        self.contract_address = "0xe706b7e30e378b89c7b2ee7bfd8ce2b91959d695"

        # getRuleSet() Function selector (first 4 bytes of Keccak hash).
        self.function_selector = "0x1db3d5ff"

        # The current rule set can be obtained from
        # getLatestRuleSetVersion(0x254e2f1e)
        # It's currently = 2
        self.function_argument = "0000000000000000000000000000000000000000000000000000000000000002"  # Argument

        self.universal_rule = self.load_rules_from_blockchain()
        self.messages: list[Message] = []

        logging.info(f"7777 rules: {self.universal_rule}")

    async def _poll(self) -> Optional[str]:
        """
        Poll for Ethereum Governance Law Changes.

        Returns
        -------
        Optional[str]
            Latest governance rules as a string, or None on error
        """
        await asyncio.sleep(self.POLL_INTERVAL)

        try:
            rules = self.load_rules_from_blockchain()
            logging.debug(f"7777 rules: {rules}")
            return rules
        except Exception as e:
            logging.error(f"Error fetching blockchain data: {e}")

    async def _raw_to_text(self, raw_input: Optional[str]) -> Optional[Message]:
        """
        Convert self.universal_rule to a human-readable Message.

        Parameters
        ----------
        raw_input : Optional[str]
            Raw input string to be processed

        Returns
        -------
        Optional[Message]
            Timestamped status or transaction notification
        """
        if raw_input is None:
            return None

        return Message(timestamp=time.time(), message=raw_input)

    async def raw_to_text(self, raw_input: Optional[str]):
        """
        Process governance rule message buffer.

        Parameters
        ----------
        raw_input : Optional[str]
            Raw input string to be processed
        """
        pending_message = await self._raw_to_text(raw_input)

        if pending_message is not None:
            if len(self.messages) == 0:
                self.messages.append(pending_message)
            # only update if there has been a change
            elif self.messages[-1].message != pending_message.message:
                self.messages.append(pending_message)

    def formatted_latest_buffer(self) -> Optional[str]:
        """
        Format and clear the latest buffer contents.

        Returns
        -------
        Optional[str]
            Formatted string of buffer contents or None if buffer is empty
        """
        if len(self.messages) == 0:
            return None

        latest_message = self.messages[-1]

        result = f"""
INPUT: {self.descriptor_for_LLM}
// START
{latest_message.message}
// END
"""

        self.io_provider.add_input(
            self.descriptor_for_LLM, latest_message.message, latest_message.timestamp
        )
        # no need to blank because we are only saving rare law changes
        # self.messages = []
        return result
