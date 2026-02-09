import asyncio
import logging
import time
from typing import Optional

from web3 import HTTPProvider, Web3
from web3.exceptions import BadFunctionCallOutput, ContractLogicError

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider


class GovernanceEthereum(FuserInput[SensorConfig, Optional[str]]):
    """Ethereum ERC-7777 reader that tracks governance rules."""

    CONTRACT_ABI = [
        {
            "inputs": [{"name": "_version", "type": "uint256"}],
            "name": "getRuleSet",
            "outputs": [{"name": "", "type": "string"}],
            "stateMutability": "view",
            "type": "function",
        }
    ]

    def __init__(self, config: SensorConfig):
        super().__init__(config)
        self.descriptor_for_LLM = "Universal Laws"
        self.io_provider = IOProvider()
        self.POLL_INTERVAL = 5.0  # seconds
        self.rpc_url = "https://holesky.drpc.org"
        self.w3 = Web3(HTTPProvider(self.rpc_url))
        self.contract_address = "0xe706b7e30e378b89c7b2ee7bfd8ce2b91959d695"

        try:
            self.contract = self.w3.eth.contract(
                address=self.w3.to_checksum_address(self.contract_address),
                abi=self.CONTRACT_ABI,
            )
        except Exception as e:
            logging.error(f"Failed to initialize Web3 contract object: {e}")
            raise

        self.rule_set_version = 2
        self.universal_rule: Optional[str] = None
        self.messages: list[Message] = []
        logging.info(
            f"GovernanceEthereum initialized for contract {self.contract_address}, "
            f"rules (version {self.rule_set_version}) will be loaded on first poll"
        )

    async def load_rules_from_blockchain(self) -> Optional[str]:
        """Load governance rules from the Ethereum blockchain using web3.py."""
        logging.info("Loading rules from Ethereum blockchain via web3.py")

        try:
            raw_result: str = self.contract.functions.getRuleSet(
                self.rule_set_estimator()
            ).call()
            logging.debug("Raw blockchain response (via web3): %s", raw_result)

            if raw_result is None:
                logging.warning("Contract function returned None.")
                return None

            # Clean non-printable characters but preserve newlines and tabs
            cleaned_string = "".join(
                ch for ch in raw_result if ch.isprintable() or ch in ["\n", "\r", "\t"]
            )
            logging.debug("Cleaned blockchain data: %s", cleaned_string)
            return cleaned_string

        except BadFunctionCallOutput as e:
            logging.error(
                "Blockchain function call failed (BadFunctionCallOutput): %s", e
            )
        except ContractLogicError as e:
            logging.error("Smart contract logic error during call: %s", e)
        except Exception as e:
            logging.error(
                "General error calling blockchain function via web3.py: %s", e
            )

        logging.error("Failed to load or decode rules from blockchain.")
        return None

    def rule_set_estimator(self) -> int:
        """Estimates the rule set version to fetch."""
        return self.rule_set_version

    async def _poll(self) -> Optional[str]:
        """Poll for Ethereum Governance Law Changes."""
        await asyncio.sleep(self.POLL_INTERVAL)
        try:
            rules = await self.load_rules_from_blockchain()
            logging.debug("7777 rules: %s", rules)
            return rules
        except Exception as e:
            logging.error("Error fetching blockchain data: %s", e)
            return None

    async def _raw_to_text(self, raw_input: Optional[str]) -> Optional[Message]:
        """Convert raw input to a human-readable Message."""
        if raw_input is None:
            return None
        return Message(timestamp=time.time(), message=raw_input)

    async def raw_to_text(self, raw_input: Optional[str]):
        """Process governance rule message buffer."""
        pending_message = await self._raw_to_text(raw_input)

        if pending_message is not None:
            if len(self.messages) == 0:
                self.messages.append(pending_message)
            elif self.messages[-1].message != pending_message.message:
                self.messages.append(pending_message)

    def formatted_latest_buffer(self) -> Optional[str]:
        """Format and return the latest buffer contents."""
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
        return result
