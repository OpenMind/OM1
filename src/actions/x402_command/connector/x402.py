import base64
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

import aiohttp
from eth_account import Account
from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.x402_command.interface import X402PaymentInput

NETWORK_CONFIG: Dict[str, Dict[str, Any]] = {
    "base-sepolia": {
        "chain_id": 84532,
        "usdc_address": "0x036CbD53842c5426634e7929541eC2318f3dCF7e",
    },
    "base": {
        "chain_id": 8453,
        "usdc_address": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
    },
}

EIP712_DOMAIN_TYPE = [
    {"name": "name", "type": "string"},
    {"name": "version", "type": "string"},
    {"name": "chainId", "type": "uint256"},
    {"name": "verifyingContract", "type": "address"},
]

EIP712_TRANSFER_TYPE = {
    "TransferWithAuthorization": [
        {"name": "from", "type": "address"},
        {"name": "to", "type": "address"},
        {"name": "value", "type": "uint256"},
        {"name": "validAfter", "type": "uint256"},
        {"name": "validBefore", "type": "uint256"},
        {"name": "nonce", "type": "bytes32"},
    ]
}


class X402Config(ActionConfig):
    """
    Configuration for the x402 payment connector.

    Parameters
    ----------
    private_key : Optional[str]
        Private key for signing EIP-712 payment authorizations.
    x402_endpoint : Optional[str]
        URL of the x402-protected endpoint.
    network : str
        Blockchain network identifier (e.g. 'base-sepolia', 'base').
    """

    private_key: Optional[str] = Field(
        default=None,
        description="Private key for signing EIP-712 payment authorizations",
    )
    x402_endpoint: Optional[str] = Field(
        default=None,
        description="URL of the x402-protected endpoint",
    )
    network: str = Field(
        default="base-sepolia",
        description="Blockchain network identifier",
    )


class X402Connector(ActionConnector[X402Config, X402PaymentInput]):
    """
    Connector that handles x402 payment protocol interactions.

    Sends messages to x402-protected endpoints by signing EIP-712
    payment authorizations and completing USDC transfers.
    """

    def __init__(self, config: X402Config):
        """
        Initialize the x402 connector.

        Parameters
        ----------
        config : X402Config
            Configuration for the x402 connector.
        """
        super().__init__(config)
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """
        Get or create an aiohttp client session.

        Returns
        -------
        aiohttp.ClientSession
            The HTTP client session.
        """
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _fetch_payment_requirements(
        self, session: aiohttp.ClientSession, endpoint: str
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch payment requirements from the x402 endpoint.

        Parameters
        ----------
        session : aiohttp.ClientSession
            The HTTP client session.
        endpoint : str
            The x402-protected endpoint URL.

        Returns
        -------
        Optional[Dict[str, Any]]
            Payment requirements dict or None on failure.
        """
        try:
            async with session.post(endpoint) as resp:
                if resp.status != 402:
                    logging.error(f"Expected 402 from endpoint, got {resp.status}")
                    return None
                data = await resp.json()
                accepts = data.get("accepts")
                if not accepts:
                    logging.error("No payment requirements in 402 response")
                    return None
                return accepts[0]
        except Exception as e:
            logging.error(f"Failed to fetch payment requirements: {e}")
            return None

    def _build_payment_payload(
        self,
        pay_to: str,
        amount: str,
        timeout_seconds: int,
    ) -> str:
        """
        Build a base64-encoded x402 payment payload.

        Parameters
        ----------
        pay_to : str
            Recipient wallet address.
        amount : str
            Payment amount in smallest unit.
        timeout_seconds : int
            Maximum validity window in seconds.

        Returns
        -------
        str
            Base64-encoded JSON payment payload.
        """
        network = self.config.network
        net_cfg = NETWORK_CONFIG.get(network)
        if net_cfg is None:
            raise ValueError(f"Unsupported network: {network}")

        chain_id = net_cfg["chain_id"]
        usdc_address = net_cfg["usdc_address"]

        account = Account.from_key(self.config.private_key)
        sender_address = account.address

        current_time = int(datetime.now().timestamp())
        valid_after = 0
        valid_before = current_time + timeout_seconds * 2
        nonce = os.urandom(32)

        domain_data = {
            "name": "USDC",
            "version": "2",
            "chainId": chain_id,
            "verifyingContract": usdc_address,
        }

        message_data = {
            "from": sender_address,
            "to": pay_to,
            "value": str(amount),
            "validAfter": str(valid_after),
            "validBefore": str(valid_before),
            "nonce": "0x" + nonce.hex(),
        }

        signed_msg = Account.sign_typed_data(
            self.config.private_key,
            domain_data,
            EIP712_TRANSFER_TYPE,
            message_data,
        )

        payload = {
            "x402Version": 1,
            "scheme": "exact",
            "network": network,
            "payload": {
                "signature": "0x" + signed_msg.signature.hex(),
                "authorization": message_data,
            },
        }

        return base64.b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8")

    async def connect(self, output_interface: X402PaymentInput) -> None:
        """
        Send a message to the x402-protected endpoint with payment.

        Handles the full 402 flow: fetches payment requirements,
        signs an EIP-712 authorization, and sends the paid request.

        Parameters
        ----------
        output_interface : X402PaymentInput
            The input containing the message to send.
        """
        if not self.config.x402_endpoint:
            logging.error("x402 endpoint is not configured")
            return

        if not self.config.private_key:
            logging.error("Private key is not configured")
            return

        endpoint = self.config.x402_endpoint
        session = await self._get_session()

        requirements = await self._fetch_payment_requirements(session, endpoint)
        if requirements is None:
            return

        pay_to = requirements.get("payTo")
        amount = requirements.get("maxAmountRequired")
        timeout = requirements.get("maxTimeoutSeconds", 300)
        network = requirements.get("network", self.config.network)

        if not pay_to or not amount:
            logging.error("Incomplete payment requirements from endpoint")
            return

        if network != self.config.network:
            logging.warning(
                f"Endpoint network '{network}' differs from configured "
                f"'{self.config.network}', using endpoint network"
            )
            self.config.network = network

        try:
            encoded_payload = self._build_payment_payload(
                pay_to=pay_to,
                amount=amount,
                timeout_seconds=int(timeout),
            )
        except (ValueError, Exception) as e:
            logging.error(f"Failed to build payment payload: {e}")
            return

        headers = {
            "X-PAYMENT": encoded_payload,
            "Content-Type": "application/json",
        }

        logging.info(
            f"Sending x402 payment to {endpoint} "
            f"with message: {output_interface.action}"
        )

        try:
            async with session.post(
                endpoint,
                headers=headers,
                json={"message": output_interface.action},
            ) as resp:
                if resp.status == 200:
                    logging.info(f"x402 payment successful: {output_interface.action}")
                else:
                    body = await resp.text()
                    logging.error(
                        f"x402 payment failed with status {resp.status}: {body}"
                    )
        except Exception as e:
            logging.error(f"Error sending x402 request: {e}")

    def stop(self) -> None:
        """
        Clean up the HTTP client session.
        """
        if self._session and not self._session.closed:
            # aiohttp session close is async; schedule if loop is running
            try:
                import asyncio

                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._session.close())
                else:
                    loop.run_until_complete(self._session.close())
            except Exception:
                pass
