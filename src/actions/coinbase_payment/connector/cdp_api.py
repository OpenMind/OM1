import asyncio
import logging
import os
import re
import time
from typing import Optional

import aiohttp
from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.coinbase_payment.interface import CoinbasePaymentInput
from providers.io_provider import IOProvider

try:
    from cdp import CdpClient
    from cdp.evm_transaction_types import TransactionRequestEIP1559
    CDP_AVAILABLE = True
except ImportError:
    CdpClient = None  # type: ignore[misc, assignment]
    TransactionRequestEIP1559 = None  # type: ignore[misc, assignment]
    CDP_AVAILABLE = False


# ANSI color codes for terminal output
class Colors:
    """ANSI color codes for terminal output."""

    SUCCESS = "\033[92m"  # Green
    PENDING = "\033[93m"  # Yellow
    FAILED = "\033[91m"   # Red
    INFO = "\033[94m"     # Blue
    BOLD = "\033[1m"
    RESET = "\033[0m"


# Box drawing helper
BOX_WIDTH = 52


def _emoji_width(text: str) -> int:
    """Count extra width from emojis (they display as 2 chars but count as 1)."""
    # These specific emojis are known to be double-width in terminals
    # Note: ✓ and ✗ are NOT wide - they are simple unicode characters
    wide_emojis = {'⏳', '📡', '💰', '📇', '📦', '⏸', '🔄'}
    extra = 0
    for char in text:
        if char in wide_emojis:
            extra += 1
    return extra


def log_box(color: str, title: str, lines: list[str]) -> None:
    """Log a colored box with title and content lines."""
    top = f"{color}┌{'─' * BOX_WIDTH}┐{Colors.RESET}"
    bottom = f"{color}└{'─' * BOX_WIDTH}┘{Colors.RESET}"

    logging.info(top)
    # Title line - account for emoji width
    title_content = f"  {title}"
    emoji_extra = _emoji_width(title_content)
    title_padded = title_content.ljust(BOX_WIDTH - emoji_extra)
    logging.info(f"{color}│{Colors.BOLD}{title_padded}{Colors.RESET}{color}│{Colors.RESET}")
    # Content lines
    for line in lines:
        line_content = f"     {line}"
        emoji_extra = _emoji_width(line_content)
        line_padded = line_content.ljust(BOX_WIDTH - emoji_extra)
        logging.info(f"{color}│{line_padded}│{Colors.RESET}")
    logging.info(bottom)


# Network configurations with token contracts
# Structure: NETWORKS[chain][environment] = {cdp_id, chain_id, tokens}
NETWORKS = {
    "base": {
        "mainnet": {
            "cdp_id": "base-mainnet",
            "chain_id": 8453,
            "explorer": "https://basescan.org",
            "tokens": {
                "usdc": {"address": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913", "decimals": 6},
                "usdbc": {"address": "0xd9aAEc86B65D86f6A7B5B1b0c42FFA531710b6CA", "decimals": 6},
                "usdt": {"address": "0xfde4C96c8593536E31F229EA8f37b2ADa2699bb2", "decimals": 6},
                "dai": {"address": "0x50c5725949A6F0c72E6C4a641F24049A917DB0Cb", "decimals": 18},
                "weth": {"address": "0x4200000000000000000000000000000000000006", "decimals": 18},
            },
        },
        "testnet": {
            "cdp_id": "base-sepolia",
            "chain_id": 84532,
            "explorer": "https://sepolia.basescan.org",
            "tokens": {
                "usdc": {"address": "0x036CbD53842c5426634e7929541eC2318f3dCF7e", "decimals": 6},
            },
        },
    },
    "ethereum": {
        "mainnet": {
            "cdp_id": "ethereum-mainnet",
            "chain_id": 1,
            "explorer": "https://etherscan.io",
            "tokens": {
                "usdc": {"address": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48", "decimals": 6},
                "usdt": {"address": "0xdAC17F958D2ee523a2206206994597C13D831ec7", "decimals": 6},
                "dai": {"address": "0x6B175474E89094C44Da98b954EescdeCB5BE1e831", "decimals": 18},
            },
        },
        "testnet": {
            "cdp_id": "ethereum-sepolia",
            "chain_id": 11155111,
            "explorer": "https://sepolia.etherscan.io",
            "tokens": {
                "usdc": {"address": "0x1c7D4B196Cb0C7B01d743Fbc6116a902379C7238", "decimals": 6},
            },
        },
    },
    "arbitrum": {
        "mainnet": {
            "cdp_id": "arbitrum-mainnet",
            "chain_id": 42161,
            "explorer": "https://arbiscan.io",
            "tokens": {
                "usdc": {"address": "0xaf88d065e77c8cC2239327C5EDb3A432268e5831", "decimals": 6},
                "usdt": {"address": "0xFd086bC7CD5C481DCC9C85ebE478A1C0b69FCbb9", "decimals": 6},
            },
        },
    },
    "polygon": {
        "mainnet": {
            "cdp_id": "polygon-mainnet",
            "chain_id": 137,
            "explorer": "https://polygonscan.com",
            "tokens": {
                "usdc": {"address": "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359", "decimals": 6},
                "usdt": {"address": "0xc2132D05D31c914a87C6611C10748AEb04B58e8F", "decimals": 6},
            },
        },
    },
}


class CoinbasePaymentConfig(ActionConfig):
    """
    Configuration for Coinbase CDP v2 payment connector.

    This connector enables cryptocurrency payments using Coinbase Developer Platform.
    Supports multiple chains (Base, Ethereum, Arbitrum, Polygon) and tokens.

    Parameters
    ----------
    api_key_id : str
        CDP API Key ID from portal.cdp.coinbase.com
    api_key_secret : str
        CDP API Key Secret (base64 encoded)
    wallet_secret : str
        CDP Wallet Secret for transaction signing (PEM format)
    account_address : str
        EVM account address (0x...). Created automatically if not provided.
    chain : str
        Blockchain network: base, ethereum, arbitrum, polygon
    testnet : bool
        Use testnet (True) or mainnet (False)
    contacts : dict
        Named contacts mapping: {"john": "0x...", "coffee": "0x..."}
    max_amounts : dict
        Per-asset limits: {"usdc": 50.0, "eth": 0.01}
    """

    api_key_id: Optional[str] = Field(
        default=None,
        description="CDP API Key ID",
    )
    api_key_secret: Optional[str] = Field(
        default=None,
        description="CDP API Key Secret",
    )
    wallet_secret: Optional[str] = Field(
        default=None,
        description="CDP Wallet Secret for signing",
    )
    account_address: Optional[str] = Field(
        default=None,
        description="EVM account address (created if not provided)",
    )
    default_asset: str = Field(
        default="usdc",
        description="Default cryptocurrency (eth, usdc)",
    )
    destination_address: Optional[str] = Field(
        default=None,
        description="Default destination wallet address",
    )
    chain: str = Field(
        default="base",
        description="Blockchain network (base, ethereum, arbitrum, polygon)",
    )
    testnet: bool = Field(
        default=True,
        description="Use testnet (True) or mainnet (False)",
    )
    ha_url: Optional[str] = Field(
        default=None,
        description="Home Assistant URL for notifications",
    )
    ha_token: Optional[str] = Field(
        default=None,
        description="Home Assistant access token",
    )
    # Notification mode: "ha_event" (default), "webhook", "none"
    notification_mode: str = Field(
        default="ha_event",
        description="Notification mode: ha_event (default, modular), webhook, none",
    )
    webhook_url: Optional[str] = Field(
        default=None,
        description="Webhook URL for notifications (used when notification_mode=webhook)",
    )
    ha_event_prefix: str = Field(
        default="om1",
        description="Event prefix for HA events (e.g., om1_payment_completed)",
    )
    contacts: Optional[dict[str, str]] = Field(
        default=None,
        description="Named contacts mapping (name -> address)",
    )
    max_amounts: Optional[dict[str, float]] = Field(
        default=None,
        description="Maximum allowed payment per asset (e.g. {'usdc': 50, 'eth': 0.01})",
    )
    payment_cooldown: float = Field(
        default=30.0,
        description="Seconds before same payment can be repeated (duplicate protection)",
    )
    # Security settings
    blocked_addresses: Optional[list[str]] = Field(
        default=None,
        description="Blacklisted addresses - payments to these will be blocked",
    )
    whitelist_only: bool = Field(
        default=False,
        description="If true, only allow payments to addresses in contacts",
    )
    # Time restrictions
    allowed_hours: Optional[dict[str, int]] = Field(
        default=None,
        description="Restrict payments to certain hours (e.g. {'start': 8, 'end': 23})",
    )
    # Retry settings
    retry_on_failure: bool = Field(
        default=False,
        description="Automatically retry failed transactions",
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retry attempts",
    )
    # Gas settings (optional override)
    gas_settings: Optional[dict[str, float]] = Field(
        default=None,
        description="Gas settings: {'max_priority_fee': 1.5, 'max_fee': 50} in gwei",
    )
    # Custom token contracts (override defaults)
    token_contracts: Optional[dict[str, str]] = Field(
        default=None,
        description="Custom token contracts: {'dai': '0x...', 'weth': '0x...'}",
    )


def extract_payment_details(
        text: str,
) -> tuple[Optional[float], Optional[str], Optional[str]]:
    """Extract payment amount, asset, and address from text."""
    text_lower = text.lower()

    # Extract amount and asset - accept ANY token name (validation happens later)
    # This makes the system fully modular - no hardcoded token list
    amount_match = re.search(r"(\d+(?:\.\d+)?)\s*([a-z]{2,10})\b", text_lower)
    amount = float(amount_match.group(1)) if amount_match else None
    asset = amount_match.group(2) if amount_match else None

    # Extract address (0x... format)
    address_match = re.search(r"(0x[a-fA-F0-9]{40})", text)
    address = address_match.group(1) if address_match else None

    return amount, asset, address


def split_multi_payments(text: str) -> list[str]:
    """Split text into multiple payment commands."""
    # Split by common conjunctions
    parts = re.split(r'\s+and\s+|\s+also\s+|\s*,\s*and\s+|\s*,\s*', text.lower())
    # Filter parts that look like payment commands
    payment_parts = []
    for part in parts:
        part = part.strip()
        if part and any(word in part for word in ["send", "pay", "transfer"]):
            payment_parts.append(part)
        elif part and payment_parts:
            # Might be continuation like "5 usdc to john" after "send"
            # Check if it has amount
            if re.search(r'\d+(?:\.\d+)?\s*(eth|usdc|btc|sol)', part):
                payment_parts.append(f"send {part}")
    return payment_parts if payment_parts else [text]


class CoinbasePaymentConnector(ActionConnector[CoinbasePaymentConfig, CoinbasePaymentInput]):
    """
    Connector for sending cryptocurrency payments via Coinbase CDP v2.

    This connector processes natural language payment commands and executes
    them using the Coinbase Developer Platform API. Features include:

    - Multi-chain support: Base, Ethereum, Arbitrum, Polygon
    - Multi-token support: ETH, USDC, USDT, USDbC, DAI, WETH
    - Contact resolution: Say "send to john" instead of addresses
    - Multi-payment: "send 5 usdc to john and 3 eth to rent"
    - Safety limits: Per-asset max amounts, blocked addresses, whitelist mode
    - Home Assistant integration: Real-time transaction status updates

    Parameters
    ----------
    config : CoinbasePaymentConfig
        Configuration for the connector including CDP credentials.
    """

    _last_action: Optional[str] = None
    _last_status: str = ""
    _last_tx_hash: str = ""
    _last_amount: str = ""
    _last_payment_key: Optional[str] = None  # "amount|asset|dest" for smart duplicate detection
    _last_payment_time: float = 0  # timestamp of last payment

    def __init__(self, config: CoinbasePaymentConfig):
        super().__init__(config)

        self.io_provider = IOProvider()

        # Network configuration
        self.chain = config.chain.lower()
        self.is_testnet = config.testnet
        env = "testnet" if self.is_testnet else "mainnet"

        # Get network config from NETWORKS
        if self.chain not in NETWORKS:
            raise ValueError(f"Unsupported chain: {self.chain}. Supported: {list(NETWORKS.keys())}")
        if env not in NETWORKS[self.chain]:
            raise ValueError(f"No {env} config for {self.chain}")

        self.network_config = NETWORKS[self.chain][env]
        self.cdp_network = self.network_config["cdp_id"]
        self.explorer_url = self.network_config["explorer"]

        # Token contracts: network defaults + user overrides
        self.token_contracts = {}
        for token, info in self.network_config["tokens"].items():
            self.token_contracts[token] = info
        if config.token_contracts:
            for token, address in config.token_contracts.items():
                self.token_contracts[token] = {"address": address, "decimals": 6}  # default 6

        # Available tokens for this chain/environment (for validation and display)
        self.available_tokens = ["eth"] + list(self.token_contracts.keys())
        self.available_tokens_display = ", ".join(t.upper() for t in self.available_tokens)

        self.default_asset = config.default_asset
        self.destination_address = config.destination_address
        self.ha_url = config.ha_url
        self.ha_token = config.ha_token
        self.notification_mode = config.notification_mode
        self.webhook_url = config.webhook_url
        self.ha_event_prefix = config.ha_event_prefix
        self.contacts = config.contacts or {}
        self.max_amounts = config.max_amounts or {"usdc": 100.0, "eth": 0.1}  # defaults
        self.payment_cooldown = config.payment_cooldown
        # Security settings
        self.blocked_addresses = [a.lower() for a in (config.blocked_addresses or [])]
        self.whitelist_only = config.whitelist_only
        self.allowed_hours = config.allowed_hours
        # Retry settings
        self.retry_on_failure = config.retry_on_failure
        self.max_retries = config.max_retries
        # Gas settings
        self.gas_settings = config.gas_settings

        self._cdp_client: Optional[CdpClient] = None
        self._account_address: Optional[str] = config.account_address
        self._initialized = False
        self._session: Optional[aiohttp.ClientSession] = None

        # Get credentials from config or env
        self._api_key_id = config.api_key_id or os.environ.get("CDP_API_KEY_ID")
        self._api_key_secret = config.api_key_secret or os.environ.get("CDP_API_KEY_SECRET")
        self._wallet_secret = config.wallet_secret or os.environ.get("CDP_WALLET_SECRET")

        if not all([self._api_key_id, self._api_key_secret, self._wallet_secret]):
            logging.warning(
                f"{Colors.PENDING}CDP credentials not complete. "
                f"Need: api_key_id, api_key_secret, wallet_secret{Colors.RESET}"
            )
        elif not CDP_AVAILABLE:
            logging.warning(
                f"{Colors.PENDING}CDP SDK not installed. Run: pip install cdp-sdk{Colors.RESET}"
            )
        else:
            self._initialized = True
            account_display = self._account_address[:16] + "..." if self._account_address else "will create"
            env_label = "testnet" if self.is_testnet else "mainnet"
            log_box(
                Colors.SUCCESS,
                "✓ CDP PAYMENT READY",
                [
                    f"Chain: {self.chain} ({env_label})",
                    f"Account: {account_display}",
                    f"Tokens: {self.available_tokens_display}",
                ],
            )

    async def _get_cdp_client(self) -> Optional[CdpClient]:
        """Get or create CDP client."""
        if not self._initialized:
            return None

        if self._cdp_client is None:
            self._cdp_client = CdpClient(
                api_key_id=self._api_key_id,
                api_key_secret=self._api_key_secret,
                wallet_secret=self._wallet_secret,
            )
        return self._cdp_client

    async def _ensure_account(self) -> Optional[str]:
        """Ensure we have an account address, create if needed."""
        if self._account_address:
            return self._account_address

        cdp = await self._get_cdp_client()
        if not cdp:
            return None

        try:
            account = await cdp.evm.create_account()
            self._account_address = account.address
            logging.info(
                f"{Colors.SUCCESS}✓ Created CDP account: {self._account_address}{Colors.RESET}"
            )
            return self._account_address
        except Exception as e:
            logging.error(
                f"{Colors.FAILED}✗ Failed to create account: {e}{Colors.RESET}"
            )
            return None

    def _resolve_contact(self, text: str) -> tuple[Optional[str], bool]:
        """Resolve contact name to address from config.

        Returns
        -------
        tuple
            (address, has_unknown_name): address if found, and flag if unknown name detected
        """
        text_lower = text.lower()

        # Check each contact name (longer names first to match "coffee shop" before "coffee")
        for name in sorted(self.contacts.keys(), key=len, reverse=True):
            if name.lower() in text_lower:
                address = self.contacts[name]
                logging.info(f"{Colors.INFO}📇 Contact resolved: {name} → {address[:16]}...{Colors.RESET}")
                return (address, False)

        # Check if user mentioned a name we don't know
        # Patterns: "to <name>", "pay <name>", "send <name>"
        unknown_name_patterns = [
            r'\bto\s+([a-zA-Z]+)\b',  # "to john", "to faruk"
            r'\bpay\s+([a-zA-Z]+)(?:\s+\d|$)',  # "pay john 5" but not "pay 5"
        ]

        for pattern in unknown_name_patterns:
            match = re.search(pattern, text_lower)
            if match:
                potential_name = match.group(1)
                # Skip common words that aren't names
                skip_words = {'my', 'the', 'a', 'an', 'for', 'with', 'address', 'wallet', 'bill', 'rent', 'internet', 'coffee', 'electricity'}
                if potential_name not in skip_words and potential_name not in [c.lower() for c in self.contacts.keys()]:
                    logging.warning(f"{Colors.FAILED}⚠️ Unknown contact: '{potential_name}' - payment blocked{Colors.RESET}")
                    return (None, True)  # Unknown name detected

        return (None, False)  # No name mentioned, can use default

    def _parse_action(
            self, text: str
    ) -> tuple[str, Optional[float], Optional[str], Optional[str], Optional[str]]:
        """Parse action text and return (action_type, amount, asset, address, error)."""
        text_lower = text.lower()

        # Check for balance query
        if "balance" in text_lower:
            return ("balance", None, None, None, None)

        # Check for payment commands
        if any(word in text_lower for word in ["send", "pay", "transfer"]):
            amount, asset, address = extract_payment_details(text)

            # Payment command detected but no amount specified
            if amount is None:
                return ("error", None, None, None, "Please specify amount, e.g., 'pay internet 10 usdc'")

            if amount:
                # Validate amount
                if amount <= 0:
                    return ("error", None, None, None, "Invalid amount: must be positive")

                # Validate token is available
                check_asset = (asset or self.default_asset).lower()
                if check_asset not in self.available_tokens:
                    return ("error", None, None, None, f"{check_asset.upper()} not available. Use: {self.available_tokens_display}")

                # Check asset-specific max amount
                max_for_asset = self.max_amounts.get(check_asset)
                if max_for_asset and amount > max_for_asset:
                    return ("error", None, None, None, f"Amount {amount} {check_asset.upper()} exceeds limit ({max_for_asset})")

                # If no 0x address found, try to resolve contact name
                if not address:
                    resolved_address, has_unknown = self._resolve_contact(text)
                    if has_unknown:
                        return ("error", None, None, None, "Unknown contact - payment blocked")
                    address = resolved_address

                # Validate address format if provided
                if address and not re.match(r'^0x[a-fA-F0-9]{40}$', address):
                    return ("error", None, None, None, f"Invalid address format: {address[:20]}...")

                final_address = address or self.destination_address

                # Security: Check blocked addresses
                if final_address and final_address.lower() in self.blocked_addresses:
                    return ("error", None, None, None, f"Address is blocked: {final_address[:16]}...")

                # Security: Whitelist mode - only allow contacts
                if self.whitelist_only and final_address:
                    contact_addresses = [a.lower() for a in self.contacts.values()]
                    if final_address.lower() not in contact_addresses:
                        return ("error", None, None, None, "Whitelist mode: only contacts allowed")

                # Security: Time restrictions
                if self.allowed_hours:
                    from datetime import datetime
                    current_hour = datetime.now().hour
                    start = self.allowed_hours.get("start", 0)
                    end = self.allowed_hours.get("end", 24)
                    if not (start <= current_hour < end):
                        return ("error", None, None, None, f"Payments blocked outside {start}:00-{end}:00")

                return (
                    "send",
                    amount,
                    asset or self.default_asset,
                    final_address,
                    None,
                )

        # Check for idle
        if "idle" in text_lower or not text.strip():
            return ("idle", None, None, None, None)

        return ("unknown", None, None, None, None)

    async def connect(self, output_interface: CoinbasePaymentInput) -> None:
        """Execute the payment action."""
        action_text = output_interface.action

        # Skip duplicate actions
        if action_text == self._last_action:
            return
        self._last_action = action_text

        # Check for LLM error messages (incomplete/unknown commands)
        if action_text.lower().startswith("error:"):
            error_msg = action_text[6:].strip()  # Remove "error:" prefix
            log_box(
                Colors.PENDING,
                "⚠ COMMAND ISSUE",
                [error_msg[:48] if len(error_msg) > 48 else error_msg],
            )
            await self._update_ha_status(status=f"INFO: {error_msg[:30]}")
            return

        # Check for balance query first
        if "balance" in action_text.lower():
            await self._get_balance()
            return

        # Check for idle
        if "idle" in action_text.lower() or not action_text.strip():
            return

        # Split into multiple payments if "and" present
        payment_commands = split_multi_payments(action_text)
        total_payments = len(payment_commands)

        if total_payments > 1:
            logging.info(f"{Colors.INFO}📦 Multi-payment detected: {total_payments} transactions{Colors.RESET}")

        for idx, cmd in enumerate(payment_commands, 1):
            action_type, amount, asset, address, error = self._parse_action(cmd)

            # Handle validation errors
            if action_type == "error":
                self._last_status = "error"
                self._last_amount = error or "Unknown error"
                log_box(
                    Colors.FAILED,
                    "✗ PAYMENT BLOCKED",
                    [f"Reason: {error}"],
                )
                await self._update_ha_status(status=f"BLOCKED: {error}")
                continue

            if action_type != "send" or not amount:
                continue

            if not address:
                self._last_status = "error"
                self._last_amount = f"No destination: {cmd}"
                log_box(
                    Colors.FAILED,
                    "✗ PAYMENT BLOCKED",
                    ["Reason: No destination address"],
                )
                await self._update_ha_status(status="BLOCKED: No destination")
                continue

            # Smart duplicate detection
            payment_key = f"{amount}|{asset}|{address[:20]}"
            current_time = time.time()
            time_since_last = current_time - self._last_payment_time

            if payment_key == self._last_payment_key and time_since_last < self.payment_cooldown:
                remaining = int(self.payment_cooldown - time_since_last)
                logging.info(
                    f"{Colors.PENDING}⏸ Duplicate payment blocked. "
                    f"Wait {remaining}s or use different amount.{Colors.RESET}"
                )
                continue

            # Update AI Command Log
            order_text = f"send {amount} {asset or self.default_asset}"
            if total_payments > 1:
                order_text = f"[{idx}/{total_payments}] {order_text}"
            await self._update_ha_order(order_text)

            # Execute payment with index info
            await self._send_payment(
                amount,
                asset or self.default_asset,
                address,
                payment_idx=idx if total_payments > 1 else None,
                total_payments=total_payments if total_payments > 1 else None,
            )

            # Update duplicate tracking
            self._last_payment_key = payment_key
            self._last_payment_time = time.time()

        return

        self._last_status = "unknown_action"
        self._last_amount = action_text
        logging.warning(f"Unknown action: {action_text}")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _notify(self, event_type: str, data: dict) -> None:
        """
        Send notification based on notification_mode config.

        Modes:
        - ha_event: Fire HA event (default, modular - works with any HA setup)
        - webhook: POST to webhook_url
        - none: Do nothing
        """
        if self.notification_mode == "none":
            return

        if self.notification_mode == "webhook":
            if not self.webhook_url:
                return
            try:
                session = await self._get_session()
                async with session.post(
                    self.webhook_url,
                    json={"event_type": event_type, "data": data},
                    timeout=5,
                ) as resp:
                    if resp.status == 200:
                        logging.debug(f"Webhook notified: {event_type}")
            except Exception:
                pass
            return

        # ha_event mode (default) - requires ha_url and ha_token
        if not self.ha_url or not self.ha_token:
            return

        headers = {
            "Authorization": f"Bearer {self.ha_token}",
            "Content-Type": "application/json",
        }

        # Fire HA event - modular, works with any HA automation
        event_name = f"{self.ha_event_prefix}_{event_type}"
        url = f"{self.ha_url}/api/events/{event_name}"
        try:
            session = await self._get_session()
            async with session.post(
                url, headers=headers, json=data, timeout=5
            ) as resp:
                if resp.status == 200:
                    logging.debug(f"HA event fired: {event_name}")
        except Exception:
            pass

    async def _update_ha_status(
        self, status: str, amount: float = 0, tx_hash: str = ""
    ) -> None:
        """Update with payment status (wrapper for backward compatibility)."""
        data = {"status": status}
        if amount > 0:
            data["amount"] = amount
        if tx_hash:
            data["tx_hash"] = tx_hash

        # Determine event type from status prefix
        status_upper = status.upper()
        if status_upper.startswith("SUCCESS") or status_upper.startswith("COMPLETE"):
            event_type = "payment_completed"
        elif status_upper.startswith("FAILED") or status_upper.startswith("ERROR") or status_upper.startswith("BLOCKED"):
            event_type = "payment_failed"
        elif status_upper.startswith("PENDING") or status_upper.startswith("WALLET") or status_upper.startswith("BROADCAST") or status_upper.startswith("CONFIRMING"):
            event_type = "payment_pending"
        else:
            event_type = "payment_completed"  # Default

        await self._notify(event_type, data)

    async def _update_ha_balance(self, balance: str, asset: str) -> None:
        """Update with wallet balance (wrapper for backward compatibility)."""
        await self._notify("balance_updated", {"balance": balance, "asset": asset})

    async def _update_ha_order(self, command: str) -> None:
        """Update with voice command (wrapper for backward compatibility)."""
        await self._notify("order_received", {"command": command, "status": command})

    async def _get_balance(self) -> None:
        """Get wallet balance using CDP v2."""
        cdp = await self._get_cdp_client()
        if not cdp:
            self._last_status = "error"
            self._last_amount = "CDP not initialized"
            logging.error(f"{Colors.FAILED}✗ {self._last_amount}{Colors.RESET}")
            return

        account_address = await self._ensure_account()
        if not account_address:
            self._last_status = "error"
            self._last_amount = "No account available"
            logging.error(f"{Colors.FAILED}✗ {self._last_amount}{Colors.RESET}")
            return

        try:
            # Get token balances
            balances = await cdp.evm.list_token_balances(
                address=account_address,
                network=self.cdp_network,
            )

            # Find USDC or ETH balance
            balance_str = "0"
            asset_upper = self.default_asset.upper()

            for token_balance in balances.balances:
                token_symbol = token_balance.token.symbol if hasattr(token_balance.token, 'symbol') else str(token_balance.token)
                amount_val = token_balance.amount.amount if hasattr(token_balance.amount, 'amount') else token_balance.amount
                decimals = token_balance.amount.decimals if hasattr(token_balance.amount, 'decimals') else 18

                if self.default_asset.lower() == "usdc" and "USDC" in token_symbol:
                    balance_str = str(float(amount_val) / (10 ** decimals))
                    break
                elif self.default_asset.lower() == "eth" and "ETH" in token_symbol:
                    balance_str = str(float(amount_val) / (10 ** decimals))
                    break

            self._last_status = "success"
            self._last_amount = f"{balance_str} {asset_upper}"
            log_box(
                Colors.SUCCESS,
                "💰 WALLET BALANCE",
                [
                    f"Balance: {balance_str} {asset_upper}",
                    f"Account: {account_address[:16]}...{account_address[-8:]}",
                ],
            )

            self.io_provider.add_input(
                self.__class__.__name__,
                f"Wallet balance: {self._last_amount}",
                time.time(),
            )

            await self._update_ha_balance(balance_str, asset_upper)

        except Exception as e:
            logging.error(
                f"{Colors.FAILED}✗ Error getting balance: {e}{Colors.RESET}"
            )
            self._last_status = "error"
            self._last_amount = str(e)

    async def _send_payment(
            self, amount: float, asset: str, destination: str,
            payment_idx: Optional[int] = None, total_payments: Optional[int] = None
    ) -> None:
        """Send cryptocurrency payment using CDP v2."""
        # Format payment number for multi-payment
        pay_num = f" [{payment_idx}/{total_payments}]" if payment_idx else ""

        cdp = await self._get_cdp_client()
        if not cdp:
            self._last_status = "error"
            self._last_amount = "CDP not initialized"
            logging.error(f"{Colors.FAILED}✗ PAYMENT ERROR{pay_num}: {self._last_amount}{Colors.RESET}")
            await self._update_ha_status(status=f"ERROR{pay_num}: CDP not initialized")
            return

        account_address = await self._ensure_account()
        if not account_address:
            self._last_status = "error"
            self._last_amount = "No account available"
            logging.error(f"{Colors.FAILED}✗ PAYMENT ERROR{pay_num}: {self._last_amount}{Colors.RESET}")
            await self._update_ha_status(status=f"ERROR{pay_num}: No account")
            return

        try:
            # Step 1: WALLET - Accessing wallet
            await self._update_ha_status(status=f"WALLET{pay_num}: Preparing transaction", amount=amount)
            await asyncio.sleep(0.3)

            # Step 2: PENDING - Building transaction
            title = f"⏳ PAYMENT{pay_num} PENDING"
            env_label = "testnet" if self.is_testnet else "mainnet"
            log_box(
                Colors.PENDING,
                title,
                [
                    f"Amount: {amount} {asset.upper()}",
                    f"To: {destination[:20]}...{destination[-6:]}",
                    f"Chain: {self.chain} ({env_label})",
                ],
            )
            self._last_status = "pending"

            await self._update_ha_status(
                status=f"PENDING{pay_num}: Sending {amount} {asset.upper()}",
                amount=amount,
            )
            await asyncio.sleep(0.3)

            # Build gas parameters if configured
            gas_kwargs = {}
            if self.gas_settings:
                if "max_priority_fee" in self.gas_settings:
                    gas_kwargs["max_priority_fee_per_gas"] = int(self.gas_settings["max_priority_fee"] * 10**9)
                if "max_fee" in self.gas_settings:
                    gas_kwargs["max_fee_per_gas"] = int(self.gas_settings["max_fee"] * 10**9)

            # Retry logic
            max_attempts = self.max_retries if self.retry_on_failure else 1
            last_error = None

            for attempt in range(1, max_attempts + 1):
                try:
                    if attempt > 1:
                        logging.info(f"{Colors.PENDING}🔄 Retry attempt {attempt}/{max_attempts}...{Colors.RESET}")
                        await asyncio.sleep(2)  # Wait before retry

                    asset_lower = asset.lower()

                    if asset_lower == "eth":
                        # Native ETH transfer
                        amount_wei = int(amount * 10**18)
                        tx_hash = await cdp.evm.send_transaction(
                            address=account_address,
                            network=self.cdp_network,
                            transaction=TransactionRequestEIP1559(
                                to=destination,
                                value=amount_wei,
                                **gas_kwargs,
                            ),
                        )
                    else:
                        # ERC20 token transfer (USDC, USDT, USDbC, etc.)
                        token_info = self.token_contracts.get(asset_lower)
                        if not token_info:
                            raise ValueError(f"{asset.upper()} not available. Use: {self.available_tokens_display}")

                        token_address = token_info["address"]
                        decimals = token_info.get("decimals", 6)
                        amount_wei = int(amount * (10 ** decimals))

                        # Encode ERC20 transfer: transfer(address,uint256)
                        # Function selector: 0xa9059cbb
                        dest_padded = destination[2:].lower().zfill(64)
                        amount_hex = hex(amount_wei)[2:].zfill(64)
                        data = f"0xa9059cbb{dest_padded}{amount_hex}"

                        tx_hash = await cdp.evm.send_transaction(
                            address=account_address,
                            network=self.cdp_network,
                            transaction=TransactionRequestEIP1559(
                                to=token_address,
                                data=data,
                                value=0,
                                **gas_kwargs,
                            ),
                        )
                    break  # Success, exit retry loop

                except Exception as retry_error:
                    last_error = retry_error
                    if attempt < max_attempts:
                        logging.warning(f"{Colors.PENDING}⚠️ Attempt {attempt} failed: {retry_error}{Colors.RESET}")
                    else:
                        raise last_error  # Re-raise on final attempt

            # Step 3: BROADCAST - Transaction sent to network
            log_box(
                Colors.INFO,
                f"📡 BROADCASTING{pay_num}",
                ["Transaction sent to network..."],
            )
            self._last_status = "broadcast"
            await self._update_ha_status(status=f"BROADCAST{pay_num}: Sending to network", amount=amount)
            await asyncio.sleep(0.5)

            # Step 4: CONFIRMING - Waiting for confirmation
            await self._update_ha_status(status=f"CONFIRMING{pay_num}: Waiting for block", amount=amount)
            await asyncio.sleep(0.3)

            if tx_hash:
                self._last_status = "complete"
                self._last_tx_hash = str(tx_hash)
                self._last_amount = f"{amount} {asset.upper()}"

                tx_str = str(tx_hash)
                log_box(
                    Colors.SUCCESS,
                    f"✓ PAYMENT{pay_num} SUCCESS",
                    [
                        f"Amount: {amount} {asset.upper()}",
                        f"TX: {tx_str[:24]}...{tx_str[-8:]}",
                        "Status: Confirmed",
                    ],
                )

                self.io_provider.add_input(
                    self.__class__.__name__,
                    f"✓ Payment{pay_num} SUCCESS: {amount} {asset.upper()} | TX: {tx_hash}",
                    time.time(),
                )

                await self._update_ha_status(
                    status=f"SUCCESS{pay_num}: {amount} {asset.upper()} sent",
                    amount=amount,
                    tx_hash=str(tx_hash),
                )

                # Update balance after successful payment (only on last payment)
                if not payment_idx or payment_idx == total_payments:
                    await self._get_balance()

        except Exception as e:
            self._last_status = "failed"
            self._last_tx_hash = ""
            error_msg = str(e)
            self._last_amount = error_msg
            # Truncate error for display
            error_display = error_msg[:40] + "..." if len(error_msg) > 40 else error_msg
            log_box(
                Colors.FAILED,
                f"✗ PAYMENT{pay_num} FAILED",
                [
                    f"Error: {error_display}",
                    f"Amount: {amount} {asset.upper()}",
                ],
            )
            self.io_provider.add_input(
                self.__class__.__name__,
                f"✗ Payment{pay_num} FAILED: {error_msg}",
                time.time(),
            )
            await self._update_ha_status(status=f"FAILED{pay_num}: {error_msg[:30]}")
            # Delay so UI can capture FAILED status before next payment starts
            await asyncio.sleep(1.0)
