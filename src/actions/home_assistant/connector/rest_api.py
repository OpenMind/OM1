import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

import aiohttp
from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.home_assistant.interface import HomeAssistantInput
from providers.io_provider import IOProvider


class HomeAssistantConfig(ActionConfig):
    """
    Configuration for Home Assistant connector with optional CDP payment integration.

    Parameters
    ----------
    ha_url : str
        Home Assistant URL (e.g., "http://homeassistant.local:8123").
    access_token : str
        Long-lived access token from Home Assistant.
    switch_entity_id : Optional[str]
        Entity ID for switch control (e.g., "switch.tapo_plug").
    climate_entity_id : Optional[str]
        Entity ID for climate/thermostat control (e.g., "climate.lg_ac").
    light_entity_id : Optional[str]
        Entity ID for light control (e.g., "light.living_room").
    script_entity_id : Optional[str]
        Entity ID for order script (e.g., "script.place_order").
    cdp_api_key_id : Optional[str]
        CDP API Key ID for payment integration.
    cdp_api_key_secret : Optional[str]
        CDP API Key Secret for payment integration.
    cdp_wallet_secret : Optional[str]
        CDP Wallet Secret for signing transactions.
    cdp_account_address : Optional[str]
        CDP wallet address.
    payment_destination : Optional[str]
        Default payment destination address.
    default_payment_asset : str
        Default cryptocurrency for payments (default: usdc).
    """

    ha_url: str = Field(
        default="http://homeassistant.local:8123",
        description="Home Assistant URL",
    )
    access_token: Optional[str] = Field(
        default=None,
        description="Long-lived access token from Home Assistant",
    )
    switch_entity_id: Optional[str] = Field(
        default=None,
        description="Entity ID for switch control",
    )
    climate_entity_id: Optional[str] = Field(
        default=None,
        description="Entity ID for climate/thermostat control",
    )
    light_entity_id: Optional[str] = Field(
        default=None,
        description="Entity ID for light control",
    )
    script_entity_id: Optional[str] = Field(
        default=None,
        description="Entity ID for order placement script",
    )
    # CDP Payment Integration
    cdp_api_key_id: Optional[str] = Field(
        default=None,
        description="CDP API Key ID for payment integration",
    )
    cdp_api_key_secret: Optional[str] = Field(
        default=None,
        description="CDP API Key Secret for payment integration",
    )
    cdp_wallet_secret: Optional[str] = Field(
        default=None,
        description="CDP Wallet Secret for signing transactions",
    )
    cdp_account_address: Optional[str] = Field(
        default=None,
        description="CDP wallet address",
    )
    payment_destination: Optional[str] = Field(
        default=None,
        description="Default payment destination address",
    )
    default_payment_asset: str = Field(
        default="usdc",
        description="Default cryptocurrency for payments",
    )
    cdp_chain: str = Field(
        default="base",
        description="Blockchain network for CDP payments",
    )
    cdp_testnet: bool = Field(
        default=True,
        description="Use testnet for CDP payments",
    )


def extract_temperature(text: str) -> Optional[float]:
    """Extract temperature value from text like 'set temperature 24' or '24 degrees'."""
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:degrees?|celsius|c)?", text.lower())
    if match:
        temp = float(match.group(1))
        if 16 <= temp <= 30:
            return temp
    return None


def extract_order_details(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract order details from text and return parsed components.

    Examples: 'place order coffee 5 usdc', 'order pizza 10 usdc'
    Returns: {'item': 'coffee', 'amount': 5.0, 'asset': 'usdc'}
    """
    # Pattern: place order <item> <amount> <asset>
    pattern = r"(?:place\s+)?order\s+(\w+)\s+(\d+(?:\.\d+)?)\s*(usdc|eth|usdt|dai)?"
    match = re.search(pattern, text.lower())
    if match:
        item = match.group(1)
        amount = float(match.group(2))
        asset = match.group(3) or "usdc"
        return {"item": item, "amount": amount, "asset": asset}
    return None


class HomeAssistantConnector(ActionConnector[HomeAssistantConfig, HomeAssistantInput]):
    """
    Connector for controlling smart home devices and placing orders via Home Assistant.
    Integrates with Coinbase CDP for crypto payments on order placement.
    """

    _last_action: Optional[str] = None
    _payment_connector: Optional[Any] = None

    def __init__(self, config: HomeAssistantConfig):
        super().__init__(config)

        self.io_provider = IOProvider()

        self.ha_url = config.ha_url
        self.access_token = config.access_token
        self.switch_entity_id = config.switch_entity_id
        self.climate_entity_id = config.climate_entity_id
        self.light_entity_id = config.light_entity_id
        self.script_entity_id = config.script_entity_id

        # CDP Payment config
        self.cdp_enabled = bool(
            config.cdp_api_key_id
            and config.cdp_api_key_secret
            and config.cdp_wallet_secret
        )
        self.cdp_config = config
        self.payment_destination = config.payment_destination
        self.default_payment_asset = config.default_payment_asset

        self._session: Optional[aiohttp.ClientSession] = None

        if not self.access_token:
            logging.warning(
                "HomeAssistant: No access_token provided. "
                "Create one in Home Assistant Profile -> Long-Lived Access Tokens"
            )

        if self.cdp_enabled:
            logging.info(
                f"\033[94mHomeAssistant: Initialized with CDP payment "
                f"({self.ha_url})\033[0m"
            )
            self._init_payment_connector()
        else:
            logging.info(
                f"\033[94mHomeAssistant: Initialized without payment "
                f"({self.ha_url})\033[0m"
            )

    def _init_payment_connector(self) -> None:
        """Initialize CDP payment connector if configured."""
        try:
            from actions.coinbase_payment.connector.cdp_api import (
                CoinbasePaymentConfig,
                CoinbasePaymentConnector,
            )

            payment_config = CoinbasePaymentConfig(
                api_key_id=self.cdp_config.cdp_api_key_id,
                api_key_secret=self.cdp_config.cdp_api_key_secret,
                wallet_secret=self.cdp_config.cdp_wallet_secret,
                account_address=self.cdp_config.cdp_account_address,
                destination_address=self.payment_destination,
                default_asset=self.default_payment_asset,
                chain=self.cdp_config.cdp_chain,
                testnet=self.cdp_config.cdp_testnet,
                # Pass HA config for notifications
                ha_url=self.ha_url,
                ha_token=self.access_token,
                notification_mode="ha_event",  # Use modular HA events
            )
            self._payment_connector = CoinbasePaymentConnector(payment_config)
            logging.info("\033[92mHomeAssistant: CDP payment connector ready\033[0m")
        except ImportError:
            logging.warning(
                "HomeAssistant: coinbase_payment module not found. "
                "Payment integration disabled."
            )
            self.cdp_enabled = False
        except Exception as e:
            logging.error(f"HomeAssistant: Failed to init payment connector: {e}")
            self.cdp_enabled = False

    def _get_headers(self) -> Dict[str, str]:
        """Generate headers for Home Assistant API requests."""
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _call_service(
        self,
        domain: str,
        service: str,
        entity_id: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Call a Home Assistant service."""
        current_state = await self.get_entity_state(entity_id)
        if current_state:
            if domain == "switch":
                if service == "turn_off" and current_state == "off":
                    logging.info(
                        f"\033[90mHomeAssistant: {entity_id} already off\033[0m"
                    )
                    return True
                if service == "turn_on" and current_state == "on":
                    logging.info(
                        f"\033[90mHomeAssistant: {entity_id} already on\033[0m"
                    )
                    return True
            if domain == "climate" and service == "set_hvac_mode":
                desired_mode = data.get("hvac_mode") if data else None
                if desired_mode and current_state == desired_mode:
                    logging.info(
                        f"\033[90mHomeAssistant: {entity_id} already {desired_mode}\033[0m"
                    )
                    return True

        url = f"{self.ha_url}/api/services/{domain}/{service}"
        session = await self._get_session()

        payload = {"entity_id": entity_id}
        if data:
            payload.update(data)

        try:
            async with session.post(
                url, headers=self._get_headers(), json=payload, timeout=10
            ) as resp:
                if resp.status == 200:
                    color = {"switch": "95", "climate": "96", "light": "93"}.get(
                        domain, "96"
                    )
                    logging.info(
                        f"\033[{color}mHomeAssistant: {domain}.{service} -> "
                        f"{entity_id}\033[0m"
                    )
                    return True
                else:
                    text = await resp.text()
                    logging.error(
                        f"\033[91mHomeAssistant: API error {resp.status}: {text}\033[0m"
                    )
        except asyncio.TimeoutError:
            logging.error("\033[91mHomeAssistant: Request timed out\033[0m")
        except aiohttp.ClientError as e:
            logging.error(f"\033[91mHomeAssistant: Connection error: {e}\033[0m")
        except Exception as e:
            logging.error(f"\033[91mHomeAssistant: Error: {e}\033[0m")

        return False

    async def _call_script(
        self, script_entity_id: str, variables: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Call a Home Assistant script.

        Parameters
        ----------
        script_entity_id : str
            Script entity ID (e.g., "script.place_order")
        variables : Optional[Dict[str, Any]]
            Variables to pass to the script
        """
        if not script_entity_id:
            logging.warning("HomeAssistant: No script_entity_id configured")
            return False

        # Extract script name from entity_id
        script_name = script_entity_id.replace("script.", "")
        url = f"{self.ha_url}/api/services/script/{script_name}"
        session = await self._get_session()

        payload = variables or {}

        try:
            async with session.post(
                url, headers=self._get_headers(), json=payload, timeout=15
            ) as resp:
                if resp.status == 200:
                    logging.info(
                        f"\033[95mHomeAssistant: Script executed -> "
                        f"{script_entity_id}\033[0m"
                    )
                    return True
                else:
                    text = await resp.text()
                    logging.error(
                        f"\033[91mHomeAssistant: Script error {resp.status}: "
                        f"{text}\033[0m"
                    )
        except asyncio.TimeoutError:
            logging.error("\033[91mHomeAssistant: Script request timed out\033[0m")
        except Exception as e:
            logging.error(f"\033[91mHomeAssistant: Script error: {e}\033[0m")

        return False

    async def _process_payment(
        self, amount: float, asset: str, item: str
    ) -> Optional[str]:
        """
        Process payment via CDP.

        Parameters
        ----------
        amount : float
            Payment amount
        asset : str
            Cryptocurrency (usdc, eth, etc.)
        item : str
            Order item description

        Returns
        -------
        Optional[str]
            Transaction hash if successful, None otherwise
        """
        if not self.cdp_enabled or not self._payment_connector:
            logging.warning("HomeAssistant: CDP payment not configured")
            return None

        try:
            from actions.coinbase_payment.interface import CoinbasePaymentInput

            # Create payment command
            payment_command = f"send {amount} {asset}"
            payment_input = CoinbasePaymentInput(action=payment_command)

            logging.info(
                f"\033[96mHomeAssistant: Processing payment for {item}: "
                f"{amount} {asset.upper()}\033[0m"
            )

            # Execute payment
            result = await self._payment_connector.connect(payment_input)

            if result and hasattr(result, "tx_hash") and result.tx_hash:
                logging.info(
                    f"\033[92mHomeAssistant: Payment successful! "
                    f"TX: {result.tx_hash[:16]}...\033[0m"
                )
                return result.tx_hash

        except Exception as e:
            logging.error(f"\033[91mHomeAssistant: Payment failed: {e}\033[0m")

        return None

    def _parse_action(self, action_text: str) -> List[Dict[str, Any]]:
        """Parse action text and return list of commands."""
        text = action_text.lower().strip()
        commands = []

        if text in ["idle", "nothing", "do nothing", "no action"]:
            return []

        # Order/payment commands (check first - highest priority)
        order_details = extract_order_details(text)
        if order_details:
            commands.append(
                {
                    "type": "order",
                    "item": order_details["item"],
                    "amount": order_details["amount"],
                    "asset": order_details["asset"],
                }
            )
            return commands  # Order is exclusive

        # Switch/plug commands
        if self.switch_entity_id:
            if any(
                x in text
                for x in [
                    "turn on switch",
                    "switch on",
                    "plug on",
                    "turn on plug",
                    "turn on fan",
                    "fan on",
                ]
            ):
                commands.append(
                    {
                        "type": "service",
                        "domain": "switch",
                        "service": "turn_on",
                        "entity_id": self.switch_entity_id,
                    }
                )
            elif any(
                x in text
                for x in [
                    "turn off switch",
                    "switch off",
                    "plug off",
                    "turn off plug",
                    "turn off fan",
                    "fan off",
                ]
            ):
                commands.append(
                    {
                        "type": "service",
                        "domain": "switch",
                        "service": "turn_off",
                        "entity_id": self.switch_entity_id,
                    }
                )

        # Light commands
        if self.light_entity_id:
            if any(
                x in text
                for x in ["turn on light", "light on", "turn on lamp", "lamp on"]
            ):
                commands.append(
                    {
                        "type": "service",
                        "domain": "light",
                        "service": "turn_on",
                        "entity_id": self.light_entity_id,
                    }
                )
            elif any(
                x in text
                for x in ["turn off light", "light off", "turn off lamp", "lamp off"]
            ):
                commands.append(
                    {
                        "type": "service",
                        "domain": "light",
                        "service": "turn_off",
                        "entity_id": self.light_entity_id,
                    }
                )

        # Climate/thermostat commands
        if self.climate_entity_id:
            temp = extract_temperature(text)
            if temp is not None:
                commands.append(
                    {
                        "type": "service",
                        "domain": "climate",
                        "service": "set_temperature",
                        "entity_id": self.climate_entity_id,
                        "data": {"temperature": temp},
                    }
                )

            if any(
                x in text
                for x in ["turn on ac", "ac on", "turn on climate", "turn on hvac"]
            ):
                commands.append(
                    {
                        "type": "service",
                        "domain": "climate",
                        "service": "set_temperature",
                        "entity_id": self.climate_entity_id,
                        "data": {"temperature": 24},
                    }
                )
            elif any(
                x in text
                for x in ["turn off ac", "ac off", "turn off climate", "turn off hvac"]
            ):
                commands.append(
                    {
                        "type": "service",
                        "domain": "climate",
                        "service": "set_hvac_mode",
                        "entity_id": self.climate_entity_id,
                        "data": {"hvac_mode": "off"},
                    }
                )
            elif any(x in text for x in ["cool mode", "cooling", "set to cool"]):
                commands.append(
                    {
                        "type": "service",
                        "domain": "climate",
                        "service": "set_hvac_mode",
                        "entity_id": self.climate_entity_id,
                        "data": {"hvac_mode": "cool"},
                    }
                )
            elif any(x in text for x in ["heat mode", "heating", "set to heat"]):
                commands.append(
                    {
                        "type": "service",
                        "domain": "climate",
                        "service": "set_hvac_mode",
                        "entity_id": self.climate_entity_id,
                        "data": {"hvac_mode": "heat"},
                    }
                )

        return commands

    async def connect(self, output_interface: HomeAssistantInput) -> None:
        """
        Execute Home Assistant actions based on LLM decision.
        For order commands, triggers script and processes payment via CDP.
        """
        action_text = output_interface.action

        if not action_text:
            return

        normalized = action_text.lower().strip()

        if normalized in ["idle", "nothing", "do nothing", "no action"]:
            logging.info("\033[93mHomeAssistant: LLM decided -> idle\033[0m")
            return

        if normalized == HomeAssistantConnector._last_action:
            logging.info("\033[90mHomeAssistant: Same action, skipping\033[0m")
            return

        logging.info(f"\033[92mHomeAssistant: LLM decided -> {normalized}\033[0m")

        commands = self._parse_action(normalized)

        for cmd in commands:
            if cmd["type"] == "order":
                # Full workflow: Order -> HA Script -> CDP Payment -> Confirmation
                item = cmd["item"]
                amount = cmd["amount"]
                asset = cmd["asset"]

                logging.info(
                    f"\033[96m{'='*50}\033[0m\n"
                    f"\033[96m  ORDER WORKFLOW STARTED\033[0m\n"
                    f"\033[96m  Item: {item}\033[0m\n"
                    f"\033[96m  Amount: {amount} {asset.upper()}\033[0m\n"
                    f"\033[96m{'='*50}\033[0m"
                )

                # Step 1: Call Home Assistant script (REQUIRED for orders)
                if not self.script_entity_id:
                    logging.error(
                        "\033[91m" + "=" * 50 + "\033[0m\n"
                        "\033[91m  ORDER FAILED: script_entity_id not configured\033[0m\n"
                        "\033[91m  Please set script_entity_id in config\033[0m\n"
                        "\033[91m  Example: script.place_order\033[0m\n"
                        "\033[91m" + "=" * 50 + "\033[0m"
                    )
                    return

                script_success = await self._call_script(
                    self.script_entity_id,
                    {"item": item, "amount": amount, "asset": asset},
                )
                if not script_success:
                    logging.error(
                        "\033[91m" + "=" * 50 + "\033[0m\n"
                        "\033[91m  ORDER FAILED: Home Assistant script error\033[0m\n"
                        f"\033[91m  Script: {self.script_entity_id}\033[0m\n"
                        "\033[91m  Check that script exists in HA\033[0m\n"
                        "\033[91m" + "=" * 50 + "\033[0m"
                    )
                    return

                logging.info(
                    "\033[92mHomeAssistant: [1/2] Order placed in HA\033[0m"
                )

                # Step 2: Process payment via CDP (only if HA script succeeded)
                if self.cdp_enabled:
                    tx_hash = await self._process_payment(amount, asset, item)
                    if tx_hash:
                        logging.info(
                            f"\033[92m{'='*50}\033[0m\n"
                            f"\033[92m  ORDER COMPLETE!\033[0m\n"
                            f"\033[92m  Item: {item}\033[0m\n"
                            f"\033[92m  Payment: {amount} {asset.upper()}\033[0m\n"
                            f"\033[92m  TX: {tx_hash[:20]}...\033[0m\n"
                            f"\033[92m{'='*50}\033[0m"
                        )
                    else:
                        logging.error(
                            "\033[91mHomeAssistant: [2/2] Payment failed\033[0m"
                        )
                else:
                    logging.warning(
                        "\033[93mHomeAssistant: [2/2] CDP not configured, "
                        "skipping payment\033[0m"
                    )

            elif cmd["type"] == "service":
                await self._call_service(
                    domain=cmd["domain"],
                    service=cmd["service"],
                    entity_id=cmd["entity_id"],
                    data=cmd.get("data"),
                )

        HomeAssistantConnector._last_action = normalized

    async def get_entity_state(self, entity_id: str) -> Optional[str]:
        """Get current state of a specific entity."""
        url = f"{self.ha_url}/api/states/{entity_id}"
        session = await self._get_session()

        try:
            async with session.get(url, headers=self._get_headers(), timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("state")
        except Exception as e:
            logging.debug(f"HomeAssistant: Error getting state for {entity_id}: {e}")

        return None

    def __del__(self):
        """Cleanup session on destruction."""
        if self._session and not self._session.closed:
            try:
                asyncio.get_event_loop().create_task(self._session.close())
            except RuntimeError:
                pass
