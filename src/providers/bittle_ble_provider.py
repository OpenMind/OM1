import asyncio
import logging
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Optional

from bleak import BleakClient, BleakScanner

NUS_TX_CHARACTERISTIC_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
NUS_RX_CHARACTERISTIC_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
DEFAULT_BITTLE_DEVICE_NAME = "BittleA6_SSP"


@dataclass(frozen=True)
class BittleBLESettings:
    """
    Connection settings for Petoi Bittle over Nordic UART Service.
    """

    device_address: Optional[str] = None
    device_name: Optional[str] = DEFAULT_BITTLE_DEVICE_NAME
    tx_characteristic_uuid: str = NUS_TX_CHARACTERISTIC_UUID
    rx_characteristic_uuid: str = NUS_RX_CHARACTERISTIC_UUID
    connect_timeout: float = 10.0
    write_with_response: bool = True
    command_suffix: str = ""
    simulate: bool = False


def _none_if_empty(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def bittle_settings_from_config(config: Any) -> BittleBLESettings:
    """
    Build Bittle BLE settings from an action connector config object.
    """
    return BittleBLESettings(
        device_address=_none_if_empty(getattr(config, "device_address", None)),
        device_name=_none_if_empty(getattr(config, "device_name", DEFAULT_BITTLE_DEVICE_NAME)),
        tx_characteristic_uuid=str(
            getattr(config, "tx_characteristic_uuid", NUS_TX_CHARACTERISTIC_UUID)
            or NUS_TX_CHARACTERISTIC_UUID
        ),
        rx_characteristic_uuid=str(
            getattr(config, "rx_characteristic_uuid", NUS_RX_CHARACTERISTIC_UUID)
            or NUS_RX_CHARACTERISTIC_UUID
        ),
        connect_timeout=float(getattr(config, "connect_timeout", 10.0)),
        write_with_response=bool(getattr(config, "write_with_response", True)),
        command_suffix=str(getattr(config, "command_suffix", "")),
        simulate=bool(getattr(config, "simulate", False)),
    )


class BittleBLEProvider:
    """
    Async BLE/NUS transport for Petoi Bittle's ASCII token protocol.
    """

    def __init__(self, settings: BittleBLESettings):
        self.settings = settings
        self._client: Optional[BleakClient] = None
        self._connect_lock: Optional[asyncio.Lock] = None
        self._connect_lock_loop: Optional[asyncio.AbstractEventLoop] = None
        self.received_messages: Deque[str] = deque(maxlen=100)
        self.sent_commands: Deque[str] = deque(maxlen=100)
        self.sent_payloads: Deque[bytes] = deque(maxlen=100)

    def _lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._connect_lock is None or self._connect_lock_loop is not loop:
            self._connect_lock = asyncio.Lock()
            self._connect_lock_loop = loop
        return self._connect_lock

    async def _resolve_device(self) -> Any:
        if self.settings.device_address:
            return self.settings.device_address

        if not self.settings.device_name:
            raise ValueError("Bittle BLE connection requires either device_address or device_name")

        logging.info("Scanning for Bittle BLE device named %s", self.settings.device_name)

        def matches(device: Any, advertisement_data: Any) -> bool:
            names = (
                getattr(device, "name", None),
                getattr(advertisement_data, "local_name", None),
            )
            return any(name == self.settings.device_name for name in names if name)

        device = await BleakScanner.find_device_by_filter(matches, timeout=self.settings.connect_timeout)
        if device is None:
            raise RuntimeError(f"Could not find Bittle BLE device named {self.settings.device_name!r}")
        return device

    async def connect(self) -> None:
        """
        Lazily connect to the Bittle BLE UART service and enable notifications.
        """
        if self.settings.simulate:
            return

        if self._client is not None and self._client.is_connected:
            return

        async with self._lock():
            if self._client is not None and self._client.is_connected:
                return

            device = await self._resolve_device()
            client = BleakClient(device, timeout=self.settings.connect_timeout)
            await client.connect()
            self._client = client

            try:
                await client.start_notify(self.settings.tx_characteristic_uuid, self._handle_notification)
            except Exception:
                logging.exception("Connected to Bittle, but failed to enable TX notifications")
                raise

            logging.info("Connected to Bittle BLE device")

    def _encode_command(self, command: str) -> bytes:
        wire_command = f"{command}{self.settings.command_suffix}"
        try:
            return wire_command.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError(f"Bittle commands must be ASCII: {command!r}") from exc

    async def send_command(self, command: str) -> None:
        """
        Send one Petoi ASCII token command to the RX characteristic.
        """
        if not command:
            raise ValueError("Bittle command cannot be empty")

        payload = self._encode_command(command)

        if self.settings.simulate:
            logging.info("Simulating Bittle BLE command: %s", command)
            self.sent_commands.append(command)
            self.sent_payloads.append(payload)
            return

        await self.connect()
        if self._client is None or not self._client.is_connected:
            raise RuntimeError("Bittle BLE client is not connected")

        await self._client.write_gatt_char(
            self.settings.rx_characteristic_uuid,
            payload,
            response=self.settings.write_with_response,
        )
        self.sent_commands.append(command)
        self.sent_payloads.append(payload)
        logging.info("Sent Bittle BLE command: %s", command)

    def _handle_notification(self, _sender: Any, data: bytearray) -> None:
        message = bytes(data).decode("ascii", errors="replace")
        self.received_messages.append(message)
        logging.debug("Bittle BLE notification: %s", message)

    async def disconnect(self) -> None:
        """
        Close the BLE connection if it is open.
        """
        if self._client is None:
            return

        try:
            if self._client.is_connected:
                try:
                    await self._client.stop_notify(self.settings.tx_characteristic_uuid)
                except Exception:
                    logging.debug("Ignoring Bittle notify stop failure", exc_info=True)
                await self._client.disconnect()
        finally:
            self._client = None


_provider_registry: dict[BittleBLESettings, BittleBLEProvider] = {}
_provider_registry_lock = threading.Lock()


def get_bittle_ble_provider(settings: BittleBLESettings) -> BittleBLEProvider:
    """
    Return one BLE provider per settings tuple so action connectors share a connection.
    """
    with _provider_registry_lock:
        provider = _provider_registry.get(settings)
        if provider is None:
            provider = BittleBLEProvider(settings)
            _provider_registry[settings] = provider
        return provider


def reset_bittle_ble_providers() -> None:
    """
    Clear provider registry. Intended for tests and process teardown paths.
    """
    with _provider_registry_lock:
        _provider_registry.clear()
