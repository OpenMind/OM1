import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inputs.base import Message, SensorConfig
from inputs.plugins.wallet_ethereum import WalletEthereum


def test_initialization_success():
    """Test successful initialization."""
    mock_web3 = MagicMock()
    mock_web3.is_connected.return_value = True

    with (
        patch("inputs.plugins.wallet_ethereum.Web3", return_value=mock_web3),
        patch("inputs.plugins.wallet_ethereum.IOProvider"),
    ):
        sensor = WalletEthereum(config=SensorConfig())

        assert sensor.ETH_balance == 0
        assert sensor.ETH_balance_previous == 0
        assert sensor.messages == []


def test_initialization_connection_failure():
    """Test initialization when Web3 connection fails."""
    mock_web3 = MagicMock()
    mock_web3.is_connected.return_value = False

    with (
        patch("inputs.plugins.wallet_ethereum.Web3", return_value=mock_web3),
        patch("inputs.plugins.wallet_ethereum.IOProvider"),
    ):
        with pytest.raises(Exception, match="Failed to connect to Ethereum"):
            WalletEthereum(config=SensorConfig())


def test_initialization_with_custom_address():
    """Test initialization with custom ETH address."""
    mock_web3 = MagicMock()
    mock_web3.is_connected.return_value = True

    custom_address = "0x1234567890123456789012345678901234567890"
    env = {"ETH_ADDRESS": custom_address}

    with (
        patch.dict(os.environ, env, clear=False),
        patch("inputs.plugins.wallet_ethereum.Web3", return_value=mock_web3),
        patch("inputs.plugins.wallet_ethereum.IOProvider"),
    ):
        sensor = WalletEthereum(config=SensorConfig())

        assert sensor.ACCOUNT_ADDRESS == custom_address


@pytest.mark.asyncio
async def test_poll_success():
    """Test successful polling."""
    mock_web3 = MagicMock()
    mock_web3.is_connected.return_value = True
    mock_web3.eth.block_number = 12345
    mock_web3.eth.get_balance.return_value = 2000000000000000000  # 2 ETH in wei
    mock_web3.from_wei.return_value = 2.0

    with (
        patch("inputs.plugins.wallet_ethereum.Web3", return_value=mock_web3),
        patch("inputs.plugins.wallet_ethereum.IOProvider"),
    ):
        sensor = WalletEthereum(config=SensorConfig())
        sensor.ETH_balance_previous = 1.0

        with (
            patch("inputs.plugins.wallet_ethereum.asyncio.sleep", new=AsyncMock()),
            patch("inputs.plugins.wallet_ethereum.random.randint", return_value=5),
        ):
            result = await sensor._poll()

            assert result is not None
            assert len(result) == 2
            assert result[0] == 2.0  # Current balance


@pytest.mark.asyncio
async def test_poll_with_exception():
    """Test polling with exception."""
    mock_web3 = MagicMock()
    mock_web3.is_connected.return_value = True
    mock_web3.eth.block_number = MagicMock(side_effect=Exception("Network error"))

    with (
        patch("inputs.plugins.wallet_ethereum.Web3", return_value=mock_web3),
        patch("inputs.plugins.wallet_ethereum.IOProvider"),
    ):
        sensor = WalletEthereum(config=SensorConfig())

        with patch("inputs.plugins.wallet_ethereum.asyncio.sleep", new=AsyncMock()):
            result = await sensor._poll()

            assert len(result) == 2


@pytest.mark.asyncio
async def test_raw_to_text_with_positive_change():
    """Test _raw_to_text with positive balance change."""
    mock_web3 = MagicMock()
    mock_web3.is_connected.return_value = True

    with (
        patch("inputs.plugins.wallet_ethereum.Web3", return_value=mock_web3),
        patch("inputs.plugins.wallet_ethereum.IOProvider"),
    ):
        sensor = WalletEthereum(config=SensorConfig())

        with patch("inputs.plugins.wallet_ethereum.time.time", return_value=1234.0):
            result = await sensor._raw_to_text([2.5, 0.5])

            assert result is not None
            assert result.timestamp == 1234.0
            assert "0.5" in result.message or "0.50" in result.message


@pytest.mark.asyncio
async def test_raw_to_text_with_zero_change():
    """Test _raw_to_text with zero balance change."""
    mock_web3 = MagicMock()
    mock_web3.is_connected.return_value = True

    with (
        patch("inputs.plugins.wallet_ethereum.Web3", return_value=mock_web3),
        patch("inputs.plugins.wallet_ethereum.IOProvider"),
    ):
        sensor = WalletEthereum(config=SensorConfig())

        result = await sensor._raw_to_text([2.0, 0.0])

        assert result is None


def test_formatted_latest_buffer_with_messages():
    """Test formatted_latest_buffer with messages."""
    mock_web3 = MagicMock()
    mock_web3.is_connected.return_value = True

    with (
        patch("inputs.plugins.wallet_ethereum.Web3", return_value=mock_web3),
        patch("inputs.plugins.wallet_ethereum.IOProvider"),
    ):
        sensor = WalletEthereum(config=SensorConfig())
        sensor.io_provider = MagicMock()

        sensor.messages = [
            Message(timestamp=1000.0, message="0.5"),
            Message(timestamp=1001.0, message="0.3"),
        ]

        result = sensor.formatted_latest_buffer()

        assert result is not None
        assert "ETH" in result or "eth" in result.lower()
        sensor.io_provider.add_input.assert_called_once()
        assert len(sensor.messages) == 0


def test_formatted_latest_buffer_empty():
    """Test formatted_latest_buffer with empty buffer."""
    mock_web3 = MagicMock()
    mock_web3.is_connected.return_value = True

    with (
        patch("inputs.plugins.wallet_ethereum.Web3", return_value=mock_web3),
        patch("inputs.plugins.wallet_ethereum.IOProvider"),
    ):
        sensor = WalletEthereum(config=SensorConfig())

        result = sensor.formatted_latest_buffer()
        assert result is None
