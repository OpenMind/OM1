import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inputs.base import Message, SensorConfig
from inputs.plugins.wallet_solana import LAMPORTS_PER_SOL, WalletSolana

VALID_SOLANA_ADDRESS = "11111111111111111111111111111111"


def _make_sensor(env_overrides=None):
    """Helper to create a WalletSolana with mocked dependencies."""
    mock_client = MagicMock()
    mock_balance_resp = MagicMock()
    mock_balance_resp.value = 0
    mock_client.get_balance.return_value = mock_balance_resp

    env = {"SOLANA_ADDRESS": VALID_SOLANA_ADDRESS}
    if env_overrides:
        env.update(env_overrides)

    with (
        patch.dict(os.environ, env, clear=False),
        patch("inputs.plugins.wallet_solana.Client", return_value=mock_client),
        patch("inputs.plugins.wallet_solana.IOProvider"),
    ):
        sensor = WalletSolana(config=SensorConfig())

    sensor._mock_client = mock_client
    return sensor


# --- Init tests ---


def test_initialization_success():
    """Test successful initialization with valid address."""
    sensor = _make_sensor()

    assert sensor.SOL_balance == 0
    assert sensor.SOL_balance_previous == 0
    assert sensor.balance_change == 0
    assert sensor.messages == []
    assert sensor.ACCOUNT_ADDRESS == VALID_SOLANA_ADDRESS


def test_initialization_missing_address():
    """Test initialization fails when SOLANA_ADDRESS is not set."""
    mock_client = MagicMock()

    with (
        patch.dict(os.environ, {}, clear=True),
        patch("inputs.plugins.wallet_solana.Client", return_value=mock_client),
        patch("inputs.plugins.wallet_solana.IOProvider"),
    ):
        with pytest.raises(
            ValueError, match="SOLANA_ADDRESS environment variable is required"
        ):
            WalletSolana(config=SensorConfig())


def test_initialization_invalid_address():
    """Test initialization fails with invalid Solana address."""
    mock_client = MagicMock()
    env = {"SOLANA_ADDRESS": "not-a-valid-address"}

    with (
        patch.dict(os.environ, env, clear=False),
        patch("inputs.plugins.wallet_solana.Client", return_value=mock_client),
        patch("inputs.plugins.wallet_solana.IOProvider"),
    ):
        with pytest.raises(ValueError, match="Invalid Solana address"):
            WalletSolana(config=SensorConfig())


def test_initialization_rpc_connection_failure():
    """Test initialization fails when RPC connection fails."""
    mock_client = MagicMock()
    mock_client.get_balance.side_effect = Exception("Connection refused")

    env = {"SOLANA_ADDRESS": VALID_SOLANA_ADDRESS}

    with (
        patch.dict(os.environ, env, clear=False),
        patch("inputs.plugins.wallet_solana.Client", return_value=mock_client),
        patch("inputs.plugins.wallet_solana.IOProvider"),
    ):
        with pytest.raises(Exception, match="Failed to connect to Solana RPC"):
            WalletSolana(config=SensorConfig())


def test_initialization_custom_rpc_url():
    """Test initialization with custom RPC URL."""
    custom_url = "https://api.mainnet-beta.solana.com"

    mock_client_cls = MagicMock()
    mock_client = MagicMock()
    mock_balance_resp = MagicMock()
    mock_balance_resp.value = 0
    mock_client.get_balance.return_value = mock_balance_resp
    mock_client_cls.return_value = mock_client

    env = {
        "SOLANA_ADDRESS": VALID_SOLANA_ADDRESS,
        "SOLANA_RPC_URL": custom_url,
    }

    with (
        patch.dict(os.environ, env, clear=False),
        patch("inputs.plugins.wallet_solana.Client", mock_client_cls),
        patch("inputs.plugins.wallet_solana.IOProvider"),
    ):
        sensor = WalletSolana(config=SensorConfig())

        assert sensor.RPC_URL == custom_url
        mock_client_cls.assert_called_once_with(custom_url)


# --- Poll tests ---


@pytest.mark.asyncio
async def test_poll_success():
    """Test successful polling with balance change."""
    sensor = _make_sensor()
    sensor.SOL_balance_previous = 1.0

    mock_balance_resp = MagicMock()
    mock_balance_resp.value = 2 * LAMPORTS_PER_SOL  # 2 SOL in lamports
    sensor.client = MagicMock()
    sensor.client.get_balance.return_value = mock_balance_resp

    with patch("inputs.plugins.wallet_solana.asyncio.sleep", new=AsyncMock()):
        result = await sensor._poll()

    assert len(result) == 2
    assert result[0] == 2.0
    assert result[1] == 1.0  # 2.0 - 1.0 previous


@pytest.mark.asyncio
async def test_poll_rpc_error():
    """Test polling handles RPC errors gracefully."""
    sensor = _make_sensor()
    sensor.client = MagicMock()
    sensor.client.get_balance.side_effect = Exception("RPC timeout")

    with patch("inputs.plugins.wallet_solana.asyncio.sleep", new=AsyncMock()):
        result = await sensor._poll()

    assert len(result) == 2
    assert result[0] == 0  # Falls back to previous values
    assert result[1] == 0


# --- raw_to_text tests ---


@pytest.mark.asyncio
async def test_raw_to_text_positive_change():
    """Test _raw_to_text with positive balance change creates message."""
    sensor = _make_sensor()

    with patch("inputs.plugins.wallet_solana.time.time", return_value=1234.0):
        result = await sensor._raw_to_text([2.5, 0.5])

    assert result is not None
    assert result.timestamp == 1234.0
    assert "0.5" in result.message
    assert "SOL" in result.message


@pytest.mark.asyncio
async def test_raw_to_text_zero_change():
    """Test _raw_to_text with zero balance change returns None."""
    sensor = _make_sensor()

    result = await sensor._raw_to_text([2.0, 0.0])

    assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_negative_change():
    """Test _raw_to_text with negative balance change returns None."""
    sensor = _make_sensor()

    result = await sensor._raw_to_text([1.5, -0.5])

    assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_appends_to_messages():
    """Test raw_to_text appends message to buffer."""
    sensor = _make_sensor()

    assert len(sensor.messages) == 0

    with patch("inputs.plugins.wallet_solana.time.time", return_value=1000.0):
        await sensor.raw_to_text([2.0, 0.5])

    assert len(sensor.messages) == 1
    assert "SOL" in sensor.messages[0].message


# --- Buffer tests ---


def test_formatted_latest_buffer_with_messages():
    """Test formatted_latest_buffer formats and clears messages."""
    sensor = _make_sensor()
    sensor.io_provider = MagicMock()

    sensor.messages = [
        Message(timestamp=1000.0, message="You just received 0.500000000 SOL."),
        Message(timestamp=1001.0, message="You just received 0.300000000 SOL."),
    ]

    result = sensor.formatted_latest_buffer()

    assert result is not None
    assert "WalletSolana" in result
    assert "SOL" in result
    sensor.io_provider.add_input.assert_called_once()
    assert len(sensor.messages) == 0


def test_formatted_latest_buffer_empty():
    """Test formatted_latest_buffer with empty buffer returns None."""
    sensor = _make_sensor()

    result = sensor.formatted_latest_buffer()

    assert result is None
