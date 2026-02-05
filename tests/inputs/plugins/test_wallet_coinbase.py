import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inputs.plugins.wallet_coinbase import Message, WalletCoinbase, WalletCoinbaseConfig


def test_initialization_with_missing_wallet_id():
    """Missing COINBASE_WALLET_ID should fall back to a safe zero state."""
    with patch.dict(os.environ, {}, clear=True):
        wallet = WalletCoinbase(config=WalletCoinbaseConfig())
        assert wallet.wallet is None
        assert wallet.balance == 0.0
        assert wallet.balance_previous == 0.0
        assert wallet.asset_id == "eth"


def test_initialization_with_wallet_fetch_failure():
    """Wallet.fetch failure should be handled gracefully."""
    env = {
        "COINBASE_WALLET_ID": "test_wallet_id",
        "COINBASE_API_KEY": "k",
        "COINBASE_API_SECRET": "s",
    }
    with (
        patch.dict(os.environ, env, clear=True),
        patch("inputs.plugins.wallet_coinbase.Cdp.configure"),
        patch("inputs.plugins.wallet_coinbase.Wallet.fetch") as mock_fetch,
    ):
        mock_fetch.side_effect = Exception("Network error")

        wallet = WalletCoinbase(config=WalletCoinbaseConfig())

        assert wallet.wallet is None
        assert wallet.balance == 0.0
        assert wallet.balance_previous == 0.0


def test_initialization_with_successful_wallet_fetch_default_asset():
    """Successful initialization should read balance using default asset_id 'eth'."""
    mock_wallet = MagicMock()
    mock_wallet.balance.return_value = "1.5"

    env = {
        "COINBASE_WALLET_ID": "test_wallet_id",
        "COINBASE_API_KEY": "k",
        "COINBASE_API_SECRET": "s",
    }
    with (
        patch.dict(os.environ, env, clear=True),
        patch("inputs.plugins.wallet_coinbase.Cdp.configure") as mock_configure,
        patch(
            "inputs.plugins.wallet_coinbase.Wallet.fetch",
            return_value=mock_wallet,
        ),
    ):
        wallet = WalletCoinbase(config=WalletCoinbaseConfig())

        assert wallet.wallet == mock_wallet
        assert wallet.asset_id == "eth"
        assert wallet.balance == 1.5
        assert wallet.balance_previous == 1.5

        mock_configure.assert_called_once_with("k", "s")
        mock_wallet.balance.assert_called_with("eth")


def test_initialization_with_custom_asset_id():
    """Custom asset_id should be respected during initialization."""
    mock_wallet = MagicMock()
    mock_wallet.balance.return_value = "100.0"

    config = WalletCoinbaseConfig(asset_id="btc")

    env = {
        "COINBASE_WALLET_ID": "test_wallet_id",
        "COINBASE_API_KEY": "k",
        "COINBASE_API_SECRET": "s",
    }
    with (
        patch.dict(os.environ, env, clear=True),
        patch("inputs.plugins.wallet_coinbase.Cdp.configure"),
        patch(
            "inputs.plugins.wallet_coinbase.Wallet.fetch",
            return_value=mock_wallet,
        ),
    ):
        wallet = WalletCoinbase(config=config)

        assert wallet.asset_id == "btc"
        assert wallet.balance == 100.0
        assert wallet.balance_previous == 100.0

        mock_wallet.balance.assert_called_with("btc")


def test_initialization_without_api_keys_does_not_call_configure():
    """
    If API key/secret are missing, Cdp.configure should not be called.
    Initialization should still safely proceed (with Wallet.fetch mocked).
    """
    mock_wallet = MagicMock()
    mock_wallet.balance.return_value = "3.0"

    env = {
        "COINBASE_WALLET_ID": "test_wallet_id",
        # Intentionally omit API key/secret
    }
    with (
        patch.dict(os.environ, env, clear=True),
        patch("inputs.plugins.wallet_coinbase.Cdp.configure") as mock_configure,
        patch(
            "inputs.plugins.wallet_coinbase.Wallet.fetch",
            return_value=mock_wallet,
        ),
    ):
        wallet = WalletCoinbase(config=WalletCoinbaseConfig())

        assert wallet.wallet == mock_wallet
        assert wallet.balance == 3.0
        assert wallet.balance_previous == 3.0

        mock_configure.assert_not_called()


@pytest.mark.asyncio
async def test_poll_with_wallet_refresh_failure_returns_zero_delta():
    """_poll should return zero delta if Wallet.fetch fails."""
    env = {
        "COINBASE_WALLET_ID": "test_wallet_id",
        "COINBASE_API_KEY": "k",
        "COINBASE_API_SECRET": "s",
    }
    with (
        patch.dict(os.environ, env, clear=True),
        patch("inputs.plugins.wallet_coinbase.Cdp.configure"),
        patch("inputs.plugins.wallet_coinbase.Wallet.fetch") as mock_fetch,
        patch(
            "inputs.plugins.wallet_coinbase.asyncio.sleep",
            new=AsyncMock(return_value=None),
        ),
    ):
        mock_fetch.side_effect = Exception("Network error")

        wallet = WalletCoinbase(config=WalletCoinbaseConfig())

        result = await wallet._poll()

        assert result == [0.0, 0.0]


@pytest.mark.asyncio
async def test_poll_with_successful_wallet_refresh_calculates_delta():
    """_poll should update balance and compute correct delta on success."""
    mock_wallet = MagicMock()
    mock_wallet.balance.return_value = "2.0"

    env = {
        "COINBASE_WALLET_ID": "test_wallet_id",
        "COINBASE_API_KEY": "k",
        "COINBASE_API_SECRET": "s",
    }
    with (
        patch.dict(os.environ, env, clear=True),
        patch("inputs.plugins.wallet_coinbase.Cdp.configure"),
        patch(
            "inputs.plugins.wallet_coinbase.Wallet.fetch",
            return_value=mock_wallet,
        ),
        patch(
            "inputs.plugins.wallet_coinbase.asyncio.sleep",
            new=AsyncMock(return_value=None),
        ),
    ):
        wallet = WalletCoinbase(config=WalletCoinbaseConfig())
        wallet.balance_previous = 1.5

        result = await wallet._poll()

        assert result == [2.0, 0.5]
        mock_wallet.balance.assert_called_with("eth")


@pytest.mark.asyncio
async def test_raw_to_text_positive_balance_change():
    """_raw_to_text should return Message for positive deltas."""
    with (
        patch.dict(os.environ, {}, clear=True),
        patch("inputs.plugins.wallet_coinbase.time.time", return_value=1234.0),
    ):
        wallet = WalletCoinbase(config=WalletCoinbaseConfig())

        raw_input = [2.0, 0.5]
        result = await wallet._raw_to_text(raw_input)

    assert result is not None
    assert isinstance(result, Message)
    assert result.timestamp == 1234.0
    assert result.message == "0.50000"


@pytest.mark.asyncio
async def test_raw_to_text_zero_balance_change():
    """_raw_to_text should return None for zero deltas."""
    with patch.dict(os.environ, {}, clear=True):
        wallet = WalletCoinbase(config=WalletCoinbaseConfig())

        raw_input = [2.0, 0.0]
        result = await wallet._raw_to_text(raw_input)

    assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_negative_balance_change():
    """_raw_to_text should return None for negative deltas."""
    with patch.dict(os.environ, {}, clear=True):
        wallet = WalletCoinbase(config=WalletCoinbaseConfig())

        raw_input = [2.0, -0.1]
        result = await wallet._raw_to_text(raw_input)

    assert result is None


def test_formatted_latest_buffer_with_multiple_transactions():
    """formatted_latest_buffer should sum messages, write IO, and clear buffer."""
    with patch.dict(os.environ, {}, clear=True):
        wallet = WalletCoinbase(config=WalletCoinbaseConfig())

    wallet.io_provider = MagicMock()

    wallet.messages = [
        Message(timestamp=1000.0, message="0.5"),
        Message(timestamp=1001.0, message="0.3"),
        Message(timestamp=1002.0, message="0.2"),
    ]

    result = wallet.formatted_latest_buffer()

    assert result is not None
    assert "WalletCoinbase INPUT" in result
    assert "You just received 1.00000 ETH." in result

    wallet.io_provider.add_input.assert_called_once()
    assert len(wallet.messages) == 0


def test_formatted_latest_buffer_with_custom_asset_symbol():
    """Custom asset should appear in upper-case in formatted output."""
    config = WalletCoinbaseConfig(asset_id="btc")

    env = {
        "COINBASE_WALLET_ID": "test_wallet_id",
        "COINBASE_API_KEY": "k",
        "COINBASE_API_SECRET": "s",
    }

    mock_wallet = MagicMock()
    mock_wallet.balance.return_value = "0.0"

    with (
        patch.dict(os.environ, env, clear=True),
        patch("inputs.plugins.wallet_coinbase.Cdp.configure"),
        patch(
            "inputs.plugins.wallet_coinbase.Wallet.fetch",
            return_value=mock_wallet,
        ),
    ):
        wallet = WalletCoinbase(config=config)

    wallet.io_provider = MagicMock()

    wallet.messages = [
        Message(timestamp=1000.0, message="10.0"),
    ]

    result = wallet.formatted_latest_buffer()

    assert result is not None
    assert "You just received 10.00000 BTC." in result

    wallet.io_provider.add_input.assert_called_once()
    assert len(wallet.messages) == 0


def test_formatted_latest_buffer_with_empty_buffer():
    """Empty buffer should return None."""
    with patch.dict(os.environ, {}, clear=True):
        wallet = WalletCoinbase(config=WalletCoinbaseConfig())

    result = wallet.formatted_latest_buffer()
    assert result is None


# =============================================================================
# Tests for wallet auto-creation and seed persistence (new functionality)
# =============================================================================


def test_create_wallet_when_no_wallet_id_set():
    """Should create new wallet when COINBASE_WALLET_ID is not set."""
    mock_wallet = MagicMock()
    mock_wallet.id = "new-wallet-id"
    mock_wallet.balance.return_value = "0.0"
    mock_wallet.default_address = MagicMock()
    mock_wallet.default_address.address_id = "0x1234"

    mock_wallet_data = MagicMock()
    mock_wallet_data.wallet_id = "new-wallet-id"
    mock_wallet_data.seed = "abc123seed"
    mock_wallet_data.network_id = "base-sepolia"
    mock_wallet.export_data.return_value = mock_wallet_data

    env = {
        "COINBASE_API_KEY": "k",
        "COINBASE_API_SECRET": "s",
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_file = f.name

    try:
        config = WalletCoinbaseConfig(wallet_seed_file=temp_file)

        with (
            patch.dict(os.environ, env, clear=True),
            patch("inputs.plugins.wallet_coinbase.Cdp.configure"),
            patch(
                "inputs.plugins.wallet_coinbase.Wallet.create",
                return_value=mock_wallet,
            ),
        ):
            wallet = WalletCoinbase(config=config)

            assert wallet.wallet == mock_wallet
            assert wallet.COINBASE_WALLET_ID == "new-wallet-id"

            # Verify seed file was created
            assert os.path.exists(temp_file)

            with open(temp_file, "r") as f:
                saved_data = json.load(f)
            assert saved_data["wallet_id"] == "new-wallet-id"
            assert saved_data["seed"] == "abc123seed"
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def test_load_wallet_from_seed_file():
    """Should load wallet from saved seed file using import_data."""
    mock_wallet = MagicMock()
    mock_wallet.id = "saved-wallet-id"
    mock_wallet.balance.return_value = "5.0"

    env = {
        "COINBASE_API_KEY": "k",
        "COINBASE_API_SECRET": "s",
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(
            {
                "wallet_id": "saved-wallet-id",
                "seed": "saved-seed-hex",
                "network_id": "base-sepolia",
            },
            f,
        )
        temp_file = f.name

    try:
        config = WalletCoinbaseConfig(wallet_seed_file=temp_file)

        with (
            patch.dict(os.environ, env, clear=True),
            patch("inputs.plugins.wallet_coinbase.Cdp.configure"),
            patch(
                "inputs.plugins.wallet_coinbase.Wallet.import_data",
                return_value=mock_wallet,
            ) as mock_import,
        ):
            wallet = WalletCoinbase(config=config)

            assert wallet.wallet == mock_wallet
            mock_import.assert_called_once()


    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def test_load_wallet_with_malformed_seed_file():
    """Should handle malformed seed file gracefully."""
    env = {
        "COINBASE_API_KEY": "k",
        "COINBASE_API_SECRET": "s",
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("not valid json {{{")
        temp_file = f.name

    try:
        config = WalletCoinbaseConfig(wallet_seed_file=temp_file)

        mock_new_wallet = MagicMock()
        mock_new_wallet.id = "new-wallet-id"
        mock_new_wallet.balance.return_value = "0.0"
        mock_new_wallet.default_address = MagicMock()
        mock_new_wallet.default_address.address_id = "0x5678"

        mock_wallet_data = MagicMock()
        mock_wallet_data.wallet_id = "new-wallet-id"
        mock_wallet_data.seed = "new-seed"
        mock_wallet_data.network_id = "base-sepolia"
        mock_new_wallet.export_data.return_value = mock_wallet_data

        with (
            patch.dict(os.environ, env, clear=True),
            patch("inputs.plugins.wallet_coinbase.Cdp.configure"),
            patch(
                "inputs.plugins.wallet_coinbase.Wallet.create",
                return_value=mock_new_wallet,
            ),
        ):
            wallet = WalletCoinbase(config=config)

            # Should create new wallet when seed file is malformed
            assert wallet.wallet == mock_new_wallet
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def test_recreate_wallet_when_deleted_from_server():
    """Should recreate wallet from seed when original was deleted from server."""
    mock_recreated_wallet = MagicMock()
    mock_recreated_wallet.id = "recreated-wallet-id"
    mock_recreated_wallet.balance.return_value = "0.0"
    mock_recreated_wallet.default_address = MagicMock()
    mock_recreated_wallet.default_address.address_id = "0x1234"

    mock_wallet_data = MagicMock()
    mock_wallet_data.wallet_id = "recreated-wallet-id"
    mock_wallet_data.seed = "original-seed"
    mock_wallet_data.network_id = "base-sepolia"
    mock_recreated_wallet.export_data.return_value = mock_wallet_data

    env = {
        "COINBASE_API_KEY": "k",
        "COINBASE_API_SECRET": "s",
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(
            {
                "wallet_id": "deleted-wallet-id",
                "seed": "original-seed",
                "network_id": "base-sepolia",
            },
            f,
        )
        temp_file = f.name

    try:
        config = WalletCoinbaseConfig(wallet_seed_file=temp_file)

        with (
            patch.dict(os.environ, env, clear=True),
            patch("inputs.plugins.wallet_coinbase.Cdp.configure"),
            patch(
                "inputs.plugins.wallet_coinbase.Wallet.import_data",
                side_effect=Exception("Wallet not found on server"),
            ),
            patch(
                "inputs.plugins.wallet_coinbase.Wallet.create_with_seed",
                return_value=mock_recreated_wallet,
            ) as mock_create_with_seed,
        ):
            wallet = WalletCoinbase(config=config)

            # Should have recreated wallet with original seed
            assert wallet.wallet == mock_recreated_wallet
            mock_create_with_seed.assert_called_once_with(
                seed="original-seed",
                network_id="base-sepolia",
            )
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def test_env_wallet_id_takes_priority_over_seed_file():
    """Environment variable COINBASE_WALLET_ID should take priority."""
    mock_env_wallet = MagicMock()
    mock_env_wallet.id = "env-wallet-id"
    mock_env_wallet.balance.return_value = "10.0"

    env = {
        "COINBASE_WALLET_ID": "env-wallet-id",
        "COINBASE_API_KEY": "k",
        "COINBASE_API_SECRET": "s",
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(
            {
                "wallet_id": "file-wallet-id",
                "seed": "file-seed",
                "network_id": "base-sepolia",
            },
            f,
        )
        temp_file = f.name

    try:
        config = WalletCoinbaseConfig(wallet_seed_file=temp_file)

        with (
            patch.dict(os.environ, env, clear=True),
            patch("inputs.plugins.wallet_coinbase.Cdp.configure"),
            patch(
                "inputs.plugins.wallet_coinbase.Wallet.fetch",
                return_value=mock_env_wallet,
            ) as mock_fetch,
        ):
            wallet = WalletCoinbase(config=config)

            # Should use wallet from env, not from file
            assert wallet.wallet == mock_env_wallet
            mock_fetch.assert_called_with("env-wallet-id")
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def test_seed_file_missing_wallet_id():
    """Should handle seed file with missing wallet_id."""
    env = {
        "COINBASE_API_KEY": "k",
        "COINBASE_API_SECRET": "s",
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"seed": "seed-only-no-id"}, f)
        temp_file = f.name

    try:
        config = WalletCoinbaseConfig(wallet_seed_file=temp_file)

        mock_new_wallet = MagicMock()
        mock_new_wallet.id = "new-wallet-id"
        mock_new_wallet.balance.return_value = "0.0"
        mock_new_wallet.default_address = MagicMock()

        mock_wallet_data = MagicMock()
        mock_wallet_data.wallet_id = "new-wallet-id"
        mock_wallet_data.seed = "new-seed"
        mock_wallet_data.network_id = "base-sepolia"
        mock_new_wallet.export_data.return_value = mock_wallet_data

        with (
            patch.dict(os.environ, env, clear=True),
            patch("inputs.plugins.wallet_coinbase.Cdp.configure"),
            patch(
                "inputs.plugins.wallet_coinbase.Wallet.create",
                return_value=mock_new_wallet,
            ),
        ):
            wallet = WalletCoinbase(config=config)

            # Should create new wallet since file is missing wallet_id
            assert wallet.wallet == mock_new_wallet
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def test_custom_network_id_config():
    """Should use custom network_id from config."""
    mock_wallet = MagicMock()
    mock_wallet.id = "mainnet-wallet"
    mock_wallet.balance.return_value = "100.0"
    mock_wallet.default_address = MagicMock()

    mock_wallet_data = MagicMock()
    mock_wallet_data.wallet_id = "mainnet-wallet"
    mock_wallet_data.seed = "mainnet-seed"
    mock_wallet_data.network_id = "base-mainnet"
    mock_wallet.export_data.return_value = mock_wallet_data

    env = {
        "COINBASE_API_KEY": "k",
        "COINBASE_API_SECRET": "s",
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_file = f.name

    try:
        config = WalletCoinbaseConfig(
            network_id="base-mainnet",
            wallet_seed_file=temp_file,
        )

        with (
            patch.dict(os.environ, env, clear=True),
            patch("inputs.plugins.wallet_coinbase.Cdp.configure"),
            patch(
                "inputs.plugins.wallet_coinbase.Wallet.create",
                return_value=mock_wallet,
            ) as mock_create,
        ):
            wallet = WalletCoinbase(config=config)

            mock_create.assert_called_with(network_id="base-mainnet")
            assert wallet.network_id == "base-mainnet"
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)
