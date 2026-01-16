import time
from unittest.mock import Mock, patch

import pytest

from inputs.base import Message, SensorConfig
from inputs.plugins.wallet_ethereum import WalletEthereum


@pytest.fixture
def mock_web3():
    with patch("inputs.plugins.wallet_ethereum.Web3") as mock:
        mock_instance = Mock()
        mock_http_provider = Mock()
        mock.HTTPProvider.return_value = mock_http_provider
        mock.return_value = mock_instance
        mock_instance.is_connected.return_value = True
        mock_instance.from_wei.return_value = 10.0
        mock_instance.eth = Mock()
        yield mock_instance


@pytest.fixture
def mock_io_provider():
    with patch("inputs.plugins.wallet_ethereum.IOProvider") as mock:
        mock_instance = Mock()
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def wallet_eth(mock_web3, mock_io_provider):
    with patch.dict("os.environ", {"ETH_ADDRESS": "0xTestAddress"}):
        return WalletEthereum(config=SensorConfig())


def test_init(wallet_eth, mock_web3, mock_io_provider):
    assert wallet_eth.ETH_balance == 0
    assert wallet_eth.ETH_balance_previous == 0
    assert isinstance(wallet_eth.messages, list)
    assert wallet_eth.ACCOUNT_ADDRESS == "0xTestAddress"
    assert wallet_eth.web3 is not None
    mock_web3.is_connected.assert_called_once()


def test_init_connection_failed(mock_web3):
    mock_web3.is_connected.return_value = False
    with pytest.raises(Exception) as exc_info:
        with patch.dict("os.environ", {"ETH_ADDRESS": "0xTestAddress"}):
            WalletEthereum(config=SensorConfig())
    assert str(exc_info.value) == "Failed to connect to Ethereum"


@pytest.mark.asyncio
async def test_poll(wallet_eth, mock_web3):
    mock_web3.eth.block_number = 12345
    mock_web3.eth.get_balance.return_value = 1000000000000000000  # 1 ETH in Wei
    mock_web3.from_wei.return_value = 1.0

    result = await wallet_eth._poll()
    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], float)  # current balance
    assert isinstance(result[1], float)  # balance change

    mock_web3.eth.get_balance.assert_called_once_with(wallet_eth.ACCOUNT_ADDRESS)
    mock_web3.from_wei.assert_called_once()


@pytest.mark.asyncio
async def test_raw_to_text_conversion_balance_change(wallet_eth):
    raw_input = [10.0, 1.0]  # balance and positive change
    result = await wallet_eth._raw_to_text(raw_input)

    assert isinstance(result, Message)
    assert "received 1.000 ETH" in result.message
    assert isinstance(result.timestamp, float)


@pytest.mark.asyncio
async def test_raw_to_text_conversion_no_change(wallet_eth):
    raw_input = [10.0, 0.0]  # balance and no change
    result = await wallet_eth._raw_to_text(raw_input)

    assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_buffer_management(wallet_eth):
    raw_input = [10.0, 1.0]
    await wallet_eth.raw_to_text(raw_input)
    assert len(wallet_eth.messages) == 1
    assert isinstance(wallet_eth.messages[0], Message)


def test_formatted_latest_buffer_with_message(wallet_eth):
    current_time = time.time()
    test_message = Message(timestamp=current_time, message="test balance update")
    wallet_eth.messages = [test_message]

    result = wallet_eth.formatted_latest_buffer()

    assert result is not None
    assert "WalletEthereum INPUT" in result
    assert "START" in result
    assert "END" in result
    assert len(wallet_eth.messages) == 0
    wallet_eth.io_provider.add_input.assert_called_once_with(
        "WalletEthereum", "test balance update", current_time
    )


def test_formatted_latest_buffer_empty(wallet_eth):
    assert wallet_eth.formatted_latest_buffer() is None
    wallet_eth.io_provider.add_input.assert_not_called()
