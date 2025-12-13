from unittest.mock import Mock, patch

import pytest

from providers.asr_provider import ASRProvider


@pytest.fixture
def ws_url():
    return "ws://test.url"


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    ASRProvider.reset()  # type: ignore
    yield
    ASRProvider.reset()  # type: ignore


@pytest.fixture
def mock_dependencies():
    with (
        patch("providers.asr_provider.ws.Client") as mock_ws_client,
        patch("providers.asr_provider.AudioInputStream") as mock_audio_stream,
    ):
        yield mock_ws_client, mock_audio_stream


def test_initialization(ws_url, mock_dependencies):
    mock_ws_client, mock_audio_stream = mock_dependencies
    provider = ASRProvider(ws_url)

    mock_ws_client.assert_called_once_with(url=ws_url)
    mock_audio_stream.assert_called_once()
    assert not provider.running


def test_singleton_pattern(ws_url):
    provider1 = ASRProvider(ws_url)
    provider2 = ASRProvider(ws_url)
    assert provider1 is provider2


def test_register_message_callback(ws_url, mock_dependencies):
    mock_ws_client, mock_audio_stream = mock_dependencies
    provider = ASRProvider(ws_url)
    callback = Mock()
    provider.register_message_callback(callback)

    mock_ws_client.return_value.register_message_callback.assert_called_once_with(
        callback
    )


def test_start(ws_url, mock_dependencies):
    mock_ws_client, mock_audio_stream = mock_dependencies
    provider = ASRProvider(ws_url)
    provider.start()

    assert provider.running
    mock_ws_client.return_value.start.assert_called_once()
    mock_audio_stream.return_value.start.assert_called_once()


def test_stop(ws_url, mock_dependencies):
    mock_ws_client, mock_audio_stream = mock_dependencies
    provider = ASRProvider(ws_url)
    provider.start()
    provider.stop()

    assert not provider.running
    mock_audio_stream.return_value.stop.assert_called_once()
    mock_ws_client.return_value.stop.assert_called_once()
