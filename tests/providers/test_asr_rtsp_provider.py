from unittest.mock import Mock, patch

import pytest

from providers.asr_rtsp_provider import ASRRTSPProvider


@pytest.fixture
def ws_url():
    return "ws://test.url"


@pytest.fixture
def rtsp_url():
    return "rtsp://localhost:8554/audio"


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    ASRRTSPProvider.reset()  # type: ignore
    yield
    ASRRTSPProvider.reset()  # type: ignore


@pytest.fixture
def mock_dependencies():
    with (
        patch("providers.asr_rtsp_provider.ws.Client") as mock_ws_client,
        patch("providers.asr_rtsp_provider.AudioRTSPInputStream") as mock_audio_stream,
    ):
        # Configure mock instances to avoid blocking
        mock_ws_instance = Mock()
        mock_ws_instance.start = Mock()
        mock_ws_instance.stop = Mock()
        mock_ws_instance.register_message_callback = Mock()
        mock_ws_client.return_value = mock_ws_instance

        mock_audio_instance = Mock()
        mock_audio_instance.start = Mock()
        mock_audio_instance.stop = Mock()
        mock_audio_stream.return_value = mock_audio_instance

        yield mock_ws_client, mock_audio_stream


def test_initialization(ws_url, rtsp_url, mock_dependencies):
    mock_ws_client, mock_audio_stream = mock_dependencies
    provider = ASRRTSPProvider(ws_url, rtsp_url)

    mock_ws_client.assert_called_once_with(url=ws_url)
    mock_audio_stream.assert_called_once()
    assert not provider.running


def test_singleton_pattern(ws_url, rtsp_url, mock_dependencies):
    provider1 = ASRRTSPProvider(ws_url, rtsp_url)
    provider2 = ASRRTSPProvider(ws_url, rtsp_url)
    assert provider1 is provider2


def test_register_message_callback(ws_url, rtsp_url, mock_dependencies):
    mock_ws_client, _ = mock_dependencies
    provider = ASRRTSPProvider(ws_url, rtsp_url)
    callback = Mock()
    provider.register_message_callback(callback)

    mock_ws_client.return_value.register_message_callback.assert_called_once_with(
        callback
    )


def test_register_message_callback_none(ws_url, rtsp_url, mock_dependencies):
    mock_ws_client, _ = mock_dependencies
    provider = ASRRTSPProvider(ws_url, rtsp_url)
    provider.register_message_callback(None)

    mock_ws_client.return_value.register_message_callback.assert_not_called()


def test_start(ws_url, rtsp_url, mock_dependencies):
    mock_ws_client, mock_audio_stream = mock_dependencies
    provider = ASRRTSPProvider(ws_url, rtsp_url)
    provider.start()

    assert provider.running
    mock_ws_client.return_value.start.assert_called_once()
    mock_audio_stream.return_value.start.assert_called_once()

    provider.stop()


def test_start_already_running(ws_url, rtsp_url, mock_dependencies):
    mock_ws_client, mock_audio_stream = mock_dependencies
    provider = ASRRTSPProvider(ws_url, rtsp_url)
    provider.start()

    # Reset mocks
    mock_ws_client.return_value.start.reset_mock()
    mock_audio_stream.return_value.start.reset_mock()

    # Try starting again
    provider.start()

    # Should not call start again
    mock_ws_client.return_value.start.assert_not_called()
    mock_audio_stream.return_value.start.assert_not_called()

    provider.stop()


def test_stop(ws_url, rtsp_url, mock_dependencies):
    mock_ws_client, mock_audio_stream = mock_dependencies
    provider = ASRRTSPProvider(ws_url, rtsp_url)
    provider.start()
    provider.stop()

    assert not provider.running
    mock_audio_stream.return_value.stop.assert_called_once()
    mock_ws_client.return_value.stop.assert_called_once()
