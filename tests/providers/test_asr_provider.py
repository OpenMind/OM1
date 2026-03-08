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


def test_initialization_with_all_parameters(mock_dependencies):
    mock_ws_client, mock_audio_stream = mock_dependencies

    provider = ASRProvider(
        ws_url="ws://custom.url",
        device_id=5,
        microphone_name="test_mic",
        rate=16000,
        chunk=8192,
        language_code="zh-CN",
        remote_input=True,
        enable_tts_interrupt=True,
    )

    mock_ws_client.assert_called_once_with(url="ws://custom.url")
    mock_audio_stream.assert_called_once()

    call_kwargs = mock_audio_stream.call_args[1]
    assert call_kwargs["rate"] == 16000
    assert call_kwargs["chunk"] == 8192
    assert call_kwargs["device"] == 5
    assert call_kwargs["device_name"] == "test_mic"
    assert call_kwargs["language_code"] == "zh-CN"
    assert call_kwargs["remote_input"] is True
    assert call_kwargs["enable_tts_interrupt"] is True
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


def test_register_message_callback_with_none(ws_url, mock_dependencies):
    mock_ws_client, mock_audio_stream = mock_dependencies
    provider = ASRProvider(ws_url)
    provider.register_message_callback(None)

    # Should not call register on ws_client when callback is None
    mock_ws_client.return_value.register_message_callback.assert_not_called()


def test_unregister_message_callback_matching(ws_url, mock_dependencies):
    mock_ws_client, mock_audio_stream = mock_dependencies
    provider = ASRProvider(ws_url)
    callback = Mock()

    mock_ws_client.return_value.message_callback = callback

    provider.unregister_message_callback(callback)

    assert mock_ws_client.return_value.message_callback is None


def test_unregister_message_callback_not_matching(ws_url, mock_dependencies):
    mock_ws_client, mock_audio_stream = mock_dependencies
    provider = ASRProvider(ws_url)
    callback1 = Mock()
    callback2 = Mock()

    mock_ws_client.return_value.message_callback = callback1

    provider.unregister_message_callback(callback2)

    assert mock_ws_client.return_value.message_callback is callback1


def test_start(ws_url, mock_dependencies):
    mock_ws_client, mock_audio_stream = mock_dependencies
    provider = ASRProvider(ws_url)
    provider.start()

    assert provider.running
    mock_ws_client.return_value.start.assert_called_once()
    mock_audio_stream.return_value.start.assert_called_once()


def test_start_when_already_running(ws_url, mock_dependencies):
    mock_ws_client, mock_audio_stream = mock_dependencies
    provider = ASRProvider(ws_url)

    provider.start()
    assert provider.running

    provider.start()

    assert mock_ws_client.return_value.start.call_count == 1
    assert mock_audio_stream.return_value.start.call_count == 1


def test_stop(ws_url, mock_dependencies):
    mock_ws_client, mock_audio_stream = mock_dependencies
    provider = ASRProvider(ws_url)
    provider.start()
    provider.stop()

    assert not provider.running
    mock_audio_stream.return_value.stop.assert_called_once()
    mock_ws_client.return_value.stop.assert_called_once()


def test_stop_when_not_running(ws_url, mock_dependencies):
    mock_ws_client, mock_audio_stream = mock_dependencies
    provider = ASRProvider(ws_url)

    provider.stop()

    assert not provider.running
    mock_audio_stream.return_value.stop.assert_called_once()
    mock_ws_client.return_value.stop.assert_called_once()
