from unittest.mock import Mock, patch

import pytest

from providers.vlm_vila_provider import VLMVilaProvider


@pytest.fixture
def ws_url():
    return "ws://test.url"


@pytest.fixture
def fps():
    return 30


@pytest.fixture(autouse=True)
def reset_singleton():
    VLMVilaProvider._instance = None
    yield


@pytest.fixture
def mock_dependencies():
    with (
        patch("providers.vlm_vila_provider.ws.Client") as mock_ws_client,
        patch("providers.vlm_vila_provider.VideoStream") as mock_video_stream,
    ):
        yield mock_ws_client, mock_video_stream


def test_initialization(ws_url, fps, mock_dependencies):
    mock_ws_client, mock_video_stream = mock_dependencies
    provider = VLMVilaProvider(ws_url, fps=fps)

    mock_ws_client.assert_called_once_with(url=ws_url)
    mock_video_stream.assert_called_once_with(
        provider.ws_client.send_message, fps=fps, device_index=0
    )

    assert not provider.running
    assert provider.ws_client is not None
    assert provider.video_stream is not None


def test_singleton_pattern(ws_url, fps):
    provider1 = VLMVilaProvider(ws_url, fps=fps)
    provider2 = VLMVilaProvider(ws_url, fps=fps)

    assert provider1 is provider2
    assert provider1.ws_client is provider2.ws_client
    assert provider1.video_stream is provider2.video_stream


def test_register_message_callback(ws_url, fps, mock_dependencies):
    provider = VLMVilaProvider(ws_url, fps=fps)
    callback = Mock()

    provider.register_message_callback(callback)
    provider.ws_client.register_message_callback.assert_called_once_with(callback)


def test_start(ws_url, fps, mock_dependencies):
    provider = VLMVilaProvider(ws_url, fps=fps)
    provider.start()

    assert provider.running
    provider.ws_client.start.assert_called_once()
    provider.video_stream.start.assert_called_once()


def test_stop(ws_url, fps, mock_dependencies):
    provider = VLMVilaProvider(ws_url, fps=fps)
    provider.start()
    provider.stop()

    assert not provider.running
    provider.video_stream.stop.assert_called_once()
    provider.ws_client.stop.assert_called_once()
